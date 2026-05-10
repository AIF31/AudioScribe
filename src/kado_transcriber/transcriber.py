from pathlib import Path

from pydantic import BaseModel

from kado_transcriber.config import Settings


class TranscriptSegment(BaseModel):
    id: int
    start: float
    end: float
    text: str


class TranscriptResult(BaseModel):
    source_file: str
    source_sha256: str | None = None
    language: str | None = None
    language_probability: float | None = None
    duration: float | None = None
    model_name: str
    device: str
    compute_type: str
    batch_size: int
    segments: list[TranscriptSegment]
    full_text: str


class FasterWhisperTranscriber:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.runtime_settings = settings
        self.fallback_reason: str | None = None
        self.model = None
        self.pipeline = None

        try:
            self._load_model(self.runtime_settings)
        except Exception as exc:
            fallback_settings = _cpu_fallback_settings(settings, exc)
            if fallback_settings is None:
                raise RuntimeError(_model_load_error_message(settings, exc)) from exc
            self.runtime_settings = fallback_settings
            self.fallback_reason = str(exc)
            self._load_model(self.runtime_settings)

    def _load_model(self, settings: Settings) -> None:
        from faster_whisper import BatchedInferencePipeline, WhisperModel

        self.model = WhisperModel(
            settings.whisper_model_name,
            device=settings.whisper_device,
            compute_type=settings.whisper_compute_type,
            use_auth_token=settings.hf_token,
        )
        self.pipeline = (
            BatchedInferencePipeline(model=self.model)
            if settings.whisper_batch_size > 1
            else None
        )

    def transcribe_file(
        self,
        audio_path: Path,
        source_sha256: str | None = None,
        source_file: Path | None = None,
    ) -> TranscriptResult:
        transcribe_target = self.pipeline if self.pipeline is not None else self.model
        kwargs = {
            "language": self.runtime_settings.whisper_language,
            "task": self.runtime_settings.whisper_task,
            "beam_size": self.runtime_settings.whisper_beam_size,
            "vad_filter": self.runtime_settings.whisper_vad_filter,
            "vad_parameters": {
                "min_silence_duration_ms": self.runtime_settings.whisper_min_silence_duration_ms,
            },
            "condition_on_previous_text": self.runtime_settings.whisper_condition_on_previous_text,
            "initial_prompt": self.runtime_settings.whisper_initial_prompt,
        }
        if self.pipeline is not None:
            kwargs["batch_size"] = self.runtime_settings.whisper_batch_size

        segments, info = transcribe_target.transcribe(str(audio_path), **kwargs)
        transcript_segments = [
            TranscriptSegment(
                id=index,
                start=float(segment.start),
                end=float(segment.end),
                text=segment.text.strip(),
            )
            for index, segment in enumerate(segments, start=1)
        ]
        full_text = "\n".join(segment.text for segment in transcript_segments if segment.text)

        return TranscriptResult(
            source_file=str(source_file or audio_path),
            source_sha256=source_sha256,
            language=getattr(info, "language", None),
            language_probability=getattr(info, "language_probability", None),
            duration=getattr(info, "duration", None),
            model_name=self.runtime_settings.whisper_model_name,
            device=self.runtime_settings.whisper_device,
            compute_type=self.runtime_settings.whisper_compute_type,
            batch_size=self.runtime_settings.whisper_batch_size,
            segments=transcript_segments,
            full_text=full_text,
        )


def _cpu_fallback_settings(settings: Settings, exc: Exception) -> Settings | None:
    if settings.whisper_device != "cuda" or not settings.whisper_allow_cpu_fallback:
        return None
    if not _is_cuda_init_error(exc):
        return None
    return settings.model_copy(
        update={
            "whisper_device": "cpu",
            "whisper_compute_type": "int8",
            "whisper_batch_size": 1,
        }
    )


def _is_cuda_init_error(exc: Exception) -> bool:
    message = str(exc).lower()
    patterns = (
        "cuda failed with error",
        "cuda driver version is insufficient",
        "failed to initialize nvml",
        "gpu access blocked",
        "cuda runtime version",
        "no cuda-capable device",
        "cuda error",
    )
    return any(pattern in message for pattern in patterns)


def _model_load_error_message(settings: Settings, exc: Exception) -> str:
    if settings.whisper_device != "cuda":
        return f"Failed to load faster-whisper model: {exc}"
    return (
        f"Failed to load faster-whisper CUDA model: {exc}\n\n"
        "Suggested checks:\n"
        "- Run nvidia-smi.\n"
        "- Run: source scripts/setup_cuda_env.sh\n"
        "- Confirm nvidia-cublas-cu12 and nvidia-cudnn-cu12==9.* are installed.\n"
        "- Try WHISPER_COMPUTE_TYPE=int8_float16.\n"
        "- Reduce WHISPER_BATCH_SIZE to 4, 2, or 1.\n"
        "- Automatic CPU fallback can be disabled with WHISPER_ALLOW_CPU_FALLBACK=false.\n"
        "- As a final fallback use WHISPER_DEVICE=cpu, WHISPER_COMPUTE_TYPE=int8, "
        "WHISPER_BATCH_SIZE=1."
    )
