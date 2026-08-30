from pathlib import Path
import base64
import json

from pydantic import BaseModel
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from audio_transcriber.config import (
    TRANSCRIPTION_BACKEND_FASTER_WHISPER,
    TRANSCRIPTION_BACKEND_OPENAI_WHISPER,
    TRANSCRIPTION_BACKEND_OPENAI_REALTIME,
    Settings,
)

OPENAI_MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024


class TranscriptSegment(BaseModel):
    id: int
    start: float
    end: float
    text: str


class TranscriptResult(BaseModel):
    source_file: str
    source_sha256: str | None = None
    transcription_backend: str = TRANSCRIPTION_BACKEND_FASTER_WHISPER
    language: str | None = None
    language_probability: float | None = None
    duration: float | None = None
    model_name: str
    accelerator: str | None = None
    device: str
    compute_type: str
    batch_size: int
    segments: list[TranscriptSegment]
    full_text: str


class TranscriberProtocol:
    def transcribe_file(
        self,
        audio_path: Path,
        source_sha256: str | None = None,
        source_file: Path | None = None,
    ) -> TranscriptResult:
        raise NotImplementedError


def create_transcriber(settings: Settings) -> TranscriberProtocol:
    if settings.transcription_backend == TRANSCRIPTION_BACKEND_OPENAI_WHISPER:
        return OpenAIWhisperTranscriber(settings)
    if settings.transcription_backend == TRANSCRIPTION_BACKEND_OPENAI_REALTIME:
        return OpenAIRealtimeWhisperTranscriber(settings)
    return FasterWhisperTranscriber(settings)


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
            device=settings.effective_whisper_device,
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
            transcription_backend=TRANSCRIPTION_BACKEND_FASTER_WHISPER,
            language=getattr(info, "language", None),
            language_probability=getattr(info, "language_probability", None),
            duration=getattr(info, "duration", None),
            model_name=self.runtime_settings.whisper_model_name,
            accelerator=self.runtime_settings.runtime_accelerator_label,
            device=self.runtime_settings.whisper_device,
            compute_type=self.runtime_settings.whisper_compute_type,
            batch_size=self.runtime_settings.whisper_batch_size,
            segments=transcript_segments,
            full_text=full_text,
        )


class OpenAIWhisperTranscriber:
    def __init__(self, settings: Settings):
        self.settings = settings
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is required when using openai-whisper")

    def transcribe_file(
        self,
        audio_path: Path,
        source_sha256: str | None = None,
        source_file: Path | None = None,
    ) -> TranscriptResult:
        from openai import OpenAI

        _validate_openai_audio_size(audio_path)

        task = self.settings.whisper_task.strip().lower()
        if task not in {"transcribe", "translate"}:
            raise ValueError(
                "WHISPER_TASK must be 'transcribe' or 'translate' when using openai-whisper"
            )

        client = OpenAI(api_key=self.settings.openai_api_key)
        kwargs = {
            "model": self.settings.openai_whisper_model,
            "response_format": "verbose_json",
        }
        if task == "transcribe" and self.settings.whisper_language:
            kwargs["language"] = self.settings.whisper_language
        if self.settings.whisper_initial_prompt:
            kwargs["prompt"] = self.settings.whisper_initial_prompt

        try:
            transcription = _create_openai_audio_transcription(client, audio_path, task, kwargs)
        except Exception as exc:
            raise RuntimeError(_openai_error_message(exc)) from exc

        data = (
            transcription.model_dump()
            if hasattr(transcription, "model_dump")
            else dict(transcription)
        )
        raw_segments = data.get("segments") or []
        transcript_segments = [
            TranscriptSegment(
                id=index,
                start=float(segment.get("start", 0.0)),
                end=float(segment.get("end", 0.0)),
                text=str(segment.get("text", "")).strip(),
            )
            for index, segment in enumerate(raw_segments, start=1)
        ]
        full_text = str(data.get("text") or "").strip()
        if not transcript_segments and full_text:
            transcript_segments = [TranscriptSegment(id=1, start=0.0, end=0.0, text=full_text)]

        return TranscriptResult(
            source_file=str(source_file or audio_path),
            source_sha256=source_sha256,
            transcription_backend=TRANSCRIPTION_BACKEND_OPENAI_WHISPER,
            language=data.get("language") or self.settings.whisper_language,
            language_probability=None,
            duration=data.get("duration"),
            model_name=self.settings.openai_whisper_model,
            accelerator="openai",
            device="openai",
            compute_type="api",
            batch_size=1,
            segments=transcript_segments,
            full_text=full_text,
        )


class OpenAIRealtimeWhisperTranscriber:
    def __init__(self, settings: Settings):
        self.settings = settings
        if not settings.openai_api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is required when using openai-realtime-whisper"
            )

    def transcribe_file(
        self,
        audio_path: Path,
        source_sha256: str | None = None,
        source_file: Path | None = None,
    ) -> TranscriptResult:
        import websocket
        from websocket import WebSocketTimeoutException

        transcript_parts: list[str] = []
        ws = websocket.create_connection(
            self.settings.openai_realtime_url,
            header=[f"Authorization: Bearer {self.settings.openai_api_key}"],
            timeout=self.settings.openai_realtime_timeout_seconds,
        )
        if hasattr(ws, "settimeout"):
            ws.settimeout(self.settings.openai_realtime_timeout_seconds)
        try:
            ws.send(json.dumps(self._session_update_event()))
            for chunk in _decode_audio_pcm_chunks(
                audio_path,
                self.settings.openai_realtime_audio_rate,
            ):
                ws.send(
                    json.dumps(
                        {
                            "type": "input_audio_buffer.append",
                            "audio": base64.b64encode(chunk).decode("ascii"),
                        }
                    )
                )
            ws.send(json.dumps({"type": "input_audio_buffer.commit"}))

            while True:
                try:
                    event = json.loads(ws.recv())
                except Exception as exc:
                    if transcript_parts and isinstance(exc, WebSocketTimeoutException):
                        break
                    raise
                event_type = event.get("type")
                if event_type == "conversation.item.input_audio_transcription.completed":
                    transcript_parts.append(event.get("transcript", ""))
                    continue
                if event_type == "error":
                    error = event.get("error") or {}
                    message = error.get("message") or str(error)
                    raise RuntimeError(f"OpenAI realtime transcription failed: {message}")
        finally:
            ws.close()

        full_text = "\n".join(part.strip() for part in transcript_parts if part.strip())
        return TranscriptResult(
            source_file=str(source_file or audio_path),
            source_sha256=source_sha256,
            transcription_backend=TRANSCRIPTION_BACKEND_OPENAI_REALTIME,
            language=self.settings.whisper_language,
            language_probability=None,
            duration=None,
            model_name=self.settings.openai_realtime_model,
            accelerator="openai",
            device="openai",
            compute_type="realtime",
            batch_size=1,
            segments=[
                TranscriptSegment(
                    id=1,
                    start=0.0,
                    end=0.0,
                    text=full_text,
                )
            ]
            if full_text
            else [],
            full_text=full_text,
        )

    def _session_update_event(self) -> dict:
        noise_reduction = (
            None
            if self.settings.openai_realtime_noise_reduction.lower() in {"none", "null", "false"}
            else {"type": self.settings.openai_realtime_noise_reduction}
        )
        turn_detection = (
            None
            if self.settings.openai_realtime_turn_detection.lower()
            in {"none", "null", "false"}
            else {
                "type": self.settings.openai_realtime_turn_detection,
                "threshold": 0.5,
                "prefix_padding_ms": 300,
                "silence_duration_ms": self.settings.whisper_min_silence_duration_ms,
            }
        )
        return {
            "type": "session.update",
            "session": {
                "type": "transcription",
                "audio": {
                    "input": {
                        "format": {
                            "type": "audio/pcm",
                            "rate": self.settings.openai_realtime_audio_rate,
                        },
                        "transcription": {
                            "model": self.settings.openai_realtime_model,
                            **(
                                {"language": self.settings.whisper_language}
                                if self.settings.whisper_language
                                else {}
                            ),
                        },
                        "turn_detection": turn_detection,
                        "noise_reduction": noise_reduction,
                    }
                },
            },
            "include": ["item.input_audio_transcription.logprobs"],
        }


def _validate_openai_audio_size(audio_path: Path) -> None:
    file_size = audio_path.stat().st_size
    if file_size <= OPENAI_MAX_FILE_SIZE_BYTES:
        return

    file_size_mb = file_size / (1024 * 1024)
    max_size_mb = OPENAI_MAX_FILE_SIZE_BYTES // (1024 * 1024)
    raise ValueError(
        f"File size ({file_size_mb:.1f} MB) exceeds the OpenAI Audio API limit of "
        f"{max_size_mb} MB. Use the local faster-whisper backend or compress/split the audio."
    )


def _is_retryable_openai_error(exc: BaseException) -> bool:
    status_code = getattr(exc, "status_code", None)
    if status_code in {408, 409, 429, 500, 502, 503, 504}:
        return True
    return exc.__class__.__name__ in {
        "APIConnectionError",
        "APITimeoutError",
        "RateLimitError",
        "InternalServerError",
    }


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception(_is_retryable_openai_error),
    reraise=True,
)
def _create_openai_audio_transcription(client, audio_path: Path, task: str, kwargs: dict):
    audio_endpoint = (
        client.audio.translations
        if task == "translate"
        else client.audio.transcriptions
    )
    with audio_path.open("rb") as audio_file:
        return audio_endpoint.create(
            file=audio_file,
            **kwargs,
        )


def _openai_error_message(exc: Exception) -> str:
    status_code = getattr(exc, "status_code", None)
    if status_code == 401 or exc.__class__.__name__ == "AuthenticationError":
        return "OpenAI transcription failed: invalid API key. Check OPENAI_API_KEY."
    if status_code == 413:
        return (
            "OpenAI transcription failed: audio file is too large for the API. "
            "Use the local faster-whisper backend or compress/split the audio."
        )
    if status_code == 429 or exc.__class__.__name__ == "RateLimitError":
        return "OpenAI transcription failed: rate limit exceeded. Try again later."
    if status_code == 503:
        return "OpenAI transcription failed: service unavailable. Try again later."
    if exc.__class__.__name__ == "APITimeoutError":
        return "OpenAI transcription failed: request timed out. Try again later."
    if exc.__class__.__name__ == "APIConnectionError":
        return "OpenAI transcription failed: network connection error. Check your connection."
    return f"OpenAI transcription failed: {exc}"


def _decode_audio_pcm_chunks(audio_path: Path, rate: int, chunk_size: int = 32_000):
    import av

    container = av.open(str(audio_path))
    try:
        audio_stream = next(stream for stream in container.streams if stream.type == "audio")
        resampler = av.audio.resampler.AudioResampler(
            format="s16",
            layout="mono",
            rate=rate,
        )
        pending = bytearray()
        for packet in container.demux(audio_stream):
            for frame in packet.decode():
                for resampled in resampler.resample(frame):
                    pending.extend(bytes(resampled.planes[0]))
                    while len(pending) >= chunk_size:
                        yield bytes(pending[:chunk_size])
                        del pending[:chunk_size]
        for resampled in resampler.resample(None):
            pending.extend(bytes(resampled.planes[0]))
        if pending:
            yield bytes(pending)
    finally:
        container.close()


def _cpu_fallback_settings(settings: Settings, exc: Exception) -> Settings | None:
    if settings.effective_whisper_device != "cuda" or not settings.whisper_allow_cpu_fallback:
        return None
    if not _is_gpu_init_error(exc):
        return None
    return settings.model_copy(
        update={
            "whisper_accelerator": "cpu",
            "whisper_device": "cpu",
            "whisper_compute_type": "int8",
            "whisper_batch_size": 1,
        }
    )


def _is_gpu_init_error(exc: Exception) -> bool:
    message = str(exc).lower()
    patterns = (
        "cuda failed with error",
        "cuda driver version is insufficient",
        "failed to initialize nvml",
        "gpu access blocked",
        "cuda runtime version",
        "no cuda-capable device",
        "cuda error",
        "hip",
        "rocm",
        "hipblas",
        "rocblas",
        "hsa",
        "gpu memory access fault",
    )
    return any(pattern in message for pattern in patterns)


def _model_load_error_message(settings: Settings, exc: Exception) -> str:
    if settings.effective_whisper_device != "cuda":
        return f"Failed to load faster-whisper model: {exc}"

    if settings.runtime_accelerator_label == "rocm":
        return (
            f"Failed to load faster-whisper ROCm/HIP model: {exc}\n\n"
            "Suggested checks:\n"
            "- Confirm your AMD GPU is supported by your installed ROCm/HIP SDK version.\n"
            "- Run rocminfo and rocm-smi when available.\n"
            "- Confirm a ROCm/HIP-enabled CTranslate2 wheel or source build is installed.\n"
            "- Do not install AudioScribe with the cuda extra for ROCm.\n"
            "- Keep WHISPER_DEVICE=cuda; CTranslate2 uses this device string for GPU execution.\n"
            "- Try WHISPER_COMPUTE_TYPE=int8_float16 or float16.\n"
            "- Reduce WHISPER_BATCH_SIZE to 4, 2, or 1.\n"
            "- Automatic CPU fallback can be disabled with WHISPER_ALLOW_CPU_FALLBACK=false.\n"
            "- As a final fallback use WHISPER_ACCELERATOR=cpu, WHISPER_DEVICE=cpu, "
            "WHISPER_COMPUTE_TYPE=int8, WHISPER_BATCH_SIZE=1."
        )

    return (
        f"Failed to load faster-whisper CUDA model: {exc}\n\n"
        "Suggested checks:\n"
        "- Run nvidia-smi.\n"
        "- Run: source scripts/setup_cuda_env.sh\n"
        "- Confirm nvidia-cublas-cu12 and nvidia-cudnn-cu12==9.* are installed.\n"
        "- Try WHISPER_COMPUTE_TYPE=int8_float16.\n"
        "- Reduce WHISPER_BATCH_SIZE to 4, 2, or 1.\n"
        "- Automatic CPU fallback can be disabled with WHISPER_ALLOW_CPU_FALLBACK=false.\n"
        "- As a final fallback use WHISPER_ACCELERATOR=cpu, WHISPER_DEVICE=cpu, "
        "WHISPER_COMPUTE_TYPE=int8, WHISPER_BATCH_SIZE=1."
    )
