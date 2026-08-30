from pathlib import Path
import os

from dotenv import load_dotenv
from pydantic import BaseModel, Field, model_validator

load_dotenv()

TRANSCRIPTION_BACKEND_FASTER_WHISPER = "faster-whisper"
TRANSCRIPTION_BACKEND_OPENAI_WHISPER = "openai-whisper"
TRANSCRIPTION_BACKEND_OPENAI_REALTIME = "openai-realtime-whisper"
TRANSCRIPTION_BACKENDS = {
    TRANSCRIPTION_BACKEND_FASTER_WHISPER,
    TRANSCRIPTION_BACKEND_OPENAI_WHISPER,
    TRANSCRIPTION_BACKEND_OPENAI_REALTIME,
}

WHISPER_ACCELERATOR_AUTO = "auto"
WHISPER_ACCELERATOR_CPU = "cpu"
WHISPER_ACCELERATOR_CUDA = "cuda"
WHISPER_ACCELERATOR_ROCM = "rocm"
WHISPER_ACCELERATORS = {
    WHISPER_ACCELERATOR_AUTO,
    WHISPER_ACCELERATOR_CPU,
    WHISPER_ACCELERATOR_CUDA,
    WHISPER_ACCELERATOR_ROCM,
}


def env_bool(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "y", "on"}


def normalize_language(raw: str | None) -> str | None:
    """Map WHISPER_LANGUAGE values to a faster-whisper language code or None.

    Empty strings and 'auto' mean automatic language detection (None);
    anything else is returned as a stripped lowercase language code.
    """
    if raw is None:
        return None
    value = raw.strip().lower()
    return None if value in {"", "auto"} else value


class Settings(BaseModel):
    transcription_backend: str = Field(
        default_factory=lambda: os.getenv(
            "TRANSCRIPTION_BACKEND",
            TRANSCRIPTION_BACKEND_FASTER_WHISPER,
        )
    )
    whisper_model_name: str = Field(
        default_factory=lambda: os.getenv("WHISPER_MODEL_NAME", "large-v3")
    )
    whisper_accelerator: str = Field(
        default_factory=lambda: os.getenv(
            "WHISPER_ACCELERATOR",
            WHISPER_ACCELERATOR_AUTO,
        )
        .strip()
        .lower()
    )
    whisper_device: str = Field(default_factory=lambda: os.getenv("WHISPER_DEVICE", "cuda"))
    whisper_compute_type: str = Field(
        default_factory=lambda: os.getenv("WHISPER_COMPUTE_TYPE", "float16")
    )
    whisper_language: str | None = Field(
        default_factory=lambda: normalize_language(os.getenv("WHISPER_LANGUAGE", "auto"))
    )
    whisper_task: str = Field(default_factory=lambda: os.getenv("WHISPER_TASK", "transcribe"))
    whisper_beam_size: int = Field(default_factory=lambda: int(os.getenv("WHISPER_BEAM_SIZE", "5")))
    whisper_batch_size: int = Field(
        default_factory=lambda: int(os.getenv("WHISPER_BATCH_SIZE", "8"))
    )
    whisper_allow_cpu_fallback: bool = Field(
        default_factory=lambda: env_bool("WHISPER_ALLOW_CPU_FALLBACK", "true")
    )
    whisper_vad_filter: bool = Field(
        default_factory=lambda: env_bool("WHISPER_VAD_FILTER", "true")
    )
    whisper_min_silence_duration_ms: int = Field(
        default_factory=lambda: int(os.getenv("WHISPER_MIN_SILENCE_DURATION_MS", "500"))
    )
    whisper_condition_on_previous_text: bool = Field(
        default_factory=lambda: env_bool("WHISPER_CONDITION_ON_PREVIOUS_TEXT", "false")
    )
    whisper_initial_prompt: str | None = Field(
        default_factory=lambda: os.getenv("WHISPER_INITIAL_PROMPT") or None
    )
    hf_token: str | None = Field(default_factory=lambda: os.getenv("HF_TOKEN") or None)
    diarization_model: str = Field(
        default_factory=lambda: os.getenv(
            "DIARIZATION_MODEL", "pyannote/speaker-diarization-3.1"
        )
    )
    diarization_device: str = Field(
        default_factory=lambda: os.getenv("DIARIZATION_DEVICE", "cuda")
    )
    openai_api_key: str | None = Field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY") or None
    )
    openai_whisper_model: str = Field(
        default_factory=lambda: os.getenv("OPENAI_WHISPER_MODEL", "whisper-1")
    )
    openai_realtime_model: str = Field(
        default_factory=lambda: os.getenv("OPENAI_REALTIME_MODEL", "gpt-realtime-whisper")
    )
    openai_realtime_url: str = Field(
        default_factory=lambda: os.getenv(
            "OPENAI_REALTIME_URL",
            "wss://api.openai.com/v1/realtime?intent=transcription",
        )
    )
    openai_realtime_audio_rate: int = Field(
        default_factory=lambda: int(os.getenv("OPENAI_REALTIME_AUDIO_RATE", "24000"))
    )
    openai_realtime_turn_detection: str = Field(
        default_factory=lambda: os.getenv("OPENAI_REALTIME_TURN_DETECTION", "server_vad")
    )
    openai_realtime_noise_reduction: str = Field(
        default_factory=lambda: os.getenv("OPENAI_REALTIME_NOISE_REDUCTION", "near_field")
    )
    openai_realtime_timeout_seconds: int = Field(
        default_factory=lambda: int(os.getenv("OPENAI_REALTIME_TIMEOUT_SECONDS", "120"))
    )

    input_audio_dir: Path = Field(
        default_factory=lambda: Path(os.getenv("INPUT_AUDIO_DIR", "./data/audio_raw"))
    )
    transcripts_output_dir: Path = Field(
        default_factory=lambda: Path(os.getenv("TRANSCRIPTS_OUTPUT_DIR", "./data/transcripts"))
    )

    skip_existing: bool = Field(default_factory=lambda: env_bool("SKIP_EXISTING", "true"))
    log_level: str = Field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))

    @model_validator(mode="after")
    def validate_backend(self) -> "Settings":
        if self.transcription_backend not in TRANSCRIPTION_BACKENDS:
            valid = ", ".join(sorted(TRANSCRIPTION_BACKENDS))
            raise ValueError(f"TRANSCRIPTION_BACKEND must be one of: {valid}")
        if self.whisper_accelerator not in WHISPER_ACCELERATORS:
            valid = ", ".join(sorted(WHISPER_ACCELERATORS))
            raise ValueError(f"WHISPER_ACCELERATOR must be one of: {valid}")
        if self.transcription_backend == TRANSCRIPTION_BACKEND_FASTER_WHISPER:
            if (
                self.whisper_accelerator == WHISPER_ACCELERATOR_CPU
                and self.whisper_device != "cpu"
            ):
                raise ValueError("WHISPER_ACCELERATOR=cpu requires WHISPER_DEVICE=cpu")
            if (
                self.whisper_accelerator
                in {WHISPER_ACCELERATOR_CUDA, WHISPER_ACCELERATOR_ROCM}
                and self.whisper_device != "cuda"
            ):
                raise ValueError(
                    f"WHISPER_ACCELERATOR={self.whisper_accelerator} requires "
                    "WHISPER_DEVICE=cuda because CTranslate2 uses device='cuda' "
                    "for GPU execution."
                )
        if (
            self.transcription_backend
            in {TRANSCRIPTION_BACKEND_OPENAI_WHISPER, TRANSCRIPTION_BACKEND_OPENAI_REALTIME}
            and not self.openai_api_key
        ):
            raise ValueError(
                "OPENAI_API_KEY is required when "
                f"TRANSCRIPTION_BACKEND={self.transcription_backend}"
            )
        return self

    @property
    def effective_whisper_device(self) -> str:
        """Device string passed to faster-whisper/CTranslate2."""
        if self.whisper_accelerator == WHISPER_ACCELERATOR_CPU:
            return "cpu"
        if self.whisper_accelerator in {
            WHISPER_ACCELERATOR_CUDA,
            WHISPER_ACCELERATOR_ROCM,
        }:
            return "cuda"
        return self.whisper_device

    @property
    def runtime_accelerator_label(self) -> str:
        """Human-facing accelerator label for CLI output and metadata."""
        if self.whisper_accelerator != WHISPER_ACCELERATOR_AUTO:
            return self.whisper_accelerator
        if self.whisper_device == "cpu":
            return WHISPER_ACCELERATOR_CPU
        return WHISPER_ACCELERATOR_CUDA


def get_settings() -> Settings:
    return Settings()
