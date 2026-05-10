from pathlib import Path
import os

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()


def env_bool(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "y", "on"}


class Settings(BaseModel):
    whisper_model_name: str = Field(
        default_factory=lambda: os.getenv("WHISPER_MODEL_NAME", "large-v3")
    )
    whisper_device: str = Field(default_factory=lambda: os.getenv("WHISPER_DEVICE", "cuda"))
    whisper_compute_type: str = Field(
        default_factory=lambda: os.getenv("WHISPER_COMPUTE_TYPE", "float16")
    )
    whisper_language: str = Field(default_factory=lambda: os.getenv("WHISPER_LANGUAGE", "es"))
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

    input_audio_dir: Path = Field(
        default_factory=lambda: Path(os.getenv("INPUT_AUDIO_DIR", "./data/audio_raw"))
    )
    transcripts_output_dir: Path = Field(
        default_factory=lambda: Path(os.getenv("TRANSCRIPTS_OUTPUT_DIR", "./data/transcripts"))
    )

    skip_existing: bool = Field(default_factory=lambda: env_bool("SKIP_EXISTING", "true"))
    log_level: str = Field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))


def get_settings() -> Settings:
    return Settings()
