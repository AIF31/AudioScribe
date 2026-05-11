from pathlib import Path
import json

from audio_transcriber.config import Settings


CONFIG_FIELDS = {
    "transcription_backend": "transcription_backend",
    "model_name": "whisper_model_name",
    "device": "whisper_device",
    "compute_type": "whisper_compute_type",
    "batch_size": "whisper_batch_size",
    "language": "whisper_language",
    "task": "whisper_task",
    "beam_size": "whisper_beam_size",
    "vad_filter": "whisper_vad_filter",
    "min_silence_duration_ms": "whisper_min_silence_duration_ms",
    "condition_on_previous_text": "whisper_condition_on_previous_text",
}

def should_skip_existing(output_dir: Path, source_sha256: str, settings: Settings) -> bool:
    if not settings.skip_existing:
        return False

    metadata_path = _find_metadata_path(output_dir, source_sha256)
    transcript_path = _find_transcript_path(output_dir, metadata_path)
    if not metadata_path.exists() or not transcript_path.exists():
        return False

    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False

    if metadata.get("source_sha256") != source_sha256:
        return False

    if not _metadata_matches_settings(metadata, settings):
        return False

    if metadata.get("initial_prompt_used") != bool(settings.whisper_initial_prompt):
        return False

    return True


def _metadata_matches_settings(metadata: dict, settings: Settings) -> bool:
    if settings.transcription_backend == "openai-whisper":
        return _openai_whisper_metadata_matches_settings(metadata, settings)
    if settings.transcription_backend == "openai-realtime-whisper":
        return _openai_realtime_metadata_matches_settings(metadata, settings)

    expected_variants = [settings]
    fallback_settings = _cpu_fallback_variant(settings)
    if fallback_settings is not None:
        expected_variants.append(fallback_settings)

    for candidate in expected_variants:
        if all(
            metadata.get(metadata_key) == getattr(candidate, settings_attr)
            for metadata_key, settings_attr in CONFIG_FIELDS.items()
        ):
            return True
    return False


def _openai_whisper_metadata_matches_settings(metadata: dict, settings: Settings) -> bool:
    expected = {
        "transcription_backend": settings.transcription_backend,
        "model_name": settings.openai_whisper_model,
        "language": settings.whisper_language,
        "initial_prompt_used": bool(settings.whisper_initial_prompt),
        "openai_whisper_model": settings.openai_whisper_model,
    }
    return all(metadata.get(key) == value for key, value in expected.items())


def _openai_realtime_metadata_matches_settings(metadata: dict, settings: Settings) -> bool:
    expected = {
        "transcription_backend": settings.transcription_backend,
        "model_name": settings.openai_realtime_model,
        "language": settings.whisper_language,
        "initial_prompt_used": bool(settings.whisper_initial_prompt),
        "openai_realtime_model": settings.openai_realtime_model,
        "openai_realtime_audio_rate": settings.openai_realtime_audio_rate,
        "openai_realtime_turn_detection": settings.openai_realtime_turn_detection,
        "openai_realtime_noise_reduction": settings.openai_realtime_noise_reduction,
        "openai_realtime_timeout_seconds": settings.openai_realtime_timeout_seconds,
    }
    return all(metadata.get(key) == value for key, value in expected.items())


def _cpu_fallback_variant(settings: Settings) -> Settings | None:
    if settings.whisper_device != "cuda" or not settings.whisper_allow_cpu_fallback:
        return None
    return settings.model_copy(
        update={
            "whisper_device": "cpu",
            "whisper_compute_type": "int8",
            "whisper_batch_size": 1,
        }
    )


def _find_metadata_path(output_dir: Path, source_sha256: str) -> Path:
    candidates = sorted(output_dir.glob("*_metadata.json"))
    legacy_path = output_dir / "metadata.json"
    if legacy_path.exists():
        candidates.append(legacy_path)

    for candidate in candidates:
        try:
            metadata = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if metadata.get("source_sha256") == source_sha256:
            return candidate

    return candidates[0] if candidates else output_dir / "metadata.json"


def _find_transcript_path(output_dir: Path, metadata_path: Path) -> Path:
    if metadata_path.name.endswith("_metadata.json"):
        stem = metadata_path.name.removesuffix("_metadata.json")
        return output_dir / f"{stem}_transcript.md"
    return output_dir / "transcript.md"
