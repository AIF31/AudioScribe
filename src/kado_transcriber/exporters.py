from datetime import UTC, datetime
from pathlib import Path
import json

from kado_transcriber.config import Settings
from kado_transcriber.transcriber import TranscriptResult


def output_stem(result: TranscriptResult) -> str:
    return Path(result.source_file).stem


def transcript_filename(result: TranscriptResult) -> str:
    return f"{output_stem(result)}_transcript.md"


def metadata_filename(result: TranscriptResult) -> str:
    return f"{output_stem(result)}_metadata.json"


def format_timestamp(seconds: float | None) -> str:
    if seconds is None:
        return "00:00:00"
    total_seconds = max(0, int(seconds))
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    return f"{hours:02}:{minutes:02}:{secs:02}"


def export_md(result: TranscriptResult, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / transcript_filename(result)
    lines = [
        f"# Transcript: {Path(result.source_file).stem}",
        "",
        f"- Source file: {result.source_file}",
        f"- Language: {result.language or 'unknown'}",
        f"- Language probability: {_format_probability(result.language_probability)}",
        f"- Model: {result.model_name}",
        f"- Device: {result.device}",
        f"- Compute type: {result.compute_type}",
        f"- Batch size: {result.batch_size}",
        f"- Duration: {format_timestamp(result.duration)}",
        f"- Segments: {len(result.segments)}",
        "",
        "## Transcript",
        "",
    ]
    for segment in result.segments:
        lines.extend(
            [
                f"[{format_timestamp(segment.start)} - {format_timestamp(segment.end)}]",
                segment.text,
                "",
            ]
        )
    output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return output_path


def export_metadata(result: TranscriptResult, output_dir: Path, settings: Settings) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / metadata_filename(result)
    metadata = build_metadata(result, settings)
    output_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output_path


def export_all(result: TranscriptResult, output_dir: Path, settings: Settings) -> list[Path]:
    return [
        export_md(result, output_dir),
        export_metadata(result, output_dir, settings),
    ]


def build_metadata(result: TranscriptResult, settings: Settings) -> dict:
    return {
        "source_file": result.source_file,
        "source_sha256": result.source_sha256,
        "created_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "model_name": result.model_name,
        "device": result.device,
        "compute_type": result.compute_type,
        "batch_size": result.batch_size,
        "language": settings.whisper_language,
        "task": settings.whisper_task,
        "beam_size": settings.whisper_beam_size,
        "vad_filter": settings.whisper_vad_filter,
        "min_silence_duration_ms": settings.whisper_min_silence_duration_ms,
        "condition_on_previous_text": settings.whisper_condition_on_previous_text,
        "initial_prompt_used": bool(settings.whisper_initial_prompt),
        "requested_model_name": settings.whisper_model_name,
        "requested_device": settings.whisper_device,
        "requested_compute_type": settings.whisper_compute_type,
        "requested_batch_size": settings.whisper_batch_size,
        "duration": result.duration,
        "detected_language": result.language,
        "language_probability": result.language_probability,
        "segment_count": len(result.segments),
    }


def _format_probability(value: float | None) -> str:
    return "unknown" if value is None else f"{value:.3f}"
