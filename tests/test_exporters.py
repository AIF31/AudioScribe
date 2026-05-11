import json

from audio_transcriber.config import Settings
from audio_transcriber.exporters import (
    export_all,
    export_md,
    export_metadata,
    format_timestamp,
    metadata_filename,
    transcript_filename,
)
from audio_transcriber.transcriber import TranscriptResult, TranscriptSegment


def sample_result() -> TranscriptResult:
    return TranscriptResult(
        source_file="sample_001.mp3",
        source_sha256="abc123",
        language="en",
        language_probability=0.99,
        duration=65.4,
        model_name="large-v3",
        device="cuda",
        compute_type="float16",
        batch_size=8,
        segments=[
            TranscriptSegment(id=1, start=0.0, end=4.2, text="Hola."),
            TranscriptSegment(id=2, start=61.0, end=65.4, text="Gracias."),
        ],
        full_text="Hola.\nGracias.",
    )


def test_format_timestamp():
    assert format_timestamp(0) == "00:00:00"
    assert format_timestamp(65.9) == "00:01:05"
    assert format_timestamp(3661) == "01:01:01"


def test_export_md(tmp_path):
    output = export_md(sample_result(), tmp_path)
    text = output.read_text(encoding="utf-8")

    assert output.name == "sample_001_transcript.md"
    assert "# Transcript: sample_001" in text
    assert "- Backend: faster-whisper" in text
    assert "[00:00:00 - 00:00:04]" in text
    assert "Hola." in text
    assert "- Device: cuda" in text


def test_export_metadata(tmp_path):
    output = export_metadata(sample_result(), tmp_path, Settings())
    metadata = json.loads(output.read_text(encoding="utf-8"))

    assert output.name == "sample_001_metadata.json"
    assert metadata["source_sha256"] == "abc123"
    assert metadata["transcription_backend"] == "faster-whisper"
    assert metadata["model_name"] == "large-v3"
    assert metadata["device"] == "cuda"
    assert metadata["compute_type"] == "float16"
    assert metadata["batch_size"] == 8
    assert metadata["requested_device"] == "cuda"
    assert metadata["requested_compute_type"] == "float16"
    assert metadata["requested_batch_size"] == 8
    assert metadata["openai_whisper_model"] == "whisper-1"
    assert metadata["openai_realtime_model"] == "gpt-realtime-whisper"
    assert metadata["segment_count"] == 2


def test_export_all_only_writes_markdown_and_metadata(tmp_path):
    outputs = export_all(sample_result(), tmp_path, Settings())

    expected = ["sample_001_metadata.json", "sample_001_transcript.md"]
    assert sorted(path.name for path in outputs) == expected
    assert sorted(path.name for path in tmp_path.iterdir()) == expected


def test_output_filenames_use_original_stem():
    result = sample_result()

    assert transcript_filename(result) == "sample_001_transcript.md"
    assert metadata_filename(result) == "sample_001_metadata.json"
