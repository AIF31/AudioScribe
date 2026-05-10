from kado_transcriber.config import Settings
from kado_transcriber.exporters import export_all
from kado_transcriber.skip import should_skip_existing
from kado_transcriber.transcriber import TranscriptResult


def test_should_skip_when_hash_and_config_match(tmp_path):
    settings = Settings()
    result = TranscriptResult(
        source_file="interview.mp3",
        source_sha256="hash",
        model_name=settings.whisper_model_name,
        device=settings.whisper_device,
        compute_type=settings.whisper_compute_type,
        batch_size=settings.whisper_batch_size,
        segments=[],
        full_text="",
    )
    export_all(result, tmp_path, settings)

    assert should_skip_existing(tmp_path, "hash", settings) is True


def test_should_not_skip_when_hash_changes(tmp_path):
    settings = Settings()
    result = TranscriptResult(
        source_file="interview.mp3",
        source_sha256="hash",
        model_name=settings.whisper_model_name,
        device=settings.whisper_device,
        compute_type=settings.whisper_compute_type,
        batch_size=settings.whisper_batch_size,
        segments=[],
        full_text="",
    )
    export_all(result, tmp_path, settings)

    assert should_skip_existing(tmp_path, "different", settings) is False


def test_should_not_skip_when_transcript_missing(tmp_path):
    settings = Settings()
    result = TranscriptResult(
        source_file="interview.mp3",
        source_sha256="hash",
        model_name=settings.whisper_model_name,
        device=settings.whisper_device,
        compute_type=settings.whisper_compute_type,
        batch_size=settings.whisper_batch_size,
        segments=[],
        full_text="",
    )
    export_all(result, tmp_path, settings)
    (tmp_path / "interview_transcript.md").unlink()

    assert should_skip_existing(tmp_path, "hash", settings) is False


def test_should_skip_legacy_transcript_and_metadata_names(tmp_path):
    settings = Settings()
    result = TranscriptResult(
        source_file="interview.mp3",
        source_sha256="hash",
        model_name=settings.whisper_model_name,
        device=settings.whisper_device,
        compute_type=settings.whisper_compute_type,
        batch_size=settings.whisper_batch_size,
        segments=[],
        full_text="",
    )
    export_all(result, tmp_path, settings)
    (tmp_path / "interview_transcript.md").rename(tmp_path / "transcript.md")
    (tmp_path / "interview_metadata.json").rename(tmp_path / "metadata.json")

    assert should_skip_existing(tmp_path, "hash", settings) is True


def test_should_skip_when_cuda_config_previously_fell_back_to_cpu(tmp_path):
    settings = Settings()
    result = TranscriptResult(
        source_file="interview.mp3",
        source_sha256="hash",
        model_name=settings.whisper_model_name,
        device="cpu",
        compute_type="int8",
        batch_size=1,
        segments=[],
        full_text="",
    )
    export_all(result, tmp_path, settings)

    assert should_skip_existing(tmp_path, "hash", settings) is True
