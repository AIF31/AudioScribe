from audio_transcriber.config import Settings
from audio_transcriber.exporters import export_all
from audio_transcriber.skip import should_skip_existing
from audio_transcriber.transcriber import TranscriptResult
import pytest


@pytest.fixture(autouse=True)
def _clear_backend_env(monkeypatch):
    monkeypatch.delenv("TRANSCRIPTION_BACKEND", raising=False)


def test_should_skip_when_hash_and_config_match(tmp_path):
    settings = Settings()
    result = TranscriptResult(
        source_file="sample.mp3",
        source_sha256="hash",
        model_name=settings.whisper_model_name,
        accelerator=settings.runtime_accelerator_label,
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
        source_file="sample.mp3",
        source_sha256="hash",
        model_name=settings.whisper_model_name,
        accelerator=settings.runtime_accelerator_label,
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
        source_file="sample.mp3",
        source_sha256="hash",
        model_name=settings.whisper_model_name,
        accelerator=settings.runtime_accelerator_label,
        device=settings.whisper_device,
        compute_type=settings.whisper_compute_type,
        batch_size=settings.whisper_batch_size,
        segments=[],
        full_text="",
    )
    export_all(result, tmp_path, settings)
    (tmp_path / "sample_transcript.md").unlink()

    assert should_skip_existing(tmp_path, "hash", settings) is False


def test_should_skip_legacy_transcript_and_metadata_names(tmp_path):
    settings = Settings()
    result = TranscriptResult(
        source_file="sample.mp3",
        source_sha256="hash",
        model_name=settings.whisper_model_name,
        accelerator=settings.runtime_accelerator_label,
        device=settings.whisper_device,
        compute_type=settings.whisper_compute_type,
        batch_size=settings.whisper_batch_size,
        segments=[],
        full_text="",
    )
    export_all(result, tmp_path, settings)
    (tmp_path / "sample_transcript.md").rename(tmp_path / "transcript.md")
    (tmp_path / "sample_metadata.json").rename(tmp_path / "metadata.json")

    assert should_skip_existing(tmp_path, "hash", settings) is True


def test_should_skip_when_cuda_config_previously_fell_back_to_cpu(tmp_path):
    settings = Settings()
    result = TranscriptResult(
        source_file="sample.mp3",
        source_sha256="hash",
        model_name=settings.whisper_model_name,
        accelerator="cpu",
        device="cpu",
        compute_type="int8",
        batch_size=1,
        segments=[],
        full_text="",
    )
    export_all(result, tmp_path, settings)

    assert should_skip_existing(tmp_path, "hash", settings) is True


def test_should_not_skip_between_faster_whisper_and_openai_realtime(tmp_path):
    faster_settings = Settings()
    result = TranscriptResult(
        source_file="sample.mp3",
        source_sha256="hash",
        model_name=faster_settings.whisper_model_name,
        accelerator=faster_settings.runtime_accelerator_label,
        device=faster_settings.whisper_device,
        compute_type=faster_settings.whisper_compute_type,
        batch_size=faster_settings.whisper_batch_size,
        segments=[],
        full_text="",
    )
    export_all(result, tmp_path, faster_settings)

    openai_settings = Settings(
        transcription_backend="openai-realtime-whisper",
        openai_api_key="sk_test_key",
    )

    assert should_skip_existing(tmp_path, "hash", openai_settings) is False


def test_should_not_skip_between_faster_whisper_and_openai_whisper(tmp_path):
    faster_settings = Settings()
    result = TranscriptResult(
        source_file="sample.mp3",
        source_sha256="hash",
        model_name=faster_settings.whisper_model_name,
        accelerator=faster_settings.runtime_accelerator_label,
        device=faster_settings.whisper_device,
        compute_type=faster_settings.whisper_compute_type,
        batch_size=faster_settings.whisper_batch_size,
        segments=[],
        full_text="",
    )
    export_all(result, tmp_path, faster_settings)

    openai_settings = Settings(
        transcription_backend="openai-whisper",
        openai_api_key="sk_test_key",
    )

    assert should_skip_existing(tmp_path, "hash", openai_settings) is False


def test_should_not_skip_openai_whisper_when_task_changes(tmp_path):
    transcribe_settings = Settings(
        transcription_backend="openai-whisper",
        openai_api_key="sk_test_key",
        whisper_task="transcribe",
    )
    result = TranscriptResult(
        source_file="sample.mp3",
        source_sha256="hash",
        transcription_backend="openai-whisper",
        language=transcribe_settings.whisper_language,
        model_name=transcribe_settings.openai_whisper_model,
        accelerator="openai",
        device="openai",
        compute_type="api",
        batch_size=1,
        segments=[],
        full_text="",
    )
    export_all(result, tmp_path, transcribe_settings)

    translate_settings = transcribe_settings.model_copy(
        update={"whisper_task": "translate"}
    )

    assert should_skip_existing(tmp_path, "hash", translate_settings) is False


def test_should_not_skip_legacy_cuda_metadata_for_rocm_settings(tmp_path):
    cuda_settings = Settings(whisper_accelerator="cuda", whisper_device="cuda")
    result = TranscriptResult(
        source_file="sample.mp3",
        source_sha256="hash",
        model_name=cuda_settings.whisper_model_name,
        device=cuda_settings.whisper_device,
        compute_type=cuda_settings.whisper_compute_type,
        batch_size=cuda_settings.whisper_batch_size,
        segments=[],
        full_text="",
    )
    export_all(result, tmp_path, cuda_settings)

    metadata_path = tmp_path / "sample_metadata.json"
    metadata_text = metadata_path.read_text(encoding="utf-8")
    metadata_text = metadata_text.replace('  "accelerator": null,\n', "")
    metadata_text = metadata_text.replace('  "requested_accelerator": "cuda",\n', "")
    metadata_text = metadata_text.replace('  "effective_device": "cuda",\n', "")
    metadata_path.write_text(metadata_text, encoding="utf-8")

    rocm_settings = Settings(whisper_accelerator="rocm", whisper_device="cuda")

    assert should_skip_existing(tmp_path, "hash", rocm_settings) is False


def test_should_skip_rocm_metadata_for_rocm_settings(tmp_path):
    settings = Settings(whisper_accelerator="rocm", whisper_device="cuda")
    result = TranscriptResult(
        source_file="sample.mp3",
        source_sha256="hash",
        model_name=settings.whisper_model_name,
        accelerator="rocm",
        device=settings.whisper_device,
        compute_type=settings.whisper_compute_type,
        batch_size=settings.whisper_batch_size,
        segments=[],
        full_text="",
    )
    export_all(result, tmp_path, settings)

    assert should_skip_existing(tmp_path, "hash", settings) is True
