from pathlib import Path

from audio_transcriber.config import Settings, env_bool


def test_default_settings_are_cuda_first(monkeypatch):
    for key in [
        "TRANSCRIPTION_BACKEND",
        "WHISPER_MODEL_NAME",
        "WHISPER_DEVICE",
        "WHISPER_COMPUTE_TYPE",
        "WHISPER_LANGUAGE",
        "WHISPER_BATCH_SIZE",
        "WHISPER_INITIAL_PROMPT",
        "HF_TOKEN",
        "DIARIZATION_MODEL",
        "DIARIZATION_DEVICE",
        "OPENAI_API_KEY",
        "OPENAI_WHISPER_MODEL",
        "OPENAI_REALTIME_MODEL",
    ]:
        monkeypatch.delenv(key, raising=False)

    settings = Settings()

    assert settings.transcription_backend == "faster-whisper"
    assert settings.whisper_model_name == "large-v3"
    assert settings.whisper_device == "cuda"
    assert settings.whisper_compute_type == "float16"
    assert settings.whisper_language is None
    assert settings.whisper_batch_size == 8
    assert settings.whisper_allow_cpu_fallback is True
    assert settings.whisper_initial_prompt is None
    assert settings.hf_token is None
    assert settings.diarization_model == "pyannote/speaker-diarization-3.1"
    assert settings.diarization_device == "cuda"
    assert settings.openai_api_key is None
    assert settings.openai_whisper_model == "whisper-1"
    assert settings.openai_realtime_model == "gpt-realtime-whisper"


def test_bool_parsing(monkeypatch):
    monkeypatch.setenv("BOOL_VALUE", "yes")
    assert env_bool("BOOL_VALUE") is True
    monkeypatch.setenv("BOOL_VALUE", "off")
    assert env_bool("BOOL_VALUE") is False


def test_whisper_language_auto_and_explicit(monkeypatch):
    monkeypatch.delenv("WHISPER_LANGUAGE", raising=False)
    assert Settings().whisper_language is None

    monkeypatch.setenv("WHISPER_LANGUAGE", "auto")
    assert Settings().whisper_language is None

    monkeypatch.setenv("WHISPER_LANGUAGE", "")
    assert Settings().whisper_language is None

    monkeypatch.setenv("WHISPER_LANGUAGE", "  ES ")
    assert Settings().whisper_language == "es"


def test_path_env_vars(monkeypatch):
    monkeypatch.setenv("INPUT_AUDIO_DIR", "./custom_audio")
    monkeypatch.setenv("TRANSCRIPTS_OUTPUT_DIR", "./custom_transcripts")

    settings = Settings()

    assert settings.input_audio_dir == Path("./custom_audio")
    assert settings.transcripts_output_dir == Path("./custom_transcripts")


def test_cpu_fallback_settings(monkeypatch):
    monkeypatch.setenv("WHISPER_MODEL_NAME", "medium")
    monkeypatch.setenv("WHISPER_DEVICE", "cpu")
    monkeypatch.setenv("WHISPER_COMPUTE_TYPE", "int8")
    monkeypatch.setenv("WHISPER_BATCH_SIZE", "1")

    settings = Settings()

    assert settings.whisper_model_name == "medium"
    assert settings.whisper_device == "cpu"
    assert settings.whisper_compute_type == "int8"
    assert settings.whisper_batch_size == 1


def test_hf_token_loads_from_env(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_test_token")

    settings = Settings()

    assert settings.hf_token == "hf_test_token"


def test_openai_realtime_backend_requires_api_key(monkeypatch):
    monkeypatch.setenv("TRANSCRIPTION_BACKEND", "openai-realtime-whisper")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    try:
        Settings()
    except ValueError as exc:
        assert "OPENAI_API_KEY is required" in str(exc)
    else:
        raise AssertionError("Expected missing OpenAI API key to fail")


def test_openai_whisper_backend_loads_from_env(monkeypatch):
    monkeypatch.setenv("TRANSCRIPTION_BACKEND", "openai-whisper")
    monkeypatch.setenv("OPENAI_API_KEY", "sk_test_key")
    monkeypatch.setenv("OPENAI_WHISPER_MODEL", "whisper-1")

    settings = Settings()

    assert settings.transcription_backend == "openai-whisper"
    assert settings.openai_api_key == "sk_test_key"
    assert settings.openai_whisper_model == "whisper-1"


def test_openai_realtime_backend_loads_from_env(monkeypatch):
    monkeypatch.setenv("TRANSCRIPTION_BACKEND", "openai-realtime-whisper")
    monkeypatch.setenv("OPENAI_API_KEY", "sk_test_key")
    monkeypatch.setenv("OPENAI_REALTIME_MODEL", "gpt-realtime-whisper")

    settings = Settings()

    assert settings.transcription_backend == "openai-realtime-whisper"
    assert settings.openai_api_key == "sk_test_key"
    assert settings.openai_realtime_model == "gpt-realtime-whisper"
