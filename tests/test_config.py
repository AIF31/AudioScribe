from pathlib import Path

from kado_transcriber.config import Settings, env_bool


def test_default_settings_are_cuda_first(monkeypatch):
    for key in [
        "WHISPER_MODEL_NAME",
        "WHISPER_DEVICE",
        "WHISPER_COMPUTE_TYPE",
        "WHISPER_LANGUAGE",
        "WHISPER_BATCH_SIZE",
        "WHISPER_INITIAL_PROMPT",
        "HF_TOKEN",
    ]:
        monkeypatch.delenv(key, raising=False)

    settings = Settings()

    assert settings.whisper_model_name == "large-v3"
    assert settings.whisper_device == "cuda"
    assert settings.whisper_compute_type == "float16"
    assert settings.whisper_language == "es"
    assert settings.whisper_batch_size == 8
    assert settings.whisper_allow_cpu_fallback is True
    assert settings.whisper_initial_prompt is None
    assert settings.hf_token is None


def test_bool_parsing(monkeypatch):
    monkeypatch.setenv("BOOL_VALUE", "yes")
    assert env_bool("BOOL_VALUE") is True
    monkeypatch.setenv("BOOL_VALUE", "off")
    assert env_bool("BOOL_VALUE") is False


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
