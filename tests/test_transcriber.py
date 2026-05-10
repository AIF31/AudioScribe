from types import SimpleNamespace
import sys

from kado_transcriber.config import Settings
from kado_transcriber.transcriber import FasterWhisperTranscriber


class _FakeSegment:
    def __init__(self, start: float, end: float, text: str):
        self.start = start
        self.end = end
        self.text = text


class _FakeModel:
    def __init__(self, _model_name: str, device: str, compute_type: str, use_auth_token=None):
        if device == "cuda":
            raise RuntimeError("CUDA failed with error CUDA driver version is insufficient for CUDA runtime version")
        self.device = device
        self.compute_type = compute_type
        self.use_auth_token = use_auth_token

    def transcribe(self, _audio_path: str, **_kwargs):
        return [
            _FakeSegment(0.0, 1.0, "Hola"),
            _FakeSegment(1.0, 2.0, "mundo"),
        ], SimpleNamespace(language="es", language_probability=0.99, duration=2.0)


class _FakePipeline:
    def __init__(self, model):
        self.model = model

    def transcribe(self, audio_path: str, **kwargs):
        return self.model.transcribe(audio_path, **kwargs)


def test_transcriber_falls_back_to_cpu_for_cuda_init_errors(monkeypatch, tmp_path):
    fake_module = SimpleNamespace(
        WhisperModel=_FakeModel,
        BatchedInferencePipeline=_FakePipeline,
    )
    monkeypatch.setitem(sys.modules, "faster_whisper", fake_module)

    settings = Settings()
    transcriber = FasterWhisperTranscriber(settings)
    result = transcriber.transcribe_file(tmp_path / "sample.mp3")

    assert transcriber.runtime_settings.whisper_device == "cpu"
    assert transcriber.runtime_settings.whisper_compute_type == "int8"
    assert transcriber.runtime_settings.whisper_batch_size == 1
    assert transcriber.fallback_reason is not None
    assert result.device == "cpu"
    assert result.compute_type == "int8"
    assert result.batch_size == 1


def test_transcriber_keeps_cuda_error_when_fallback_disabled(monkeypatch):
    fake_module = SimpleNamespace(
        WhisperModel=_FakeModel,
        BatchedInferencePipeline=_FakePipeline,
    )
    monkeypatch.setitem(sys.modules, "faster_whisper", fake_module)

    settings = Settings(
        whisper_allow_cpu_fallback=False,
    )

    try:
        FasterWhisperTranscriber(settings)
    except RuntimeError as exc:
        assert "Failed to load faster-whisper CUDA model" in str(exc)
    else:
        raise AssertionError("Expected CUDA initialization failure")
