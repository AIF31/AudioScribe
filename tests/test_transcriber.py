from types import SimpleNamespace
import sys
import json

from audio_transcriber.config import Settings
from audio_transcriber.transcriber import (
    FasterWhisperTranscriber,
    OpenAIRealtimeWhisperTranscriber,
    OpenAIWhisperTranscriber,
    create_transcriber,
)


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
        ], SimpleNamespace(language="en", language_probability=0.99, duration=2.0)


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

    assert transcriber.runtime_settings.whisper_accelerator == "cpu"
    assert transcriber.runtime_settings.whisper_device == "cpu"
    assert transcriber.runtime_settings.whisper_compute_type == "int8"
    assert transcriber.runtime_settings.whisper_batch_size == 1
    assert transcriber.fallback_reason is not None
    assert result.device == "cpu"
    assert result.accelerator == "cpu"
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


def test_transcriber_passes_cuda_device_for_rocm(monkeypatch, tmp_path):
    calls = []

    class _RocmFakeModel:
        def __init__(self, model_name: str, device: str, compute_type: str, use_auth_token=None):
            calls.append(
                {
                    "model_name": model_name,
                    "device": device,
                    "compute_type": compute_type,
                    "use_auth_token": use_auth_token,
                }
            )

        def transcribe(self, _audio_path: str, **_kwargs):
            return [_FakeSegment(0.0, 1.0, "Hola")], SimpleNamespace(
                language="es",
                language_probability=0.99,
                duration=1.0,
            )

    fake_module = SimpleNamespace(
        WhisperModel=_RocmFakeModel,
        BatchedInferencePipeline=_FakePipeline,
    )
    monkeypatch.setitem(sys.modules, "faster_whisper", fake_module)

    settings = Settings(
        whisper_accelerator="rocm",
        whisper_device="cuda",
        whisper_batch_size=1,
    )
    result = FasterWhisperTranscriber(settings).transcribe_file(tmp_path / "sample.mp3")

    assert calls[0]["device"] == "cuda"
    assert result.accelerator == "rocm"
    assert result.device == "cuda"


def test_transcriber_falls_back_to_cpu_for_rocm_init_errors(monkeypatch, tmp_path):
    class _RocmFailingModel:
        def __init__(self, _model_name: str, device: str, compute_type: str, use_auth_token=None):
            if device == "cuda":
                raise RuntimeError("hipblas failed to initialize")

        def transcribe(self, _audio_path: str, **_kwargs):
            return [_FakeSegment(0.0, 1.0, "Hola")], SimpleNamespace(
                language="es",
                language_probability=0.99,
                duration=1.0,
            )

    fake_module = SimpleNamespace(
        WhisperModel=_RocmFailingModel,
        BatchedInferencePipeline=_FakePipeline,
    )
    monkeypatch.setitem(sys.modules, "faster_whisper", fake_module)

    settings = Settings(whisper_accelerator="rocm", whisper_device="cuda")
    transcriber = FasterWhisperTranscriber(settings)
    result = transcriber.transcribe_file(tmp_path / "sample.mp3")

    assert transcriber.runtime_settings.whisper_accelerator == "cpu"
    assert transcriber.runtime_settings.whisper_device == "cpu"
    assert result.accelerator == "cpu"


def test_transcriber_keeps_rocm_error_when_fallback_disabled(monkeypatch):
    class _RocmFailingModel:
        def __init__(self, _model_name: str, device: str, compute_type: str, use_auth_token=None):
            if device == "cuda":
                raise RuntimeError("rocblas failed to initialize")

    fake_module = SimpleNamespace(
        WhisperModel=_RocmFailingModel,
        BatchedInferencePipeline=_FakePipeline,
    )
    monkeypatch.setitem(sys.modules, "faster_whisper", fake_module)

    settings = Settings(
        whisper_accelerator="rocm",
        whisper_device="cuda",
        whisper_allow_cpu_fallback=False,
    )

    try:
        FasterWhisperTranscriber(settings)
    except RuntimeError as exc:
        assert "Failed to load faster-whisper ROCm/HIP model" in str(exc)
    else:
        raise AssertionError("Expected ROCm initialization failure")


def test_create_transcriber_selects_openai_realtime_backend(monkeypatch):
    settings = Settings(
        transcription_backend="openai-realtime-whisper",
        openai_api_key="sk_test_key",
    )

    transcriber = create_transcriber(settings)

    assert isinstance(transcriber, OpenAIRealtimeWhisperTranscriber)


def test_create_transcriber_selects_openai_whisper_backend(monkeypatch):
    settings = Settings(
        transcription_backend="openai-whisper",
        openai_api_key="sk_test_key",
        whisper_language="en",
    )

    transcriber = create_transcriber(settings)

    assert isinstance(transcriber, OpenAIWhisperTranscriber)


def test_openai_whisper_transcriber_uses_transcriptions_api(monkeypatch, tmp_path):
    transcription_calls = []
    translation_calls = []

    class _FakeTranscriptions:
        def create(self, **kwargs):
            transcription_calls.append(kwargs)
            return SimpleNamespace(
                model_dump=lambda: {
                    "text": "Hola mundo",
                    "language": "spanish",
                    "duration": 2.0,
                    "segments": [
                        {"start": 0.0, "end": 1.0, "text": "Hola"},
                        {"start": 1.0, "end": 2.0, "text": "mundo"},
                    ],
                }
            )

    class _FakeTranslations:
        def create(self, **kwargs):
            translation_calls.append(kwargs)
            raise AssertionError("translations endpoint should not be called")

    class _FakeOpenAI:
        def __init__(self, api_key):
            self.api_key = api_key
            self.audio = SimpleNamespace(
                transcriptions=_FakeTranscriptions(),
                translations=_FakeTranslations(),
            )

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=_FakeOpenAI))

    audio_path = tmp_path / "sample.mp3"
    audio_path.write_bytes(b"audio")
    settings = Settings(
        transcription_backend="openai-whisper",
        openai_api_key="sk_test_key",
        whisper_language="en",
    )
    result = OpenAIWhisperTranscriber(settings).transcribe_file(audio_path)

    assert transcription_calls[0]["model"] == "whisper-1"
    assert transcription_calls[0]["response_format"] == "verbose_json"
    assert transcription_calls[0]["language"] == "en"
    assert translation_calls == []
    assert result.transcription_backend == "openai-whisper"
    assert result.model_name == "whisper-1"
    assert result.accelerator == "openai"
    assert result.full_text == "Hola mundo"
    assert len(result.segments) == 2


def test_openai_whisper_transcriber_uses_translations_api_for_translate(monkeypatch, tmp_path):
    transcription_calls = []
    translation_calls = []

    class _FakeTranscriptions:
        def create(self, **kwargs):
            transcription_calls.append(kwargs)
            raise AssertionError("transcriptions endpoint should not be called")

    class _FakeTranslations:
        def create(self, **kwargs):
            translation_calls.append(kwargs)
            return SimpleNamespace(
                model_dump=lambda: {
                    "text": "Hello world",
                    "duration": 2.0,
                }
            )

    class _FakeOpenAI:
        def __init__(self, api_key):
            self.api_key = api_key
            self.audio = SimpleNamespace(
                transcriptions=_FakeTranscriptions(),
                translations=_FakeTranslations(),
            )

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=_FakeOpenAI))

    audio_path = tmp_path / "sample.mp3"
    audio_path.write_bytes(b"audio")
    settings = Settings(
        transcription_backend="openai-whisper",
        openai_api_key="sk_test_key",
        whisper_language="es",
        whisper_task="translate",
    )
    result = OpenAIWhisperTranscriber(settings).transcribe_file(audio_path)

    assert transcription_calls == []
    assert translation_calls[0]["model"] == "whisper-1"
    assert translation_calls[0]["response_format"] == "verbose_json"
    assert "language" not in translation_calls[0]
    assert result.full_text == "Hello world"
    assert result.segments[0].text == "Hello world"


def test_openai_realtime_transcriber_sends_session_and_audio(monkeypatch, tmp_path):
    sent_events = []

    class WebSocketTimeoutException(Exception):
        pass

    class _FakeWebSocket:
        def __init__(self):
            self.recv_count = 0

        def send(self, payload):
            sent_events.append(json.loads(payload))

        def settimeout(self, _timeout):
            pass

        def recv(self):
            self.recv_count += 1
            if self.recv_count == 1:
                return json.dumps(
                    {
                        "type": "conversation.item.input_audio_transcription.completed",
                        "transcript": "Hola mundo",
                    }
                )
            raise WebSocketTimeoutException()

        def close(self):
            pass

    fake_websocket_module = SimpleNamespace(
        WebSocketTimeoutException=WebSocketTimeoutException,
        create_connection=lambda *_args, **_kwargs: _FakeWebSocket()
    )
    monkeypatch.setitem(sys.modules, "websocket", fake_websocket_module)
    monkeypatch.setattr(
        "audio_transcriber.transcriber._decode_audio_pcm_chunks",
        lambda *_args, **_kwargs: [b"pcm-bytes"],
    )

    settings = Settings(
        transcription_backend="openai-realtime-whisper",
        openai_api_key="sk_test_key",
        whisper_language="en",
    )
    result = OpenAIRealtimeWhisperTranscriber(settings).transcribe_file(
        tmp_path / "sample.mp3"
    )

    assert sent_events[0]["type"] == "session.update"
    assert sent_events[0]["session"]["type"] == "transcription"
    audio_input = sent_events[0]["session"]["audio"]["input"]
    assert audio_input["format"] == {"type": "audio/pcm", "rate": 24000}
    assert audio_input["transcription"] == {
        "model": "gpt-realtime-whisper",
        "language": "en",
    }
    assert audio_input["turn_detection"] == {
        "type": "server_vad",
        "threshold": 0.5,
        "prefix_padding_ms": 300,
        "silence_duration_ms": 500,
    }
    assert audio_input["noise_reduction"] == {"type": "near_field"}
    assert "input_audio_format" not in sent_events[0]
    assert "input_audio_transcription" not in sent_events[0]
    assert "turn_detection" not in sent_events[0]
    assert "input_audio_noise_reduction" not in sent_events[0]
    assert sent_events[1]["type"] == "input_audio_buffer.append"
    assert sent_events[2]["type"] == "input_audio_buffer.commit"
    assert result.transcription_backend == "openai-realtime-whisper"
    assert result.model_name == "gpt-realtime-whisper"
    assert result.accelerator == "openai"
    assert result.full_text == "Hola mundo"
