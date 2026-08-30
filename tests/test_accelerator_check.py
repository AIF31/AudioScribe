from types import SimpleNamespace
import sys

from audio_transcriber.accelerator_check import check_accelerator
from audio_transcriber.config import Settings


def test_cuda_path_checks_nvidia_smi(monkeypatch):
    commands = []
    model_calls = []

    monkeypatch.setattr(
        "audio_transcriber.accelerator_check.shutil.which",
        lambda command: f"/usr/bin/{command}",
    )

    def fake_run(args, check, capture_output, text):
        commands.append(args)
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    class _FakeModel:
        def __init__(self, model_name, device, compute_type, use_auth_token=None):
            model_calls.append((model_name, device, compute_type, use_auth_token))

    monkeypatch.setattr("audio_transcriber.accelerator_check.subprocess.run", fake_run)
    monkeypatch.setitem(sys.modules, "faster_whisper", SimpleNamespace(WhisperModel=_FakeModel))
    monkeypatch.setitem(
        sys.modules,
        "ctranslate2",
        SimpleNamespace(__version__="4.7.1", get_cuda_device_count=lambda: 1),
    )

    settings = Settings(whisper_accelerator="cuda", whisper_device="cuda")
    result = check_accelerator(settings)

    assert commands == [["nvidia-smi"]]
    assert model_calls[0][1] == "cuda"
    assert result.accelerator == "cuda"
    assert result.ctranslate2_version == "4.7.1"
    assert result.ctranslate2_gpu_count == 1
    assert result.faster_whisper_gpu_ok is True


def test_rocm_path_checks_rocm_tools_and_uses_cuda_device(monkeypatch):
    commands = []
    model_calls = []

    monkeypatch.setattr(
        "audio_transcriber.accelerator_check.shutil.which",
        lambda command: f"/usr/bin/{command}",
    )

    def fake_run(args, check, capture_output, text):
        commands.append(args)
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    class _FakeModel:
        def __init__(self, model_name, device, compute_type, use_auth_token=None):
            model_calls.append((model_name, device, compute_type, use_auth_token))

    monkeypatch.setattr("audio_transcriber.accelerator_check.subprocess.run", fake_run)
    monkeypatch.setitem(sys.modules, "faster_whisper", SimpleNamespace(WhisperModel=_FakeModel))

    settings = Settings(whisper_accelerator="rocm", whisper_device="cuda")
    result = check_accelerator(settings)

    assert commands == [["rocminfo"], ["rocm-smi"], ["amd-smi"], ["hipcc", "--version"]]
    assert model_calls[0][0] == "tiny"
    assert model_calls[0][1] == "cuda"
    assert result.accelerator == "rocm"
    assert result.faster_whisper_gpu_ok is True


def test_cpu_path_skips_gpu_model_load(monkeypatch):
    def fail_gpu_check(*_args, **_kwargs):
        raise AssertionError("CPU accelerator check should not load faster-whisper")

    monkeypatch.setattr(
        "audio_transcriber.accelerator_check.check_faster_whisper_gpu",
        fail_gpu_check,
    )

    settings = Settings(whisper_accelerator="cpu", whisper_device="cpu")
    result = check_accelerator(settings)

    assert result.accelerator == "cpu"
    assert result.system_checks == []
    assert result.faster_whisper_gpu_ok is False
    assert result.error is None
