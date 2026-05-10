from dataclasses import dataclass
import os
import shutil
import subprocess


@dataclass
class CudaCheckResult:
    nvidia_smi_found: bool
    nvidia_smi_output: str | None
    faster_whisper_cuda_ok: bool
    error: str | None = None


def run_nvidia_smi() -> tuple[bool, str | None]:
    if shutil.which("nvidia-smi") is None:
        return False, None
    completed = subprocess.run(
        ["nvidia-smi"],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.returncode == 0, completed.stdout or completed.stderr


def check_faster_whisper_cuda() -> tuple[bool, str | None]:
    try:
        from faster_whisper import WhisperModel

        WhisperModel(
            "tiny",
            device="cuda",
            compute_type="float16",
            use_auth_token=os.getenv("HF_TOKEN") or None,
        )
        return True, None
    except Exception as exc:
        return False, str(exc)


def check_cuda() -> CudaCheckResult:
    nvidia_ok, nvidia_output = run_nvidia_smi()
    fw_ok, error = check_faster_whisper_cuda()
    return CudaCheckResult(
        nvidia_smi_found=nvidia_ok,
        nvidia_smi_output=nvidia_output,
        faster_whisper_cuda_ok=fw_ok,
        error=error,
    )
