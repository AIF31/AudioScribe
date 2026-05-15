from dataclasses import dataclass
import os
import shutil
import subprocess

from audio_transcriber.config import Settings


@dataclass
class CommandCheck:
    name: str
    found: bool
    ok: bool
    output: str | None = None


@dataclass
class AcceleratorCheckResult:
    accelerator: str
    ctranslate2_version: str | None
    ctranslate2_gpu_count: int | None
    system_checks: list[CommandCheck]
    faster_whisper_gpu_ok: bool
    error: str | None = None

    @property
    def nvidia_smi_found(self) -> bool:
        check = self._system_check("nvidia-smi")
        return bool(check and check.ok)

    @property
    def nvidia_smi_output(self) -> str | None:
        check = self._system_check("nvidia-smi")
        return check.output if check else None

    @property
    def faster_whisper_cuda_ok(self) -> bool:
        return self.faster_whisper_gpu_ok

    def _system_check(self, name: str) -> CommandCheck | None:
        return next((check for check in self.system_checks if check.name == name), None)


def _run_command(name: str, args: list[str]) -> CommandCheck:
    if shutil.which(args[0]) is None:
        return CommandCheck(name=name, found=False, ok=False, output=None)
    completed = subprocess.run(
        args,
        check=False,
        capture_output=True,
        text=True,
    )
    return CommandCheck(
        name=name,
        found=True,
        ok=completed.returncode == 0,
        output=completed.stdout or completed.stderr,
    )


def _ct2_version_and_gpu_count() -> tuple[str | None, int | None]:
    try:
        import ctranslate2
    except Exception:
        return None, None

    version = getattr(ctranslate2, "__version__", None)
    gpu_count = None
    if hasattr(ctranslate2, "get_cuda_device_count"):
        try:
            gpu_count = ctranslate2.get_cuda_device_count()
        except Exception:
            gpu_count = None
    return version, gpu_count


def check_faster_whisper_gpu(settings: Settings) -> tuple[bool, str | None]:
    try:
        from faster_whisper import WhisperModel

        WhisperModel(
            "tiny",
            device=settings.effective_whisper_device,
            compute_type=settings.whisper_compute_type,
            use_auth_token=os.getenv("HF_TOKEN") or None,
        )
        return True, None
    except Exception as exc:
        return False, str(exc)


def check_accelerator(settings: Settings) -> AcceleratorCheckResult:
    accelerator = settings.runtime_accelerator_label
    checks: list[CommandCheck] = []

    if accelerator == "cuda":
        checks.append(_run_command("nvidia-smi", ["nvidia-smi"]))
    elif accelerator == "rocm":
        checks.append(_run_command("rocminfo", ["rocminfo"]))
        checks.append(_run_command("rocm-smi", ["rocm-smi"]))
        checks.append(_run_command("amd-smi", ["amd-smi"]))
        checks.append(_run_command("hipcc", ["hipcc", "--version"]))
    elif accelerator == "cpu":
        return AcceleratorCheckResult(
            accelerator=accelerator,
            ctranslate2_version=None,
            ctranslate2_gpu_count=None,
            system_checks=[],
            faster_whisper_gpu_ok=False,
            error=None,
        )

    version, gpu_count = _ct2_version_and_gpu_count()
    fw_ok, error = check_faster_whisper_gpu(settings)

    return AcceleratorCheckResult(
        accelerator=accelerator,
        ctranslate2_version=version,
        ctranslate2_gpu_count=gpu_count,
        system_checks=checks,
        faster_whisper_gpu_ok=fw_ok,
        error=error,
    )
