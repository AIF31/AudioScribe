from audio_transcriber.accelerator_check import AcceleratorCheckResult, check_accelerator
from audio_transcriber.config import Settings


def check_cuda() -> AcceleratorCheckResult:
    settings = Settings(whisper_accelerator="cuda", whisper_device="cuda")
    return check_accelerator(settings)
