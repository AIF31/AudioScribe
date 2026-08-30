"""Verify large-v3-turbo loads and transcribes on CUDA, mirroring app settings."""
import time

from dotenv import load_dotenv

load_dotenv()

from audio_transcriber.config import get_settings

settings = get_settings()
print(f"Settings: model={settings.whisper_model_name} device={settings.whisper_device} "
      f"compute={settings.whisper_compute_type} batch={settings.whisper_batch_size} "
      f"lang={settings.whisper_language}")

from faster_whisper import BatchedInferencePipeline, WhisperModel

t0 = time.perf_counter()
model = WhisperModel(
    settings.whisper_model_name,
    device=settings.whisper_device,
    compute_type=settings.whisper_compute_type,
    use_auth_token=settings.hf_token,
)
pipeline = BatchedInferencePipeline(model=model)
print(f"MODEL_LOAD_OK in {time.perf_counter() - t0:.1f}s")

# Decode the first 30 seconds of a real recording for an end-to-end GPU test.
import av
import numpy as np

media = sorted((settings.input_audio_dir).glob("**/*"))
media = [p for p in media if p.suffix.lower() in {".m4a", ".mp3", ".wav", ".ogg", ".flac", ".mp4", ".mov", ".webm"}]
if not media:
    print("NO_MEDIA_FOUND: skipping transcribe test")
else:
    path = media[0]
    container = av.open(str(path))
    stream = next(s for s in container.streams if s.type == "audio")
    resampler = av.AudioResampler(format="s16", layout="mono", rate=16000)
    chunks: list[np.ndarray] = []
    for packet in container.demux(stream):
        for frame in packet.decode():
            for resampled in resampler.resample(frame):
                chunks.append(resampled.to_ndarray().reshape(1, -1))
        if chunks and sum(c.shape[1] for c in chunks) >= 16000 * 30:
            break
    container.close()
    audio = np.concatenate(chunks, axis=1).astype(np.float32) / 32768.0
    audio = audio[0, : 16000 * 30]

    t0 = time.perf_counter()
    segments, info = pipeline.transcribe(
        audio,
        language=settings.whisper_language,
        task=settings.whisper_task,
        beam_size=settings.whisper_beam_size,
        vad_filter=settings.whisper_vad_filter,
        batch_size=settings.whisper_batch_size,
    )
    text = " ".join(s.text for s in segments)
    elapsed = time.perf_counter() - t0
    print(f"TRANSCRIBE_OK lang={info.language} prob={info.language_probability:.2f} "
          f"30s_audio_in_{elapsed:.2f}s ({30 / elapsed:.0f}x realtime)")
    print(f"SAMPLE_TEXT: {text[:200]}")

print("VERIFICATION_PASSED")
