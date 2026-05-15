# Technical Guide

This document captures the lower-level configuration and operational details for the audio transcription project. For a quick start, use the README.

## Requirements

- Python 3.10, 3.11, or 3.12
- No GPU runtime for CPU mode
- NVIDIA GPU and visible `nvidia-smi` for CUDA mode
- CUDA 12 cuBLAS and cuDNN 9 runtime libraries for NVIDIA faster-whisper/CTranslate2
- AMD GPU supported by the installed ROCm/HIP SDK/runtime for experimental ROCm/HIP mode
- CTranslate2 >= 4.7.1 from a ROCm/HIP wheel or source build with `-DWITH_HIP=ON` for AMD GPU mode
- OpenAI API key only when using `TRANSCRIPTION_BACKEND=openai-whisper` or `TRANSCRIPTION_BACKEND=openai-realtime-whisper`

Faster-whisper decodes audio through PyAV, so system FFmpeg is not required for the default file transcription pipeline.

If you run this project from a sandboxed agent session, GPU checks and GPU transcription commands should be executed outside the sandbox. Sandboxed sessions can block GPU access and surface misleading CUDA, HIP, or ROCm initialization errors even when the host setup is healthy.

ROCm/HIP support is experimental. For AMD GPU mode, set `WHISPER_ACCELERATOR=rocm` and keep `WHISPER_DEVICE=cuda`; CTranslate2/faster-whisper still use `device="cuda"` for GPU execution on ROCm builds. Validate AMD GPU runs with a small file before using large batches.

## Install

Base CPU/OpenAI install:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
cp .env.example .env
```

NVIDIA CUDA install:

```bash
python -m pip install -e ".[dev,cuda]"
source scripts/setup_cuda_env.sh
cp .env.cuda.low-vram.example .env
audio-transcribe check-accelerator
```

AMD ROCm/HIP install:

```bash
python -m pip install -e ".[dev]"
# Install a ROCm/HIP-enabled CTranslate2 wheel or build CTranslate2 with -DWITH_HIP=ON.
cp .env.rocm.example .env
audio-transcribe check-accelerator
```

Do not install `.[cuda]` for AMD GPUs. That extra installs NVIDIA libraries only.

## Configuration

Local faster-whisper is the default backend:

```env
TRANSCRIPTION_BACKEND=faster-whisper
WHISPER_MODEL_NAME=large-v3
WHISPER_ACCELERATOR=auto
WHISPER_DEVICE=cuda
WHISPER_COMPUTE_TYPE=float16
HF_TOKEN=hf_your_token_here
```

Accelerator settings:

| Setting | Meaning |
|---|---|
| `WHISPER_ACCELERATOR=auto` | Default label. Uses `WHISPER_DEVICE` to infer CPU or CUDA-style GPU metadata. |
| `WHISPER_ACCELERATOR=cpu` | CPU-only local transcription; requires `WHISPER_DEVICE=cpu` for `faster-whisper`. |
| `WHISPER_ACCELERATOR=cuda` | NVIDIA CUDA local transcription; requires `WHISPER_DEVICE=cuda`. |
| `WHISPER_ACCELERATOR=rocm` | AMD ROCm/HIP local transcription; requires `WHISPER_DEVICE=cuda` because CTranslate2 uses `cuda` as the GPU device string. |

OpenAI backends ignore local accelerator/device compatibility because no local faster-whisper model is loaded.

OpenAI cloud file mode uses the OpenAI Audio API:

```env
TRANSCRIPTION_BACKEND=openai-whisper
OPENAI_API_KEY=sk_your_openai_api_key_here
OPENAI_WHISPER_MODEL=whisper-1
```

OpenAI Realtime Whisper settings are also available:

```env
TRANSCRIPTION_BACKEND=openai-realtime-whisper
OPENAI_API_KEY=sk_your_openai_api_key_here
OPENAI_REALTIME_MODEL=gpt-realtime-whisper
OPENAI_REALTIME_URL=wss://api.openai.com/v1/realtime?intent=transcription
OPENAI_REALTIME_AUDIO_RATE=24000
OPENAI_REALTIME_TURN_DETECTION=server_vad
OPENAI_REALTIME_NOISE_REDUCTION=near_field
OPENAI_REALTIME_TIMEOUT_SECONDS=120
```

The `openai-whisper` backend uploads the media file to the OpenAI Audio API, using the transcriptions endpoint when `WHISPER_TASK=transcribe` and the translations endpoint when `WHISPER_TASK=translate`. The `openai-realtime-whisper` backend decodes existing media files locally, resamples them to 24 kHz mono PCM, and streams the audio to a server-to-server Realtime transcription session.

## Commands

Check configuration:

```bash
audio-transcribe inspect-config
```

Check the configured accelerator:

```bash
audio-transcribe check-accelerator
```

Run `audio-transcribe check-accelerator` outside the sandbox if you want to validate real GPU availability. For ROCm/HIP, this command checks CTranslate2 version, GPU count when available, ROCm/HIP diagnostic tools, and whether faster-whisper can load a tiny GPU model.

Legacy CUDA-only check:

```bash
audio-transcribe check-cuda
```

Transcribe one file:

```bash
audio-transcribe transcribe-file ./data/audio_raw/sample.mp3
```

If `WHISPER_DEVICE=cuda`, run the transcription command outside the sandbox.

Transcribe a batch:

```bash
audio-transcribe transcribe-batch \
  --input-dir ./data/audio_raw \
  --output-dir ./data/transcripts
```

Module invocation also works:

```bash
python -m audio_transcriber.cli transcribe-batch
```

## Outputs

Each source file writes:

```text
data/transcripts/sample/
  sample_transcript.md
  sample_metadata.json
```

`<original_file_name>_transcript.md` is the human-readable transcript with segment timestamps when the backend provides them. `<original_file_name>_metadata.json` records source hash, backend, model, accelerator, device, compute type, batch size, language, VAD settings, and segment count. Existing outputs are skipped only when both files exist and the source hash plus transcription configuration still match.

For a successful AMD GPU run, metadata should include:

```json
{
  "requested_accelerator": "rocm",
  "requested_device": "cuda",
  "effective_device": "cuda",
  "accelerator": "rocm",
  "device": "cuda"
}
```

If `accelerator` is `cpu`, the run used CPU fallback instead of the AMD GPU.

## Model Settings

Default CUDA quality mode:

```env
WHISPER_MODEL_NAME=large-v3
WHISPER_ACCELERATOR=auto
WHISPER_DEVICE=cuda
WHISPER_COMPUTE_TYPE=float16
WHISPER_BATCH_SIZE=8
```

Lower-VRAM CUDA mode:

```bash
cp .env.cuda.low-vram.example .env
```

CPU fallback:

```bash
cp .env.cpu.example .env
```

Experimental AMD ROCm/HIP mode:

```bash
cp .env.rocm.example .env
audio-transcribe check-accelerator
```

For a strict AMD GPU smoke test, temporarily set:

```env
WHISPER_ALLOW_CPU_FALLBACK=false
```

This makes ROCm initialization failures explicit instead of silently completing on CPU.

Set language when you know it:

```env
WHISPER_LANGUAGE=en
WHISPER_TASK=transcribe
```

Use `WHISPER_LANGUAGE=auto` only if your workflow is updated to support automatic language detection consistently.

## Quality Review

Before running a large batch, transcribe a short representative file and review the transcript. If CUDA memory fails, reduce `WHISPER_BATCH_SIZE` or switch to `WHISPER_COMPUTE_TYPE=int8_float16`.

## Troubleshooting

If `nvidia-smi` is not found, fix NVIDIA/WSL GPU visibility before debugging Python.

If a CUDA command fails in a sandboxed session, rerun the same command outside the sandbox before assuming the host driver or WSL GPU setup is broken.

If CUDA libraries are not found:

```bash
python -m pip install nvidia-cublas-cu12 nvidia-cudnn-cu12==9.*
source scripts/setup_cuda_env.sh
```

If CUDA runs out of memory:

```env
WHISPER_BATCH_SIZE=4
WHISPER_COMPUTE_TYPE=int8_float16
```

If it still fails:

```env
WHISPER_BATCH_SIZE=2
```

or use CPU fallback:

```env
WHISPER_ACCELERATOR=cpu
WHISPER_DEVICE=cpu
WHISPER_COMPUTE_TYPE=int8
WHISPER_BATCH_SIZE=1
```

If ROCm/HIP fails, confirm your AMD GPU is supported by the installed ROCm/HIP runtime, run `rocminfo`, `rocm-smi`, `amd-smi`, or `hipcc --version` when available, and confirm the installed CTranslate2 package is a ROCm/HIP-enabled wheel or source build. Do not install `.[cuda]` for AMD GPUs.

If OpenAI cloud transcription fails with a 413 error, the media file exceeds the 25 MB upload limit of the OpenAI Audio Transcriptions API (`Maximum content size limit (26214400) exceeded`). Switch to the `faster-whisper` backend for large files, or split the media into smaller segments before uploading.

CPU transcription is significantly slower than CUDA and is only recommended for short or small files. Large or long recordings may take hours to complete. If you have an NVIDIA GPU available, switch to `WHISPER_DEVICE=cuda` for practical performance.

The first local transcription may download the selected faster-whisper model. Later runs reuse the local cache. Use `HF_TOKEN` if unauthenticated Hugging Face downloads are slow or rate limited.
