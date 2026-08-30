# Technical Guide

This document captures the lower-level configuration and operational details for the audio transcription project. For a quick start, use the README.

## Requirements

- Python 3.10, 3.11, or 3.12
- NVIDIA GPU and visible `nvidia-smi` for CUDA mode
- CUDA 12 cuBLAS and cuDNN 9 runtime libraries for faster-whisper/CTranslate2
- OpenAI API key only when using `TRANSCRIPTION_BACKEND=openai-whisper` or `TRANSCRIPTION_BACKEND=openai-realtime-whisper`

Faster-whisper decodes audio through PyAV, so system FFmpeg is not required for the default file transcription pipeline.

Local speaker labeling is available as an optional post-processing step. See
[Local Speaker Diarization](diarization.md) for its pinned dependencies, gated model access, and
commands.

If you run this project from a sandboxed agent session, CUDA checks and CUDA transcription commands should be executed outside the sandbox. Sandboxed sessions can block GPU access and surface misleading CUDA initialization errors even when the host NVIDIA/WSL setup is healthy.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev,cuda]"
cp .env.example .env
source scripts/setup_cuda_env.sh
```

If you use `uv`:

```bash
uv venv
source .venv/bin/activate
uv pip install -e ".[dev,cuda]"
cp .env.example .env
source scripts/setup_cuda_env.sh
```

The CUDA library helper uses the NVIDIA Python wheels:

```bash
python -m pip install nvidia-cublas-cu12 nvidia-cudnn-cu12==9.*
source scripts/setup_cuda_env.sh
```

## Configuration

Local faster-whisper is the default backend:

```env
TRANSCRIPTION_BACKEND=faster-whisper
WHISPER_MODEL_NAME=large-v3
WHISPER_DEVICE=cuda
WHISPER_COMPUTE_TYPE=float16
HF_TOKEN=hf_your_token_here
```

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

Check CUDA:

```bash
audio-transcribe check-cuda
```

Run `audio-transcribe check-cuda` outside the sandbox if you want to validate real GPU availability.

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

`<original_file_name>_transcript.md` is the human-readable transcript with segment timestamps when the backend provides them. `<original_file_name>_metadata.json` records source hash, backend, model, device, compute type, batch size, language, VAD settings, and segment count. Existing outputs are skipped only when both files exist and the source hash plus transcription configuration still match.

## Model Settings

Default CUDA quality mode:

```env
WHISPER_MODEL_NAME=large-v3
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
WHISPER_DEVICE=cpu
WHISPER_COMPUTE_TYPE=int8
WHISPER_BATCH_SIZE=1
```

The first local transcription may download the selected faster-whisper model. Later runs reuse the local cache. Use `HF_TOKEN` if unauthenticated Hugging Face downloads are slow or rate limited.
