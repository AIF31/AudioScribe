# KADO Transcription

CUDA-first Spanish interview transcription for KADO using
[SYSTRAN/faster-whisper](https://github.com/SYSTRAN/faster-whisper).

V1 is transcription-only. It does not include diarization, speaker labels, pyannote, NeMo,
WhisperX, or LLM insight extraction.

## Requirements

- Python 3.10, 3.11, or 3.12
- NVIDIA GPU and visible `nvidia-smi` for CUDA mode
- CUDA 12 cuBLAS and cuDNN 9 runtime libraries for faster-whisper/CTranslate2

Faster-whisper decodes audio through PyAV, so system FFmpeg is not required for this v1
pipeline.

If you run this project from a sandboxed agent session, CUDA checks and CUDA transcription
commands should be executed outside the sandbox. Sandboxed sessions can block GPU access
and surface misleading CUDA initialization errors even when the host NVIDIA/WSL setup is
healthy.

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

## Usage

Optional: set a Hugging Face token in `.env` before the first model download for higher
rate limits:

```env
HF_TOKEN=hf_your_token_here
```

Create a token at <https://huggingface.co/settings/tokens>. Keep `.env` private.

Check configuration:

```bash
kado-transcribe inspect-config
```

Check CUDA:

```bash
kado-transcribe check-cuda
```

Run `kado-transcribe check-cuda` outside the sandbox if you want to validate real GPU
availability.

Transcribe one file:

```bash
kado-transcribe transcribe-file ./data/audio_raw/interview_001.mp3
```

If `WHISPER_DEVICE=cuda`, run the transcription command outside the sandbox.

Transcribe a batch:

```bash
kado-transcribe transcribe-batch \
  --input-dir ./data/audio_raw \
  --output-dir ./data/transcripts
```

Module invocation also works:

```bash
python -m kado_transcriber.cli transcribe-batch
```

## Outputs

Each interview writes:

```text
data/transcripts/interview_001/
  interview_001_transcript.md
  interview_001_metadata.json
```

`<original_file_name>_transcript.md` is the human-readable transcript with segment
timestamps. `<original_file_name>_metadata.json` records source hash, model, device,
compute type, batch size, language, VAD settings, and segment count. Existing outputs are
skipped only when both files exist and the source hash plus transcription configuration
still match.

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

For KADO interviews, Spanish is the default language:

```env
WHISPER_LANGUAGE=es
WHISPER_TASK=transcribe
```

## Quality Review

Before running all long interviews, transcribe a short representative file and review
`transcript.md`. If CUDA memory fails, reduce `WHISPER_BATCH_SIZE` or switch to
`WHISPER_COMPUTE_TYPE=int8_float16`.

## Troubleshooting

If `nvidia-smi` is not found, fix NVIDIA/WSL GPU visibility before debugging Python.

If a CUDA command fails in a sandboxed session, rerun the same command outside the sandbox
before assuming the host driver or WSL GPU setup is broken.

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

The first transcription may download the selected faster-whisper model. Later runs reuse
the local cache. Use `HF_TOKEN` if unauthenticated Hugging Face downloads are slow or rate
limited.
