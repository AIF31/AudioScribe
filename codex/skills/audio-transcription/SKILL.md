---
name: audio-transcription
description: Transcribe audio or video files with this transcription project using local faster-whisper or OpenAI cloud transcription. Use when Codex needs to transcribe, process, or convert recordings, meetings, lectures, notes, podcasts, .m4a/.mp3/.wav/.mp4/.mov/.webm/.ogg/.flac media, or any provided audio/video file into Markdown transcripts.
---

# Audio Transcription

## Overview

Use this repository to transcribe audio and video files. The default backend is local `SYSTRAN/faster-whisper`; the project also supports OpenAI cloud file transcription with `TRANSCRIPTION_BACKEND=openai-whisper` and an `OPENAI_API_KEY` in `.env`. Outputs are Markdown transcript files plus metadata JSON, named from the original media stem. CUDA and ROCm/HIP runs must be executed outside the sandbox because sandboxed sessions can block GPU access and surface misleading initialization errors.

## Workflow

1. Resolve the audio paths the user provided. If they gave a directory, collect supported media files recursively.
2. Prefer the helper script. Local faster-whisper is the default and uses CUDA when `.env` selects `WHISPER_DEVICE=cuda`; run CUDA jobs outside the sandbox:

```bash
${CODEX_HOME:-$HOME/.codex}/skills/audio-transcription/scripts/transcribe_audio.sh <audio-file-or-dir> [...]
```

3. For OpenAI cloud transcription, ensure `.env` contains `OPENAI_API_KEY`, then override the backend for the run. This mode does not require CUDA:

```bash
TRANSCRIPTION_BACKEND=openai-whisper OPENAI_WHISPER_MODEL=whisper-1 ${CODEX_HOME:-$HOME/.codex}/skills/audio-transcription/scripts/transcribe_audio.sh <audio-file-or-dir> [...]
```

4. If the helper cannot be used, run the project manually. For local CUDA mode, this also must be run outside the sandbox:

```bash
cd "$AUDIO_TRANSCRIPTION_PROJECT_DIR"
source .venv/bin/activate
source scripts/setup_cuda_env.sh
audio-transcribe transcribe-batch --input-dir ./data/audio_raw --output-dir ./data/transcripts
```

5. For a one-off OpenAI cloud file transcription without CUDA setup:

```bash
cd "$AUDIO_TRANSCRIPTION_PROJECT_DIR"
source .venv/bin/activate
TRANSCRIPTION_BACKEND=openai-whisper OPENAI_WHISPER_MODEL=whisper-1 audio-transcribe transcribe-file ./data/audio_raw/example.m4a --output-dir ./data/transcripts
```

6. Report the generated transcript paths. Do not paste full transcripts unless the user asks.

## Behavior

- Copy supplied media files into `$AUDIO_TRANSCRIPTION_PROJECT_DIR/data/audio_raw` before transcription unless they are already there.
- Supported extensions: `.aac`, `.flac`, `.m4a`, `.mov`, `.mp3`, `.mp4`, `.ogg`, `.wav`, `.webm`.
- Expected output for `sample_001.m4a`:
  - `$AUDIO_TRANSCRIPTION_PROJECT_DIR/data/transcripts/sample_001/sample_001_transcript.md`
  - `$AUDIO_TRANSCRIPTION_PROJECT_DIR/data/transcripts/sample_001/sample_001_metadata.json`
- The pipeline skips existing outputs when source hash and config match.
- `TRANSCRIPTION_BACKEND=faster-whisper` uses the local model and can use `HF_TOKEN` from `.env` when model downloads need higher Hugging Face limits.
- `TRANSCRIPTION_BACKEND=openai-whisper` uses the OpenAI Audio Transcriptions API and requires `OPENAI_API_KEY` in `.env`. Set `OPENAI_WHISPER_MODEL=whisper-1` unless the project is updated for another supported file transcription model.
- Prefer `audio-transcribe check-accelerator` for local runtime validation.
- `audio-transcribe check-cuda` remains available as a CUDA compatibility check.
- AMD ROCm/HIP support is experimental. For AMD hosts, use `.env.rocm.example`, set `WHISPER_ACCELERATOR=rocm`, keep `WHISPER_DEVICE=cuda`, and validate with `audio-transcribe check-accelerator` plus a small file before large batches.
- `audio-transcribe check-accelerator` and any `WHISPER_DEVICE=cuda` local transcription run should be executed outside the sandbox.

## Troubleshooting

- If model download fails with DNS/network errors, rerun the same command with network approval.
- If OpenAI cloud transcription fails with authentication errors, confirm `.env` contains a valid `OPENAI_API_KEY` and that the key has access to the Audio Transcriptions API.
- If CUDA library loading fails in a sandboxed session, rerun the same CUDA command outside the sandbox before assuming the host driver is broken.
- If CUDA library loading fails outside the sandbox, ensure the virtualenv is active and `source scripts/setup_cuda_env.sh` ran.
- If CUDA memory fails, use `.env.cuda.low-vram.example` settings or set `WHISPER_BATCH_SIZE=4` and `WHISPER_COMPUTE_TYPE=int8_float16`.
- If the user wants CPU fallback, copy `.env.cpu.example` to `.env` or set `WHISPER_DEVICE=cpu`, `WHISPER_COMPUTE_TYPE=int8`, and `WHISPER_BATCH_SIZE=1`.
- CPU transcription is significantly slower than CUDA and is only recommended for short or small files. Large recordings may take hours.
- If OpenAI cloud transcription fails with a 413 error, the file exceeds the 25 MB upload limit. Use the `faster-whisper` backend or split the media.
