# Audio Transcription

Audio Transcription is a small command-line project for turning audio and video files into Markdown transcripts. It can run locally with faster-whisper, or use OpenAI cloud transcription when you prefer an API-based workflow.

The project is useful for lectures, calls, voice notes, meetings, podcasts, research recordings, and other media files where you want a clean transcript plus metadata.

## What It Does

- Transcribes individual audio/video files or whole folders.
- Supports common media formats like `.mp3`, `.m4a`, `.wav`, `.mp4`, `.mov`, `.webm`, `.ogg`, `.flac`, and `.aac`.
- Writes a readable Markdown transcript for each file.
- Writes metadata with the source hash, model, backend, language, and segment count.
- Skips files that were already transcribed with the same settings.
- Lets you choose between local transcription and OpenAI cloud transcription.

## Choose A Backend

Local mode runs on your machine:

```env
TRANSCRIPTION_BACKEND=faster-whisper
```

Use this when you want local processing, lower API cost, or GPU acceleration with CUDA. The first run may download a faster-whisper model from Hugging Face.

OpenAI cloud mode sends the media file to the OpenAI Audio Transcriptions API:

```env
TRANSCRIPTION_BACKEND=openai-whisper
OPENAI_API_KEY=sk_your_openai_api_key_here
OPENAI_WHISPER_MODEL=whisper-1
```

Use this when you want a simple cloud-backed transcription path and have an OpenAI API key.

## Quick Start

Create the environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev,cuda]"
cp .env.example .env
```

For CPU-only or OpenAI cloud usage, installing without CUDA extras is enough:

```bash
python -m pip install -e ".[dev]"
```

Edit `.env` and choose your backend. Keep real API keys only in `.env`; it is ignored by Git.

## Transcribe A File

Put media files in `data/audio_raw`, then run:

```bash
audio-transcribe transcribe-file ./data/audio_raw/example.m4a
```

Transcripts are written to:

```text
data/transcripts/example/
  example_transcript.md
  example_metadata.json
```

## Transcribe A Folder

```bash
audio-transcribe transcribe-batch \
  --input-dir ./data/audio_raw \
  --output-dir ./data/transcripts
```

## Recommended Settings

For NVIDIA GPU local transcription:

```env
TRANSCRIPTION_BACKEND=faster-whisper
WHISPER_MODEL_NAME=large-v3
WHISPER_DEVICE=cuda
WHISPER_COMPUTE_TYPE=float16
WHISPER_BATCH_SIZE=8
```

For CPU local transcription:

```bash
cp .env.cpu.example .env
```

For OpenAI cloud transcription:

```env
TRANSCRIPTION_BACKEND=openai-whisper
OPENAI_API_KEY=sk_your_openai_api_key_here
OPENAI_WHISPER_MODEL=whisper-1
```

## Codex Skill

This repo includes a reusable Codex skill at `codex/skills/audio-transcription`. After a fresh clone, install it into `${CODEX_HOME:-$HOME/.codex}/skills`:

```bash
scripts/install_codex_skill.sh
```

See [docs/codex-skill.md](docs/codex-skill.md) for details.

## More Details

For CUDA setup, realtime settings, configuration reference, and troubleshooting, see [docs/technical-guide.md](docs/technical-guide.md).
