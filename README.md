# AudioScribe

<p align="center">
  <img src="Images/AudioScribe.png" alt="AudioScribe local-first audio transcription" width="100%">
</p>

<p align="center">
  <strong>Local-first audio and video transcription with a simple CLI, reproducible outputs, and optional OpenAI cloud transcription.</strong>
</p>

<p align="center">
  <a href="#quick-start">Quick start</a>
  ·
  <a href="#backends">Backends</a>
  ·
  <a href="#codex-skill">Codex skill</a>
  ·
  <a href="docs/technical-guide.md">Technical guide</a>
</p>

<p align="center">
  <img alt="Python 3.10-3.12" src="https://img.shields.io/badge/python-3.10--3.12-39d5ff">
  <img alt="Local backend" src="https://img.shields.io/badge/local-faster--whisper-59f0c8">
  <img alt="Cloud backend" src="https://img.shields.io/badge/cloud-OpenAI%20Whisper-39d5ff">
  <img alt="Interface" src="https://img.shields.io/badge/interface-CLI-0b1220">
</p>

AudioScribe turns recordings into clean Markdown transcripts and metadata files. It is built for people who want a practical transcription pipeline they can run from the terminal: local when privacy or cost matters, cloud-backed when convenience matters.

## Why AudioScribe

- **Local-first by default**: use `faster-whisper` on CPU or CUDA without sending media to an external service.
- **Cloud when you want it**: switch to OpenAI transcription with one `.env` setting and an API key.
- **Batch-friendly outputs**: transcribe a file or a folder and get one organized output directory per source file.
- **Reproducible skips**: completed transcripts are skipped when the source hash and transcription settings still match.
- **Codex-ready**: includes an installable Codex skill so agents can use the project after a fresh clone.
- **Common formats**: supports `.mp3`, `.m4a`, `.wav`, `.mp4`, `.mov`, `.webm`, `.ogg`, `.flac`, and `.aac`.

## Quick Start

```bash
git clone https://github.com/AIF31/AudioScribe.git
cd AudioScribe

python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"

cp .env.example .env
audio-transcribe transcribe-file ./data/audio_raw/example.m4a
```

For NVIDIA GPU acceleration, install the CUDA extras:

```bash
python -m pip install -e ".[dev,cuda]"
source scripts/setup_cuda_env.sh
```

## Backends

Choose the backend in `.env`.

```env
TRANSCRIPTION_BACKEND=faster-whisper
WHISPER_MODEL_NAME=large-v3
WHISPER_DEVICE=cpu
WHISPER_COMPUTE_TYPE=int8
```

Use local mode when you want transcription to run on your own machine. Switch `WHISPER_DEVICE=cuda` and `WHISPER_COMPUTE_TYPE=float16` for GPU-backed runs. CPU mode (`WHISPER_DEVICE=cpu`) is significantly slower and only recommended for short or small files.

```env
TRANSCRIPTION_BACKEND=openai-whisper
OPENAI_API_KEY=sk_your_openai_api_key_here
OPENAI_WHISPER_MODEL=whisper-1
```

Use cloud mode when you prefer an API-backed workflow. Keep real keys only in `.env`; the file is ignored by Git. Note: the OpenAI Audio Transcriptions API has a 25 MB file upload limit. For larger files, use the `faster-whisper` local backend.

AudioScribe creates a predictable output folder for each input file:

```text
data/transcripts/example/
  example_transcript.md
  example_metadata.json
```

The transcript is written for reading and review. The metadata records the backend, model, device, language, source hash, and segment count so repeated runs can be compared safely.

## CLI

```bash
# Inspect the active configuration
audio-transcribe inspect-config

# Transcribe one file
audio-transcribe transcribe-file ./data/audio_raw/example.m4a

# Transcribe a folder
audio-transcribe transcribe-batch \
  --input-dir ./data/audio_raw \
  --output-dir ./data/transcripts
```

## Codex Skill

AudioScribe ships with a Codex skill in `codex/skills/audio-transcription`. Install it after cloning:

```bash
scripts/install_codex_skill.sh
```

After restarting Codex, use the `audio-transcription` skill to transcribe audio or video files through the same local or cloud backends. See [docs/codex-skill.md](docs/codex-skill.md).

## Documentation

- [Technical guide](docs/technical-guide.md): CUDA setup, realtime settings, configuration reference, and troubleshooting.
- [Codex skill guide](docs/codex-skill.md): installing and using the bundled Codex skill.
- [.env.example](.env.example): documented configuration template with safe placeholder values.

## Project Status

AudioScribe is intentionally small: a focused CLI, clear configuration, and transcript outputs that are easy to review or feed into downstream workflows.
