# Codex Skill

This repository includes the `audio-transcription` Codex skill so a fresh clone can install the same local workflow used for this project.

## Install After Clone

From the repository root:

```bash
scripts/install_codex_skill.sh
```

The installer copies `codex/skills/audio-transcription` into:

```text
${CODEX_HOME:-$HOME/.codex}/skills/audio-transcription
```

It also writes the current clone path to `.project-dir` inside the installed skill, so the helper script can find this project without hardcoded machine paths.

## Use

Restart Codex after installing the skill so the new skill metadata is discovered. Then ask Codex to use `$audio-transcription`, or ask it to transcribe an audio/video file.

Local faster-whisper mode:

```bash
${CODEX_HOME:-$HOME/.codex}/skills/audio-transcription/scripts/transcribe_audio.sh ./data/audio_raw/example.m4a
```

OpenAI cloud mode:

```bash
TRANSCRIPTION_BACKEND=openai-whisper OPENAI_WHISPER_MODEL=whisper-1 \
  ${CODEX_HOME:-$HOME/.codex}/skills/audio-transcription/scripts/transcribe_audio.sh ./data/audio_raw/example.m4a
```

## Configuration

Copy `.env.example` to `.env` and put real secrets only in `.env`.

- `HF_TOKEN` is optional and only needed for higher Hugging Face model download limits.
- `OPENAI_API_KEY` is required for `TRANSCRIPTION_BACKEND=openai-whisper` or `TRANSCRIPTION_BACKEND=openai-realtime-whisper`.
- Use `audio-transcribe inspect-config` and `audio-transcribe check-accelerator` to validate local runtime configuration.
- AMD ROCm/HIP support is experimental. For AMD hosts, use `.env.rocm.example`, keep `WHISPER_DEVICE=cuda`, and validate with a small file before large batches.
- `.env` is ignored by Git and must not be committed.

## Skill Source

The source of truth for the skill lives in:

```text
codex/skills/audio-transcription/
```

If the skill changes, update that directory first, then rerun `scripts/install_codex_skill.sh`.
