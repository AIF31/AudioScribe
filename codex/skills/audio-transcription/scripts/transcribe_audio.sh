#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${AUDIO_TRANSCRIPTION_PROJECT_DIR:-}"
if [ -z "$PROJECT_DIR" ] && [ -f "$SCRIPT_DIR/../.project-dir" ]; then
  PROJECT_DIR="$(cat "$SCRIPT_DIR/../.project-dir")"
fi
if [ -z "$PROJECT_DIR" ]; then
  echo "Set AUDIO_TRANSCRIPTION_PROJECT_DIR or reinstall this skill from the project clone." >&2
  exit 2
fi

INPUT_DIR="$PROJECT_DIR/data/audio_raw"
OUTPUT_DIR="$PROJECT_DIR/data/transcripts"
SUPPORTED_REGEX='\.(aac|flac|m4a|mov|mp3|mp4|ogg|wav|webm)$'

if [ "$#" -lt 1 ]; then
  echo "Usage: transcribe_audio.sh <audio-file-or-dir> [...]" >&2
  exit 2
fi

mkdir -p "$INPUT_DIR" "$OUTPUT_DIR"

copy_media() {
  local src="$1"
  local base
  base="$(basename "$src")"
  if [ "$(realpath "$src")" != "$(realpath "$INPUT_DIR/$base" 2>/dev/null || true)" ]; then
    cp -n "$src" "$INPUT_DIR/$base"
  fi
}

for target in "$@"; do
  if [ -d "$target" ]; then
    while IFS= read -r -d '' file; do
      if [[ "$file" =~ $SUPPORTED_REGEX ]]; then
        copy_media "$file"
      fi
    done < <(find "$target" -type f -print0)
  elif [ -f "$target" ]; then
    if [[ "$target" =~ $SUPPORTED_REGEX ]]; then
      copy_media "$target"
    else
      echo "Skipping unsupported file: $target" >&2
    fi
  else
    echo "Not found: $target" >&2
    exit 1
  fi
done

cd "$PROJECT_DIR"
source .venv/bin/activate

backend="${TRANSCRIPTION_BACKEND:-}"
if [ -z "$backend" ] && [ -f "$PROJECT_DIR/.env" ]; then
  backend="$(grep -E '^TRANSCRIPTION_BACKEND=' "$PROJECT_DIR/.env" | tail -n1 | cut -d= -f2- | tr -d '"'\''[:space:]')"
fi
backend="${backend:-faster-whisper}"

if [ "$backend" = "faster-whisper" ]; then
  source scripts/setup_cuda_env.sh
fi

audio-transcribe transcribe-batch --input-dir "$INPUT_DIR" --output-dir "$OUTPUT_DIR"
find "$OUTPUT_DIR" -maxdepth 2 -type f \( -name '*_transcript.md' -o -name '*_metadata.json' \) -print
