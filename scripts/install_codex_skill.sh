#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE_DIR="$PROJECT_DIR/codex/skills/audio-transcription"
CODEX_DIR="${CODEX_HOME:-$HOME/.codex}"
TARGET_DIR="$CODEX_DIR/skills/audio-transcription"

if [ ! -d "$SOURCE_DIR" ]; then
  echo "Skill source not found: $SOURCE_DIR" >&2
  exit 1
fi

mkdir -p "$CODEX_DIR/skills"
rm -rf "$TARGET_DIR"
cp -R "$SOURCE_DIR" "$TARGET_DIR"
printf '%s\n' "$PROJECT_DIR" > "$TARGET_DIR/.project-dir"
chmod +x "$TARGET_DIR/scripts/transcribe_audio.sh"

echo "Installed Codex skill: $TARGET_DIR"
echo "Project path recorded in: $TARGET_DIR/.project-dir"
