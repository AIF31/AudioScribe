# KADO Interview Transcription Pipeline — CUDA-First Codex Implementation Plan

## 2026-04-24 Implementation Update

The implemented v1 output scope was narrowed after review:

- Generate `transcript.md` as the only transcript export.
- Keep `metadata.json` for safe skip/versioning checks.
- Do not generate `.txt`, `.json`, `.jsonl`, or `.srt` transcript files.
- Do not implement FFmpeg preprocessing in v1; `faster-whisper`/PyAV handles decoding directly.

## Goal

Build a local, transcription-only pipeline for KADO interview recordings using `SYSTRAN/faster-whisper`, optimized to run directly on an NVIDIA CUDA GPU.

The system should process long Spanish audio/video interview files, generate clean transcripts, and save structured outputs that can later be used for customer-discovery insight extraction.

This version intentionally avoids speaker recognition, diarization, pyannote, NeMo, WhisperX diarization, or MeetMemo-style speaker labeling because previous experiments failed on the speaker-recognition side. The first milestone is reliable transcription only.

---

## Product Context

KADO is a SaaS platform for nutritionists in Mexico/LatAm. The team has several Spanish customer-discovery interviews as audio/video recordings. The goal is to transcribe those recordings and later extract insights such as:

- Pain points
- Feature requests
- Current workflows
- Pricing signals
- Objections
- Direct quotes
- Software/tooling currently used by nutritionists
- Repeated workflow frictions during consultations

The main language is Spanish. Transcript quality matters because the output will later feed an LLM-based qualitative analysis workflow.

---

## Core Technical Decision

Use `faster-whisper` directly as the transcription engine, with CUDA enabled by default.

Reasoning:

- It is a faster CTranslate2-based implementation of Whisper.
- It supports multilingual transcription, including Spanish.
- It supports CUDA execution on NVIDIA GPUs.
- It supports `float16` for fast GPU inference.
- It supports `int8_float16` for lower VRAM usage on CUDA.
- It supports VAD filtering to remove long non-speech regions.
- It does not require diarization.
- It is simpler and more robust than full meeting-minutes tools for the current use case.

For the first CUDA implementation, default to:

```env
WHISPER_MODEL_NAME=large-v3
WHISPER_DEVICE=cuda
WHISPER_COMPUTE_TYPE=float16
WHISPER_LANGUAGE=es
WHISPER_TASK=transcribe
WHISPER_BEAM_SIZE=5
WHISPER_VAD_FILTER=true
WHISPER_MIN_SILENCE_DURATION_MS=500
WHISPER_CONDITION_ON_PREVIOUS_TEXT=false
WHISPER_BATCH_SIZE=8
```

If CUDA runs out of VRAM, switch to:

```env
WHISPER_COMPUTE_TYPE=int8_float16
WHISPER_BATCH_SIZE=4
```

If CUDA is not available or not correctly configured, use the CPU fallback:

```env
WHISPER_DEVICE=cpu
WHISPER_COMPUTE_TYPE=int8
WHISPER_BATCH_SIZE=1
```

Important hardware note: this CUDA plan assumes an NVIDIA GPU. AMD GPUs will not use CUDA through `faster-whisper`; for AMD machines, keep the CPU fallback path or investigate a separate ROCm-compatible stack later.

---

## CUDA Requirements

The CUDA-first implementation should target the current `faster-whisper` / `CTranslate2` stack:

- NVIDIA GPU
- Recent NVIDIA driver
- CUDA 12 runtime libraries
- cuBLAS for CUDA 12
- cuDNN 9 for CUDA 12
- Python 3.10 or 3.11 preferred

For Linux/WSL, the easiest Python-level CUDA dependency path is:

```bash
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12==9.*
```

Then set `LD_LIBRARY_PATH` before running Python:

```bash
export LD_LIBRARY_PATH=$(python3 -c 'import os; import nvidia.cublas.lib; import nvidia.cudnn.lib; print(os.path.dirname(nvidia.cublas.lib.__file__) + ":" + os.path.dirname(nvidia.cudnn.lib.__file__))')
```

Add this export to the README and optionally to a helper script:

```bash
scripts/setup_cuda_env.sh
```

Example helper script:

```bash
#!/usr/bin/env bash
set -euo pipefail

export LD_LIBRARY_PATH=$(python3 -c 'import os; import nvidia.cublas.lib; import nvidia.cudnn.lib; print(os.path.dirname(nvidia.cublas.lib.__file__) + ":" + os.path.dirname(nvidia.cudnn.lib.__file__))')
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
```

Usage:

```bash
source scripts/setup_cuda_env.sh
```

Do not assume PyTorch CUDA availability is the same thing as CTranslate2 CUDA availability. The tool should test `faster-whisper` directly.

---

## Recommended Milestones

### Milestone 1 — CUDA Environment Validation

Before implementing the full pipeline, add a CUDA smoke test command that validates:

- `nvidia-smi` is available.
- CUDA libraries can be found.
- `faster-whisper` can instantiate a model with `device="cuda"`.
- A short sample can be transcribed with `compute_type="float16"`.

### Milestone 2 — Reliable CUDA Transcription

Implement a local CLI that converts Spanish audio/video interview files into structured transcript packages.

Must include:

- Batch transcription
- Single-file transcription
- CUDA-first defaults
- Spanish as default language
- No diarization
- Segment-level timestamps
- TXT, MD, JSON, JSONL, SRT, and metadata exports
- Config-aware skip/versioning logic
- Tests for config, exporters, and hashing

### Milestone 3 — Quality Review Workflow

Add commands or documented usage for comparing model quality on short samples before processing all interviews.

Must include:

- Ability to run one file or a small preview sample
- Clear model recommendations
- Guidance for comparing `medium`, `large-v3`, and `turbo`
- Guidance for comparing `float16` vs `int8_float16`

### Milestone 4 — Future Insight Extraction

Do not implement in the first build unless explicitly requested.

Prepare the transcript structure so a later LLM workflow can extract:

- Pain points
- Feature requests
- Current workflows
- Pricing signals
- Objections
- Direct quotes with timestamps

---

## Desired Final Capabilities

The finished project should allow the user to run:

```bash
source scripts/setup_cuda_env.sh

python -m kado_transcriber.cli transcribe-batch \
  --input-dir ./data/audio_raw \
  --output-dir ./data/transcripts
```

or, after installing the package:

```bash
source scripts/setup_cuda_env.sh

kado-transcribe transcribe-batch \
  --input-dir ./data/audio_raw \
  --output-dir ./data/transcripts
```

And produce:

```text
data/transcripts/
  entrevista_001/
    transcript.txt
    transcript.md
    transcript.json
    transcript.jsonl
    transcript.srt
    metadata.json
  entrevista_002/
    transcript.txt
    transcript.md
    transcript.json
    transcript.jsonl
    transcript.srt
    metadata.json
```

Each transcript should include segment-level timestamps. Speaker labels are not required.

---

## Repository Structure to Create

Create or refactor the project into this structure:

```text
kado-transcription/
  README.md
  pyproject.toml
  .env.example
  .gitignore
  data/
    audio_raw/
    audio_processed/
    transcripts/
  scripts/
    setup_cuda_env.sh
    smoke_test_cuda.py
  src/
    kado_transcriber/
      __init__.py
      cli.py
      config.py
      audio.py
      transcriber.py
      exporters.py
      hashing.py
      cuda_check.py
      utils.py
  tests/
    test_exporters.py
    test_config.py
    test_hashing.py
```

If the project already exists, adapt the existing structure instead of replacing everything blindly.

---

## Python Environment

Use Python 3.10 or 3.11 if possible. Python 3.12 may work, but if dependency issues appear, prefer Python 3.11.

Use `uv` if available; otherwise support standard `pip`.

---

## Recommended `pyproject.toml`

Use a stable `faster-whisper` pin for the first implementation rather than a loose lower bound from an older release.

```toml
[project]
name = "kado-transcription"
version = "0.1.0"
description = "CUDA-first Spanish interview transcription pipeline for KADO using faster-whisper."
requires-python = ">=3.10,<3.13"
dependencies = [
    "faster-whisper==1.2.1",
    "python-dotenv>=1.0.1",
    "typer>=0.12.0",
    "rich>=13.7.0",
    "pydantic>=2.7.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "ruff>=0.6.0",
]
cuda = [
    "nvidia-cublas-cu12; platform_system == 'Linux'",
    "nvidia-cudnn-cu12==9.*; platform_system == 'Linux'",
]

[project.scripts]
kado-transcribe = "kado_transcriber.cli:app"

[tool.ruff]
line-length = 100

[tool.pytest.ini_options]
testpaths = ["tests"]
```

Install with CUDA dependencies:

```bash
uv pip install -e ".[dev,cuda]"
```

or:

```bash
pip install -e ".[dev,cuda]"
```

If dependency resolution fails on the local machine, Codex may relax the exact pin to:

```toml
"faster-whisper>=1.2.1,<2.0.0"
```

but should first try the exact pin for reproducibility.

---

## `.env.example`

Create a CUDA-first `.env.example`:

```env
# Core faster-whisper settings — CUDA-first
WHISPER_MODEL_NAME=large-v3
WHISPER_DEVICE=cuda
WHISPER_COMPUTE_TYPE=float16
WHISPER_LANGUAGE=es
WHISPER_TASK=transcribe
WHISPER_BEAM_SIZE=5

# Batch inference
# Higher values are faster but use more VRAM.
# Start with 8 on GPUs with around 8GB+ VRAM.
# Use 4 or 2 if CUDA runs out of memory.
WHISPER_BATCH_SIZE=8

# VAD: removes long silence/non-speech regions
WHISPER_VAD_FILTER=true
WHISPER_MIN_SILENCE_DURATION_MS=500

# Long-interview behavior
# false can reduce repeated/hallucinated text in long recordings
WHISPER_CONDITION_ON_PREVIOUS_TEXT=false

# Optional domain prompt for KADO/nutrition interviews
WHISPER_INITIAL_PROMPT=Entrevista en español sobre nutrición, consulta nutricional, expediente clínico, antropometría, IMC, dieta, pacientes, seguimiento y software para nutriólogos.

# Input/output defaults
INPUT_AUDIO_DIR=./data/audio_raw
PROCESSED_AUDIO_DIR=./data/audio_processed
TRANSCRIPTS_OUTPUT_DIR=./data/transcripts

# Export settings
EXPORT_TXT=true
EXPORT_MD=true
EXPORT_JSON=true
EXPORT_JSONL=true
EXPORT_SRT=true

# Optional preprocessing
PREPROCESS_AUDIO=true
AUDIO_SAMPLE_RATE=16000
AUDIO_CHANNELS=1
KEEP_PROCESSED_AUDIO=true

# Behavior
SKIP_EXISTING=true
LOG_LEVEL=INFO
```

Also create `.env.cpu.example` as a fallback:

```env
WHISPER_MODEL_NAME=medium
WHISPER_DEVICE=cpu
WHISPER_COMPUTE_TYPE=int8
WHISPER_LANGUAGE=es
WHISPER_TASK=transcribe
WHISPER_BEAM_SIZE=5
WHISPER_BATCH_SIZE=1
WHISPER_VAD_FILTER=true
WHISPER_MIN_SILENCE_DURATION_MS=500
WHISPER_CONDITION_ON_PREVIOUS_TEXT=false
```

Also create `.env.cuda.low-vram.example`:

```env
WHISPER_MODEL_NAME=large-v3
WHISPER_DEVICE=cuda
WHISPER_COMPUTE_TYPE=int8_float16
WHISPER_LANGUAGE=es
WHISPER_TASK=transcribe
WHISPER_BEAM_SIZE=5
WHISPER_BATCH_SIZE=4
WHISPER_VAD_FILTER=true
WHISPER_MIN_SILENCE_DURATION_MS=500
WHISPER_CONDITION_ON_PREVIOUS_TEXT=false
```

---

## Installation Commands

### With `uv`

```bash
uv venv
source .venv/bin/activate
uv pip install -e ".[dev,cuda]"
source scripts/setup_cuda_env.sh
```

### With `pip`

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,cuda]"
source scripts/setup_cuda_env.sh
```

### System Dependency

Install FFmpeg:

```bash
sudo apt update
sudo apt install -y ffmpeg
```

For Windows/WSL, run the command inside WSL if using a WSL project.

### Confirm NVIDIA GPU Visibility

Run:

```bash
nvidia-smi
```

If `nvidia-smi` fails inside WSL, fix WSL/NVIDIA driver visibility before debugging Python.

---

## CUDA Smoke Test

Add a smoke test script:

```text
scripts/smoke_test_cuda.py
```

Suggested content:

```python
from faster_whisper import WhisperModel


def main() -> None:
    model_name = "tiny"
    print("Loading faster-whisper model on CUDA...")
    model = WhisperModel(model_name, device="cuda", compute_type="float16")
    print("CUDA faster-whisper model loaded successfully:", model_name)


if __name__ == "__main__":
    main()
```

Run:

```bash
source scripts/setup_cuda_env.sh
python scripts/smoke_test_cuda.py
```

Also add a CLI command:

```bash
kado-transcribe check-cuda
```

The command should:

- Print whether `nvidia-smi` is available.
- Print the configured `WHISPER_DEVICE` and `WHISPER_COMPUTE_TYPE`.
- Try loading a tiny model with `device="cuda"` and `compute_type="float16"`.
- Print a clear success/failure message.

---

## Implementation Details

## 1. `config.py`

Create a typed settings class that reads from `.env`.

Required fields:

```python
from pathlib import Path
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os

load_dotenv()


def _env_bool(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "y", "on"}


class Settings(BaseModel):
    whisper_model_name: str = Field(default_factory=lambda: os.getenv("WHISPER_MODEL_NAME", "large-v3"))
    whisper_device: str = Field(default_factory=lambda: os.getenv("WHISPER_DEVICE", "cuda"))
    whisper_compute_type: str = Field(default_factory=lambda: os.getenv("WHISPER_COMPUTE_TYPE", "float16"))
    whisper_language: str = Field(default_factory=lambda: os.getenv("WHISPER_LANGUAGE", "es"))
    whisper_task: str = Field(default_factory=lambda: os.getenv("WHISPER_TASK", "transcribe"))
    whisper_beam_size: int = Field(default_factory=lambda: int(os.getenv("WHISPER_BEAM_SIZE", "5")))
    whisper_batch_size: int = Field(default_factory=lambda: int(os.getenv("WHISPER_BATCH_SIZE", "8")))

    whisper_vad_filter: bool = Field(default_factory=lambda: _env_bool("WHISPER_VAD_FILTER", "true"))
    whisper_min_silence_duration_ms: int = Field(
        default_factory=lambda: int(os.getenv("WHISPER_MIN_SILENCE_DURATION_MS", "500"))
    )

    whisper_condition_on_previous_text: bool = Field(
        default_factory=lambda: _env_bool("WHISPER_CONDITION_ON_PREVIOUS_TEXT", "false")
    )
    whisper_initial_prompt: str | None = Field(
        default_factory=lambda: os.getenv("WHISPER_INITIAL_PROMPT") or None
    )

    input_audio_dir: Path = Field(default_factory=lambda: Path(os.getenv("INPUT_AUDIO_DIR", "./data/audio_raw")))
    processed_audio_dir: Path = Field(
        default_factory=lambda: Path(os.getenv("PROCESSED_AUDIO_DIR", "./data/audio_processed"))
    )
    transcripts_output_dir: Path = Field(
        default_factory=lambda: Path(os.getenv("TRANSCRIPTS_OUTPUT_DIR", "./data/transcripts"))
    )

    export_txt: bool = Field(default_factory=lambda: _env_bool("EXPORT_TXT", "true"))
    export_md: bool = Field(default_factory=lambda: _env_bool("EXPORT_MD", "true"))
    export_json: bool = Field(default_factory=lambda: _env_bool("EXPORT_JSON", "true"))
    export_jsonl: bool = Field(default_factory=lambda: _env_bool("EXPORT_JSONL", "true"))
    export_srt: bool = Field(default_factory=lambda: _env_bool("EXPORT_SRT", "true"))

    preprocess_audio: bool = Field(default_factory=lambda: _env_bool("PREPROCESS_AUDIO", "true"))
    audio_sample_rate: int = Field(default_factory=lambda: int(os.getenv("AUDIO_SAMPLE_RATE", "16000")))
    audio_channels: int = Field(default_factory=lambda: int(os.getenv("AUDIO_CHANNELS", "1")))
    keep_processed_audio: bool = Field(default_factory=lambda: _env_bool("KEEP_PROCESSED_AUDIO", "true"))

    skip_existing: bool = Field(default_factory=lambda: _env_bool("SKIP_EXISTING", "true"))
    log_level: str = Field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))


def get_settings() -> Settings:
    return Settings()
```

Implementation notes:

- Use a helper like `_env_bool()` instead of repeating boolean parsing logic.
- Keep settings simple and explicit.
- `WHISPER_INITIAL_PROMPT` should be optional.
- Use `Path` objects for paths.
- Default to `cuda` and `float16`.
- Use `WHISPER_BATCH_SIZE` to control batched GPU inference.

---

## 2. `cuda_check.py`

Create helper functions for CUDA diagnostics.

```python
import shutil
import subprocess
from dataclasses import dataclass


@dataclass
class CudaCheckResult:
    nvidia_smi_found: bool
    nvidia_smi_output: str | None
    faster_whisper_cuda_ok: bool
    error: str | None = None


def run_nvidia_smi() -> tuple[bool, str | None]:
    if shutil.which("nvidia-smi") is None:
        return False, None
    completed = subprocess.run(
        ["nvidia-smi"],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.returncode == 0, completed.stdout or completed.stderr


def check_faster_whisper_cuda() -> tuple[bool, str | None]:
    try:
        from faster_whisper import WhisperModel

        WhisperModel("tiny", device="cuda", compute_type="float16")
        return True, None
    except Exception as exc:
        return False, str(exc)
```

Expose this through:

```bash
kado-transcribe check-cuda
```

---

## 3. `hashing.py`

Add file hashing to make skip/versioning safer.

Create:

```python
from pathlib import Path
import hashlib


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()
```

Use this hash in `metadata.json` so the project knows whether a transcript corresponds to the same source file and same transcription config.

---

## 4. `audio.py`

Responsibilities:

- Discover audio/video files.
- Validate extensions.
- Optionally convert files to 16 kHz mono WAV using FFmpeg.
- Return paths to files ready for transcription.

Supported extensions:

```python
SUPPORTED_MEDIA_EXTENSIONS = {
    ".mp3", ".m4a", ".wav", ".webm", ".mp4", ".mov", ".aac", ".ogg", ".flac"
}
```

Create function:

```python
def discover_media_files(input_dir: Path) -> list[Path]:
    ...
```

Create function:

```python
def preprocess_to_wav(
    input_path: Path,
    output_dir: Path,
    sample_rate: int = 16000,
    channels: int = 1,
    skip_existing: bool = True,
) -> Path:
    ...
```

Use FFmpeg command:

```bash
ffmpeg -y -i input_file -vn -ac 1 -ar 16000 -c:a pcm_s16le output_file.wav
```

Implementation notes:

- Use `subprocess.run` with `check=True`.
- Create output directory if missing.
- Preserve base filename.
- If output already exists and `skip_existing=true`, reuse it.
- Log or print useful errors when FFmpeg fails.
- Before running FFmpeg, verify it is available with `shutil.which("ffmpeg")`.
- If FFmpeg is missing, show a clear message:

```bash
sudo apt update
sudo apt install -y ffmpeg
```

---

## 5. `transcriber.py`

Responsibilities:

- Load the faster-whisper model once.
- Use CUDA by default.
- Use batched inference when `WHISPER_BATCH_SIZE > 1`.
- Transcribe one audio file.
- Return a structured result.

Create data models:

```python
from pydantic import BaseModel


class TranscriptSegment(BaseModel):
    id: int
    start: float
    end: float
    text: str


class TranscriptResult(BaseModel):
    source_file: str
    source_sha256: str | None = None
    language: str | None = None
    language_probability: float | None = None
    duration: float | None = None
    model_name: str
    device: str
    compute_type: str
    batch_size: int
    segments: list[TranscriptSegment]
    full_text: str
```

Create class:

```python
class FasterWhisperTranscriber:
    def __init__(self, settings: Settings):
        ...

    def transcribe_file(self, audio_path: Path, source_sha256: str | None = None) -> TranscriptResult:
        ...
```

Core model loading:

```python
from faster_whisper import WhisperModel, BatchedInferencePipeline

self.model = WhisperModel(
    settings.whisper_model_name,
    device=settings.whisper_device,
    compute_type=settings.whisper_compute_type,
)

self.pipeline = None
if settings.whisper_batch_size > 1:
    self.pipeline = BatchedInferencePipeline(model=self.model)
```

Core transcription:

```python
transcribe_target = self.pipeline if self.pipeline is not None else self.model

kwargs = dict(
    language=settings.whisper_language,
    task=settings.whisper_task,
    beam_size=settings.whisper_beam_size,
    vad_filter=settings.whisper_vad_filter,
    vad_parameters={
        "min_silence_duration_ms": settings.whisper_min_silence_duration_ms,
    },
    condition_on_previous_text=settings.whisper_condition_on_previous_text,
    initial_prompt=settings.whisper_initial_prompt,
)

if self.pipeline is not None:
    kwargs["batch_size"] = settings.whisper_batch_size

segments, info = transcribe_target.transcribe(str(audio_path), **kwargs)
```

Important: `segments` is a generator. Convert it to a list during processing.

Build `TranscriptSegment` objects with:

- `id`
- `start`
- `end`
- `text.strip()`

Build `full_text` by joining segment texts with newlines rather than one long paragraph.

Recommended:

```python
full_text = "\n".join(segment.text for segment in transcript_segments if segment.text)
```

### CUDA Error Handling

If model loading fails with CUDA errors, show suggestions:

- Run `nvidia-smi`.
- Run `source scripts/setup_cuda_env.sh`.
- Confirm `nvidia-cublas-cu12` and `nvidia-cudnn-cu12==9.*` are installed.
- Try `WHISPER_COMPUTE_TYPE=int8_float16`.
- Try reducing `WHISPER_BATCH_SIZE` from `8` to `4`, `2`, or `1`.
- Use CPU fallback only if CUDA cannot be fixed:

```env
WHISPER_DEVICE=cpu
WHISPER_COMPUTE_TYPE=int8
WHISPER_BATCH_SIZE=1
```

---

## 6. `exporters.py`

Responsibilities:

- Export `.txt`
- Export `.md`
- Export `.json`
- Export `.jsonl`
- Export `.srt`
- Export `metadata.json`
- Export everything through a single `export_all()` helper

Functions:

```python
def export_txt(result: TranscriptResult, output_dir: Path) -> Path:
    ...


def export_md(result: TranscriptResult, output_dir: Path) -> Path:
    ...


def export_json(result: TranscriptResult, output_dir: Path) -> Path:
    ...


def export_jsonl(result: TranscriptResult, output_dir: Path) -> Path:
    ...


def export_srt(result: TranscriptResult, output_dir: Path) -> Path:
    ...


def export_metadata(result: TranscriptResult, output_dir: Path, settings: Settings) -> Path:
    ...


def export_all(result: TranscriptResult, output_dir: Path, settings: Settings) -> list[Path]:
    ...
```

### TXT Format

Only transcript text:

```text
Texto transcrito completo...
```

### MD Format

Human-readable transcript:

```md
# Transcript: entrevista_001

- Source file: entrevista_001.wav
- Language: es
- Model: large-v3
- Device: cuda
- Compute type: float16
- Batch size: 8
- Duration: 01:04:23

## Full Transcript

[00:00:01 - 00:00:06]
Texto del segmento...

[00:00:07 - 00:00:12]
Texto del segmento...
```

Use this block-style timestamp format because it is easier to parse later for quote extraction and LLM analysis.

### JSON Format

Machine-readable:

```json
{
  "source_file": "entrevista_001.wav",
  "source_sha256": "...",
  "language": "es",
  "language_probability": 0.99,
  "duration": 3863.2,
  "model_name": "large-v3",
  "device": "cuda",
  "compute_type": "float16",
  "batch_size": 8,
  "segments": [
    {"id": 1, "start": 0.0, "end": 4.2, "text": "..."}
  ],
  "full_text": "..."
}
```

### JSONL Format

Useful later for RAG, LightRAG, AnythingLLM, or custom chunking:

```jsonl
{"interview_id":"entrevista_001","source_file":"entrevista_001.wav","start":0.0,"end":12.4,"text":"..."}
{"interview_id":"entrevista_001","source_file":"entrevista_001.wav","start":12.4,"end":25.1,"text":"..."}
```

### SRT Format

Implement timestamp formatting:

```python
def format_srt_timestamp(seconds: float) -> str:
    milliseconds = int((seconds - int(seconds)) * 1000)
    total_seconds = int(seconds)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    return f"{hours:02}:{minutes:02}:{secs:02},{milliseconds:03}"
```

SRT segment format:

```text
1
00:00:00,000 --> 00:00:04,200
Texto del segmento
```

### Metadata Format

`metadata.json` should include enough information to decide whether an existing transcript can be safely reused.

Required metadata:

```json
{
  "source_file": "entrevista_001.wav",
  "source_sha256": "...",
  "created_at": "2026-04-24T12:00:00Z",
  "model_name": "large-v3",
  "device": "cuda",
  "compute_type": "float16",
  "batch_size": 8,
  "language": "es",
  "task": "transcribe",
  "beam_size": 5,
  "vad_filter": true,
  "min_silence_duration_ms": 500,
  "condition_on_previous_text": false,
  "initial_prompt_used": true,
  "duration": 3863.2,
  "language_probability": 0.99,
  "segment_count": 412
}
```

---

## 7. Skip/Versioning Logic

Do not skip files only because an output folder exists.

Instead, skip only when:

1. `SKIP_EXISTING=true`
2. `metadata.json` exists
3. `transcript.txt` or another required transcript output exists
4. The current source SHA256 matches `metadata.json`
5. The current transcription config matches `metadata.json`

Config fields that should be compared:

- `model_name`
- `device`
- `compute_type`
- `batch_size`
- `language`
- `task`
- `beam_size`
- `vad_filter`
- `min_silence_duration_ms`
- `condition_on_previous_text`
- `initial_prompt_used`

If the source file changed or the model settings changed, re-transcribe.

Create helper:

```python
def should_skip_existing(
    output_dir: Path,
    source_sha256: str,
    settings: Settings,
) -> bool:
    ...
```

---

## 8. `cli.py`

Use `typer` and `rich`.

Commands:

```bash
kado-transcribe inspect-config
kado-transcribe check-cuda
kado-transcribe transcribe-file path/to/audio.m4a
kado-transcribe transcribe-batch --input-dir ./data/audio_raw --output-dir ./data/transcripts
kado-transcribe clean-processed
```

Also support module invocation:

```bash
python -m kado_transcriber.cli transcribe-batch
```

CLI behavior:

- Load settings from `.env`.
- Create required directories.
- Discover media files.
- Compute source file hash.
- Check whether output can be skipped safely.
- Preprocess each file if enabled.
- Load model once for batch transcription.
- Use CUDA by default.
- Use batched inference when `WHISPER_BATCH_SIZE > 1`.
- Transcribe each file.
- Export all configured formats.
- Optionally delete processed WAV files if `KEEP_PROCESSED_AUDIO=false`.
- Print a summary table:
  - file name
  - status
  - output folder
  - duration
  - number of segments
  - model
  - device
  - compute type
  - batch size
  - skipped or transcribed

Pseudo-flow:

```python
@app.command("transcribe-batch")
def transcribe_batch(input_dir: Path | None = None, output_dir: Path | None = None):
    settings = get_settings()
    input_dir = input_dir or settings.input_audio_dir
    output_dir = output_dir or settings.transcripts_output_dir

    files = discover_media_files(input_dir)
    if not files:
        raise typer.BadParameter(f"No supported media files found in {input_dir}")

    transcriber = FasterWhisperTranscriber(settings)

    for file in files:
        source_sha256 = file_sha256(file)
        file_output_dir = output_dir / file.stem

        if should_skip_existing(file_output_dir, source_sha256, settings):
            report_skipped(file)
            continue

        ready_file = file
        if settings.preprocess_audio:
            ready_file = preprocess_to_wav(
                file,
                settings.processed_audio_dir,
                settings.audio_sample_rate,
                settings.audio_channels,
                settings.skip_existing,
            )

        result = transcriber.transcribe_file(ready_file, source_sha256=source_sha256)
        file_output_dir.mkdir(parents=True, exist_ok=True)
        export_all(result, file_output_dir, settings)

        if settings.preprocess_audio and not settings.keep_processed_audio:
            maybe_delete_processed_file(ready_file, original_file=file)
```

---

## 9. Preview / Quality-Control Workflow

Before processing all long interviews, test a short sample.

Option A: implement a preview command:

```bash
kado-transcribe transcribe-file ./data/audio_raw/interview_001.mp3 \
  --output-dir ./data/transcripts_preview
```

Option B: document a manual FFmpeg cut:

```bash
mkdir -p data/audio_preview
ffmpeg -y -i data/audio_raw/interview_001.mp3 -t 600 -ac 1 -ar 16000 data/audio_preview/interview_001_first_10min.wav
```

Then run:

```bash
source scripts/setup_cuda_env.sh

kado-transcribe transcribe-file data/audio_preview/interview_001_first_10min.wav \
  --output-dir ./data/transcripts_preview
```

Recommended comparison before full batch:

1. Run `large-v3` with CUDA `float16`, batch size `8`.
2. If VRAM issues appear, run `large-v3` with CUDA `int8_float16`, batch size `4`.
3. If speed is still more important than maximum quality, test `turbo` with CUDA `float16`.
4. Compare the first 10 minutes manually.
5. Decide whether the quality improvement justifies the runtime.

This is important because KADO’s later insight extraction depends more on transcript quality than raw speed.

---

## Recommended Model Settings

### CUDA Quality Default

Use for most KADO Spanish interviews when an NVIDIA GPU is available:

```env
WHISPER_MODEL_NAME=large-v3
WHISPER_DEVICE=cuda
WHISPER_COMPUTE_TYPE=float16
WHISPER_LANGUAGE=es
WHISPER_BEAM_SIZE=5
WHISPER_BATCH_SIZE=8
WHISPER_VAD_FILTER=true
WHISPER_CONDITION_ON_PREVIOUS_TEXT=false
```

### CUDA Lower-VRAM Mode

Use if the GPU runs out of memory:

```env
WHISPER_MODEL_NAME=large-v3
WHISPER_DEVICE=cuda
WHISPER_COMPUTE_TYPE=int8_float16
WHISPER_LANGUAGE=es
WHISPER_BEAM_SIZE=5
WHISPER_BATCH_SIZE=4
WHISPER_VAD_FILTER=true
WHISPER_CONDITION_ON_PREVIOUS_TEXT=false
```

If still running out of memory:

```env
WHISPER_BATCH_SIZE=2
```

or:

```env
WHISPER_MODEL_NAME=medium
```

### CUDA Faster Mode

Use if speed matters more than maximum quality:

```env
WHISPER_MODEL_NAME=turbo
WHISPER_DEVICE=cuda
WHISPER_COMPUTE_TYPE=float16
WHISPER_LANGUAGE=es
WHISPER_BEAM_SIZE=5
WHISPER_BATCH_SIZE=8
WHISPER_VAD_FILTER=true
WHISPER_CONDITION_ON_PREVIOUS_TEXT=false
```

### CPU Fallback

Use only if CUDA cannot be fixed:

```env
WHISPER_MODEL_NAME=medium
WHISPER_DEVICE=cpu
WHISPER_COMPUTE_TYPE=int8
WHISPER_LANGUAGE=es
WHISPER_BEAM_SIZE=5
WHISPER_BATCH_SIZE=1
WHISPER_VAD_FILTER=true
WHISPER_CONDITION_ON_PREVIOUS_TEXT=false
```

### Domain-Specific Spanish Prompt

Recommended for KADO interviews:

```env
WHISPER_INITIAL_PROMPT=Entrevista en español sobre nutrición, consulta nutricional, expediente clínico, antropometría, IMC, dieta, pacientes, seguimiento y software para nutriólogos.
```

This can help the model handle KADO-relevant vocabulary such as:

- Nutriólogo
- Nutricionista
- Antropometría
- Expediente clínico
- IMC
- Consulta
- Seguimiento
- Dieta
- Pacientes
- Recordatorio
- Plan alimenticio

---

## Quality Controls

Add these checks:

1. If no files are found, show a clear error.
2. If FFmpeg is missing, show install instructions.
3. If `WHISPER_DEVICE=cuda`, run or recommend `kado-transcribe check-cuda` before long jobs.
4. If model loading fails, show likely causes:
   - Invalid model name
   - NVIDIA driver not visible
   - CUDA libraries not available in `LD_LIBRARY_PATH`
   - cuDNN/cuBLAS mismatch
   - No internet for first model download
   - GPU does not support the selected compute type
5. If CUDA runs out of memory, suggest:
   - Lower `WHISPER_BATCH_SIZE`
   - Use `WHISPER_COMPUTE_TYPE=int8_float16`
   - Use `WHISPER_MODEL_NAME=medium`
6. If output already exists and `SKIP_EXISTING=true`, skip only if source hash and config match.
7. Save `metadata.json` so each transcript records model, language, compute type, date, source file hash, and transcription settings.
8. Print a clear summary table after batch transcription.
9. Do not import or install pyannote, NeMo, WhisperX, or diarization dependencies.
10. Keep transcript generation separate from insight extraction.

---

## Acceptance Criteria

Codex should consider the task complete when:

- The project installs successfully with `pip install -e .[dev,cuda]` or `uv pip install -e .[dev,cuda]`.
- `source scripts/setup_cuda_env.sh` works on Linux/WSL.
- `kado-transcribe inspect-config` prints loaded CUDA-first settings.
- `kado-transcribe check-cuda` successfully loads a tiny faster-whisper model on CUDA.
- `kado-transcribe transcribe-file ./data/audio_raw/sample.mp3` works with `WHISPER_DEVICE=cuda`.
- `kado-transcribe transcribe-batch` processes multiple files.
- The output folder contains `.txt`, `.md`, `.json`, `.jsonl`, `.srt`, and `metadata.json`.
- `metadata.json` records `device`, `compute_type`, and `batch_size`.
- The transcript language is Spanish when `WHISPER_LANGUAGE=es`.
- The code does not import pyannote, NeMo, WhisperX, or diarization dependencies.
- Existing outputs are skipped only when source hash and transcription config match.
- Tests pass with `pytest`.
- README includes CUDA installation, CUDA smoke test, fallback settings, model recommendations, and troubleshooting.

---

## Tests to Add

### `test_exporters.py`

Test:

- SRT timestamp formatting.
- JSON export contains `segments` and `full_text`.
- JSON export contains `device`, `compute_type`, and `batch_size`.
- JSONL export writes one valid JSON object per segment.
- TXT export writes clean text.
- MD export includes source file and timestamps.
- `export_all()` respects export settings.

### `test_config.py`

Test:

- Default settings load as CUDA-first.
- Boolean env vars parse correctly.
- Path env vars become `Path` objects.
- `WHISPER_INITIAL_PROMPT` loads as `None` when empty.
- `WHISPER_CONDITION_ON_PREVIOUS_TEXT` defaults to `false`.
- `WHISPER_BATCH_SIZE` defaults to `8`.
- CPU fallback settings can be loaded from env vars.

### `test_hashing.py`

Test:

- `file_sha256()` returns the same hash for the same file.
- `file_sha256()` returns different hashes for different file contents.

Do not test real model inference in unit tests because it is slow and may download model files. Keep model inference as a manual integration test or smoke test.

---

## Manual Test Procedure

After implementation:

```bash
mkdir -p data/audio_raw
cp /path/to/short-spanish-sample.mp3 data/audio_raw/
cp .env.example .env
source scripts/setup_cuda_env.sh
kado-transcribe inspect-config
kado-transcribe check-cuda
kado-transcribe transcribe-batch
```

Verify:

```bash
ls -R data/transcripts
cat data/transcripts/<sample-name>/transcript.md
cat data/transcripts/<sample-name>/metadata.json
```

Expected result:

- CUDA check passes.
- Spanish transcript text appears.
- Segment timestamps appear.
- JSON is valid.
- JSONL has one segment per line.
- SRT opens in a subtitle editor or media player.
- Metadata records model settings, CUDA device, compute type, batch size, and source hash.

---

## Optional Future Extension: Insight Extraction

Do not implement this in the first milestone unless explicitly requested. Leave the project ready for this next stage.

Future folder:

```text
src/kado_transcriber/insights.py
prompts/
  kado_customer_discovery_es.md
```

Future extracted insight schema:

```json
{
  "interview_id": "entrevista_001",
  "summary": "...",
  "pain_points": ["..."],
  "current_workflow": ["..."],
  "tools_used_today": ["..."],
  "feature_requests": ["..."],
  "objections": ["..."],
  "willingness_to_pay_signals": ["..."],
  "direct_quotes": [
    {
      "quote": "...",
      "timestamp": "00:12:34",
      "theme": "workflow friction"
    }
  ]
}
```

Recommended later approach:

1. Do not extract insights from raw audio.
2. Extract insights only from reviewed transcript text.
3. Use transcript segments and timestamps to preserve quote traceability.
4. Store insight outputs separately from transcript outputs.

---

## Codex Execution Instructions

1. Inspect the current repository.
2. If a transcription project already exists, adapt it instead of replacing everything blindly.
3. Remove or ignore diarization/speaker-recognition dependencies for this milestone.
4. Implement the modules described above.
5. Keep the code simple, local-first, CLI-first, and CUDA-first.
6. Use Spanish as the default transcription language.
7. Make CUDA `float16` the default execution mode.
8. Add `int8_float16` as the documented lower-VRAM CUDA fallback.
9. Add CPU `int8` as the final fallback.
10. Add `kado-transcribe check-cuda`.
11. Add source hashing and config-aware skip/versioning.
12. Add TXT, MD, JSON, JSONL, SRT, and metadata exports.
13. Add a clear README with CUDA installation, `LD_LIBRARY_PATH` setup, configuration, usage examples, model recommendations, and troubleshooting.
14. Add tests for config, exporters, and hashing.
15. Run formatting/tests and report what passed or failed.
16. Do not implement diarization.
17. Do not implement LLM insight extraction in v1 unless explicitly requested.

---

## Suggested README Content

The README should include:

- Project goal.
- Why no diarization in v1.
- CUDA requirements.
- Installation with `.[dev,cuda]`.
- `LD_LIBRARY_PATH` setup.
- CUDA smoke test.
- `.env` setup.
- Batch transcription command.
- Single-file transcription command.
- File output explanation.
- Model recommendations.
- Quality-review workflow.
- Troubleshooting.

Troubleshooting examples:

### `nvidia-smi` not found

Confirm the NVIDIA driver is installed and visible from the current environment.

For WSL, confirm GPU support is working inside WSL before debugging Python.

### CUDA libraries not found

Run:

```bash
source scripts/setup_cuda_env.sh
```

If dependencies are missing, install:

```bash
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12==9.*
```

### CUDA out of memory

Try:

```env
WHISPER_BATCH_SIZE=4
WHISPER_COMPUTE_TYPE=int8_float16
```

If still failing:

```env
WHISPER_BATCH_SIZE=2
```

or:

```env
WHISPER_MODEL_NAME=medium
```

### Float16 not supported or inefficient

Try:

```env
WHISPER_COMPUTE_TYPE=int8_float16
```

If CUDA still fails, use CPU fallback:

```env
WHISPER_DEVICE=cpu
WHISPER_COMPUTE_TYPE=int8
WHISPER_BATCH_SIZE=1
```

### First run is slow

The first run may download the selected faster-whisper model. Later runs should reuse the local cache.

### Transcript quality is poor

Try:

```env
WHISPER_MODEL_NAME=large-v3
WHISPER_BEAM_SIZE=5
WHISPER_LANGUAGE=es
WHISPER_CONDITION_ON_PREVIOUS_TEXT=false
```

Also check the source audio quality and consider preprocessing noisy recordings separately.

### Existing transcript is not being regenerated

Check:

```env
SKIP_EXISTING=false
```

or delete the specific transcript output folder.

The improved implementation should also regenerate automatically if the source hash or model configuration changes.

---

## Final Deliverable

A working local CUDA-first CLI tool that transforms KADO Spanish interview recordings into structured transcripts without speaker recognition.

The first milestone is complete when this command works:

```bash
source scripts/setup_cuda_env.sh
kado-transcribe transcribe-batch --input-dir ./data/audio_raw --output-dir ./data/transcripts
```

And produces a usable transcript package for each interview recording:

```text
transcript.txt
transcript.md
transcript.json
transcript.jsonl
transcript.srt
metadata.json
```

The implementation should be stable enough for the KADO team to transcribe several 1-hour-plus Spanish interviews on an NVIDIA CUDA GPU and later reuse the outputs for customer-discovery insight extraction.
