from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
import re
from typing import Any


TIMESTAMP_RE = re.compile(
    r"^\[(\d{2}):(\d{2}):(\d{2}) - (\d{2}):(\d{2}):(\d{2})\]$"
)


@dataclass(frozen=True)
class TranscriptLine:
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class SpeakerTurn:
    start: float
    end: float
    speaker: str


@dataclass(frozen=True)
class DiarizedLine:
    start: float
    end: float
    speaker: str
    text: str


def parse_transcript(path: Path) -> list[TranscriptLine]:
    lines = path.read_text(encoding="utf-8").splitlines()
    segments: list[TranscriptLine] = []
    index = 0

    while index < len(lines):
        match = TIMESTAMP_RE.match(lines[index].strip())
        if match is None:
            index += 1
            continue

        start = _timestamp_seconds(match.groups()[:3])
        end = _timestamp_seconds(match.groups()[3:])
        index += 1
        text: list[str] = []
        while index < len(lines) and TIMESTAMP_RE.match(lines[index].strip()) is None:
            if lines[index].strip():
                text.append(lines[index].strip())
            index += 1
        segments.append(TranscriptLine(start=start, end=end, text=" ".join(text)))

    if not segments:
        raise ValueError(f"No timestamped transcript segments found in {path}")
    return segments


def assign_speakers(
    segments: list[TranscriptLine], turns: list[SpeakerTurn]
) -> list[DiarizedLine]:
    if not turns:
        raise ValueError("Diarization returned no speaker turns")

    output: list[DiarizedLine] = []
    for segment in segments:
        overlap_by_speaker: dict[str, float] = defaultdict(float)
        for turn in turns:
            overlap = max(0.0, min(segment.end, turn.end) - max(segment.start, turn.start))
            overlap_by_speaker[turn.speaker] += overlap

        if overlap_by_speaker and max(overlap_by_speaker.values()) > 0:
            speaker = max(overlap_by_speaker, key=overlap_by_speaker.get)
        else:
            midpoint = (segment.start + segment.end) / 2
            speaker = min(
                turns,
                key=lambda turn: min(
                    abs(midpoint - turn.start), abs(midpoint - turn.end)
                ),
            ).speaker

        output.append(
            DiarizedLine(
                start=segment.start,
                end=segment.end,
                speaker=speaker,
                text=segment.text,
            )
        )
    return output


def run_diarization(
    media_path: Path,
    model_name: str,
    token: str,
    device: str,
    num_speakers: int | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
) -> list[SpeakerTurn]:
    import numpy as np

    # pyannote.audio 3.3 uses this NumPy alias, removed in NumPy 2.
    if "NAN" not in np.__dict__:
        np.NAN = np.nan

    try:
        import torch
        from pyannote.audio import Pipeline
    except ImportError as exc:
        raise RuntimeError(
            'Diarization dependencies are missing. Install with `pip install -e ".[diarize]"`.'
        ) from exc

    waveform, sample_rate = decode_waveform(media_path)
    pipeline = Pipeline.from_pretrained(model_name, use_auth_token=token)
    if pipeline is None:
        raise RuntimeError(
            f"Could not load {model_name}. Confirm HF_TOKEN and gated model access."
        )
    pipeline.to(torch.device(device))

    options: dict[str, int] = {}
    if num_speakers is not None:
        options["num_speakers"] = num_speakers
    else:
        if min_speakers is not None:
            options["min_speakers"] = min_speakers
        if max_speakers is not None:
            options["max_speakers"] = max_speakers

    annotation = pipeline(
        {"waveform": waveform, "sample_rate": sample_rate},
        **options,
    )
    return [
        SpeakerTurn(start=float(turn.start), end=float(turn.end), speaker=speaker)
        for turn, _, speaker in annotation.itertracks(yield_label=True)
    ]


def decode_waveform(media_path: Path, sample_rate: int = 16_000) -> tuple[Any, int]:
    import av
    import numpy as np
    import torch

    container = av.open(str(media_path))
    chunks: list[Any] = []
    try:
        stream = next(stream for stream in container.streams if stream.type == "audio")
        resampler = av.AudioResampler(format="s16", layout="mono", rate=sample_rate)
        for packet in container.demux(stream):
            for frame in packet.decode():
                for resampled in resampler.resample(frame):
                    chunks.append(resampled.to_ndarray().reshape(1, -1))
        for resampled in resampler.resample(None):
            chunks.append(resampled.to_ndarray().reshape(1, -1))
    finally:
        container.close()

    if not chunks:
        raise ValueError(f"No audio stream could be decoded from {media_path}")
    audio = np.concatenate(chunks, axis=1).astype(np.float32) / 32768.0
    return torch.from_numpy(audio), sample_rate


def export_diarized(
    source_path: Path,
    transcript_path: Path,
    output_dir: Path,
    segments: list[DiarizedLine],
    turns: list[SpeakerTurn],
    model_name: str,
    device: str,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    markdown_path = output_dir / f"{source_path.stem}_diarized.md"
    metadata_path = output_dir / f"{source_path.stem}_diarized_metadata.json"
    speakers = sorted({turn.speaker for turn in turns})

    markdown = [
        f"# Diarized Transcript: {source_path.stem}",
        "",
        f"- Source file: {source_path}",
        f"- Base transcript: {transcript_path}",
        f"- Diarization model: {model_name}",
        f"- Device: {device}",
        f"- Speakers: {len(speakers)} ({', '.join(speakers)})",
        "",
        "## Transcript",
        "",
    ]
    for segment in segments:
        markdown.extend(
            [
                f"[{_format_timestamp(segment.start)} - {_format_timestamp(segment.end)}] {segment.speaker}",
                segment.text,
                "",
            ]
        )
    markdown_path.write_text("\n".join(markdown).rstrip() + "\n", encoding="utf-8")

    metadata = {
        "source_file": str(source_path),
        "base_transcript": str(transcript_path),
        "created_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "model_name": model_name,
        "device": device,
        "speaker_count": len(speakers),
        "speakers": speakers,
        "segment_count": len(segments),
        "speaker_turn_count": len(turns),
        "segments": [asdict(segment) for segment in segments],
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return markdown_path, metadata_path


def _timestamp_seconds(parts: tuple[str, ...]) -> float:
    hours, minutes, seconds = (int(part) for part in parts)
    return float(hours * 3600 + minutes * 60 + seconds)


def _format_timestamp(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    return f"{total_seconds // 3600:02}:{(total_seconds % 3600) // 60:02}:{total_seconds % 60:02}"
