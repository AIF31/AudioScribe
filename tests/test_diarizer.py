import json

from audio_transcriber.diarizer import (
    SpeakerTurn,
    TranscriptLine,
    assign_speakers,
    export_diarized,
    parse_transcript,
)


def test_parse_transcript(tmp_path):
    path = tmp_path / "sample_transcript.md"
    path.write_text(
        "# Transcript: sample\n\n## Transcript\n\n"
        "[00:00:01 - 00:00:05]\nHola.\n\n"
        "[00:01:00 - 00:01:03]\nGracias.\n",
        encoding="utf-8",
    )

    segments = parse_transcript(path)

    assert segments == [
        TranscriptLine(start=1.0, end=5.0, text="Hola."),
        TranscriptLine(start=60.0, end=63.0, text="Gracias."),
    ]


def test_assign_speakers_uses_greatest_overlap():
    segments = [TranscriptLine(start=1.0, end=5.0, text="Hola.")]
    turns = [
        SpeakerTurn(start=0.0, end=2.0, speaker="SPEAKER_00"),
        SpeakerTurn(start=2.0, end=6.0, speaker="SPEAKER_01"),
    ]

    output = assign_speakers(segments, turns)

    assert output[0].speaker == "SPEAKER_01"


def test_export_diarized(tmp_path):
    source = tmp_path / "sample.mp4"
    transcript = tmp_path / "sample_transcript.md"
    segments = assign_speakers(
        [TranscriptLine(start=0.0, end=2.0, text="Hola.")],
        [SpeakerTurn(start=0.0, end=2.0, speaker="SPEAKER_00")],
    )

    markdown, metadata = export_diarized(
        source,
        transcript,
        tmp_path,
        segments,
        [SpeakerTurn(start=0.0, end=2.0, speaker="SPEAKER_00")],
        "pyannote/speaker-diarization-3.1",
        "cuda",
    )

    assert "[00:00:00 - 00:00:02] SPEAKER_00" in markdown.read_text(encoding="utf-8")
    data = json.loads(metadata.read_text(encoding="utf-8"))
    assert data["speaker_count"] == 1
    assert data["segments"][0]["text"] == "Hola."
