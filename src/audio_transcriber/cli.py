from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from audio_transcriber.audio import discover_media_files
from audio_transcriber.config import Settings, get_settings
from audio_transcriber.cuda_check import check_cuda
from audio_transcriber.diarizer import (
    assign_speakers,
    export_diarized,
    parse_transcript,
    run_diarization,
)
from audio_transcriber.exporters import export_all, format_timestamp
from audio_transcriber.hashing import file_sha256
from audio_transcriber.skip import should_skip_existing
from audio_transcriber.transcriber import TranscriberProtocol, create_transcriber

app = typer.Typer(help="Audio transcription with local faster-whisper or OpenAI models.")
console = Console()


@app.command("inspect-config")
def inspect_config() -> None:
    settings = get_settings()
    table = Table(title="Audio Transcriber Settings")
    table.add_column("Setting")
    table.add_column("Value")
    for key, value in settings.model_dump().items():
        if key in {"hf_token", "openai_api_key"} and value:
            value = "***"
        table.add_row(key, str(value))
    console.print(table)


@app.command("check-cuda")
def check_cuda_command() -> None:
    settings = get_settings()
    console.print(f"Configured device: {settings.whisper_device}")
    console.print(f"Configured compute type: {settings.whisper_compute_type}")
    result = check_cuda()

    if result.nvidia_smi_found:
        console.print("[green]nvidia-smi is available.[/green]")
    else:
        console.print("[red]nvidia-smi is not available or failed.[/red]")

    if result.faster_whisper_cuda_ok:
        console.print("[green]faster-whisper loaded a tiny CUDA model successfully.[/green]")
    else:
        console.print("[red]faster-whisper CUDA check failed.[/red]")
        if result.error:
            console.print(result.error)
        raise typer.Exit(code=1)


@app.command("transcribe-file")
def transcribe_file(
    audio_path: Path = typer.Argument(..., exists=True, file_okay=True, dir_okay=False),
    output_dir: Path | None = typer.Option(None, "--output-dir", "-o"),
) -> None:
    settings = get_settings()
    output_root = output_dir or settings.transcripts_output_dir
    output_root.mkdir(parents=True, exist_ok=True)
    _transcribe_files([audio_path], output_root, settings)


@app.command("transcribe-batch")
def transcribe_batch(
    input_dir: Path | None = typer.Option(None, "--input-dir", "-i"),
    output_dir: Path | None = typer.Option(None, "--output-dir", "-o"),
) -> None:
    settings = get_settings()
    source_dir = input_dir or settings.input_audio_dir
    output_root = output_dir or settings.transcripts_output_dir

    files = discover_media_files(source_dir)
    if not files:
        raise typer.BadParameter(f"No supported media files found in {source_dir}")

    output_root.mkdir(parents=True, exist_ok=True)
    _transcribe_files(files, output_root, settings)


@app.command("diarize-file")
def diarize_file(
    media_path: Path = typer.Argument(..., exists=True, file_okay=True, dir_okay=False),
    output_dir: Path | None = typer.Option(None, "--output-dir", "-o"),
    transcript_path: Path | None = typer.Option(None, "--transcript"),
    num_speakers: int | None = typer.Option(None, "--num-speakers", min=1),
    min_speakers: int | None = typer.Option(None, "--min-speakers", min=1),
    max_speakers: int | None = typer.Option(None, "--max-speakers", min=1),
) -> None:
    """Add local pyannote speaker labels to an existing transcript."""
    settings = get_settings()
    if not settings.hf_token:
        raise typer.BadParameter("HF_TOKEN is required for pyannote diarization")
    if num_speakers is not None and (min_speakers is not None or max_speakers is not None):
        raise typer.BadParameter(
            "--num-speakers cannot be combined with --min-speakers or --max-speakers"
        )
    if min_speakers is not None and max_speakers is not None and min_speakers > max_speakers:
        raise typer.BadParameter("--min-speakers cannot exceed --max-speakers")

    output_root = output_dir or settings.transcripts_output_dir
    file_output_dir = output_root / media_path.stem
    source_transcript = transcript_path or file_output_dir / f"{media_path.stem}_transcript.md"
    if not source_transcript.exists():
        raise typer.BadParameter(f"Transcript not found: {source_transcript}")

    console.print(f"Decoding and diarizing [bold]{media_path.name}[/bold]...")
    turns = run_diarization(
        media_path,
        model_name=settings.diarization_model,
        token=settings.hf_token,
        device=settings.diarization_device,
        num_speakers=num_speakers,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
    )
    segments = assign_speakers(parse_transcript(source_transcript), turns)
    markdown_path, metadata_path = export_diarized(
        media_path,
        source_transcript,
        file_output_dir,
        segments,
        turns,
        settings.diarization_model,
        settings.diarization_device,
    )
    console.print(
        f"Wrote {markdown_path} and {metadata_path} "
        f"({len(set(turn.speaker for turn in turns))} speakers)"
    )


def main() -> None:
    app()


def _transcribe_files(files: list[Path], output_root: Path, settings: Settings) -> None:
    table = Table(title="Transcription Summary")
    table.add_column("File")
    table.add_column("Status")
    table.add_column("Output")
    table.add_column("Duration")
    table.add_column("Segments")
    table.add_column("Model")
    table.add_column("Device")
    table.add_column("Compute")
    table.add_column("Batch")

    transcriber: TranscriberProtocol | None = None

    for media_file in files:
        source_sha256 = file_sha256(media_file)
        file_output_dir = output_root / media_file.stem

        if should_skip_existing(file_output_dir, source_sha256, settings):
            table.add_row(
                media_file.name,
                "skipped",
                str(file_output_dir),
                "-",
                "-",
                settings.whisper_model_name,
                settings.whisper_device,
                settings.whisper_compute_type,
                str(settings.whisper_batch_size),
            )
            continue

        if transcriber is None:
            transcriber = create_transcriber(settings)

        result = transcriber.transcribe_file(
            media_file,
            source_sha256=source_sha256,
            source_file=media_file,
        )
        export_all(result, file_output_dir, settings)
        table.add_row(
            media_file.name,
            "transcribed",
            str(file_output_dir),
            format_timestamp(result.duration),
            str(len(result.segments)),
            result.model_name,
            result.device,
            result.compute_type,
            str(result.batch_size),
        )

    console.print(table)


if __name__ == "__main__":
    main()
