# Local Speaker Diarization

AudioScribe can add local speaker labels to an existing timestamped transcript with
`pyannote.audio`. Media remains on the local machine. PyAV decodes audio and pyannote runs the
speaker model on CUDA or CPU.

## Model Access

The default pipeline uses these gated Hugging Face models:

- `pyannote/speaker-diarization-3.1`
- `pyannote/segmentation-3.0`

Accept the conditions on both model pages, create a read token at
<https://huggingface.co/settings/tokens>, and set it only in `.env`:

```env
HF_TOKEN=hf_your_token_here
DIARIZATION_MODEL=pyannote/speaker-diarization-3.1
DIARIZATION_DEVICE=cuda
```

Do not commit `.env`.

## Install

The versions are intentionally pinned. pyannote.audio 4.x uses a different gated model, and
Hugging Face Hub 1.x removed an authentication argument required by pyannote.audio 3.3.1.
Torch 2.4 also avoids the checkpoint loading default changed in torch 2.6.

For NVIDIA CUDA 12:

```bash
source .venv/bin/activate
python -m pip install \
  torch==2.4.1 torchaudio==2.4.1 \
  --index-url https://download.pytorch.org/whl/cu121
python -m pip install -e ".[diarize]"
```

The normal faster-whisper CUDA setup remains unchanged:

```bash
source scripts/setup_cuda_env.sh
```

System FFmpeg is not required. The diarizer decodes supported media through PyAV and sends a
16 kHz mono waveform directly to pyannote, bypassing torchcodec and torchaudio file decoding.

## Workflow

Transcribe first:

```bash
audio-transcribe transcribe-file ./data/audio_raw/meeting.mp4
```

Then add speaker labels:

```bash
audio-transcribe diarize-file ./data/audio_raw/meeting.mp4
```

If the number of participants is known, exact speaker count gives the most stable clustering:

```bash
audio-transcribe diarize-file ./data/audio_raw/meeting.mp4 \
  --num-speakers 3
```

Otherwise constrain the search to a reasonable range:

```bash
audio-transcribe diarize-file ./data/audio_raw/meeting.mp4 \
  --min-speakers 2 \
  --max-speakers 4
```

Use a renamed or externally edited transcript with `--transcript`:

```bash
audio-transcribe diarize-file ./data/audio_raw/meeting.mp4 \
  --transcript ./data/transcripts/meeting/reviewed.md \
  --min-speakers 2 \
  --max-speakers 4
```

The transcript must contain timestamp lines in AudioScribe's normal format:

```text
[00:01:12 - 00:01:30]
Transcript text.
```

## Outputs

The command preserves the original transcript and writes:

```text
data/transcripts/meeting/
  meeting_transcript.md
  meeting_diarized.md
  meeting_diarized_metadata.json
```

The Markdown uses anonymous, recording-local labels:

```text
[00:01:12 - 00:01:30] SPEAKER_01
Transcript text.
```

Labels are not identities and are not stable across different recordings. Rename them manually
after reviewing representative sections if participant names are known.

## Assignment Method

Whisper and pyannote produce independent time ranges. AudioScribe assigns each complete Whisper
segment to the pyannote speaker with the greatest overlapping duration. If there is no overlap, it
uses the nearest speaker turn.

This is reliable for normal alternating conversation, but a short interruption inside a long
Whisper segment may not receive a separate label. The metadata can therefore list more pyannote
clusters than appear as dominant labels in the Markdown transcript. Exact word-level speaker
alignment would require word timestamps and text splitting, which this post-processing command
does not attempt.

## Troubleshooting

### Gated model errors

Confirm model terms are accepted for both repositories and that `HF_TOKEN` is a valid read token.
The token can be checked without printing it:

```bash
audio-transcribe inspect-config
```

The token is displayed as `***` when set.

### CUDA memory

Run transcription and diarization sequentially. Each command exits and releases its model before
the next command starts. To use CPU for diarization:

```env
DIARIZATION_DEVICE=cpu
```

### Speaker count looks wrong

Use `--num-speakers` when participant count is known. Otherwise narrow `--min-speakers` and
`--max-speakers`. Background audio, overlapping speech, and very short interjections can create
small extra clusters.

### Warnings

The torch checkpoint security warning and pyannote TF32 reproducibility warning are expected for
this pinned local model stack. Only load the documented Hugging Face model repositories.
