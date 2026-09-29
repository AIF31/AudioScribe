# Local Enrolled-Speaker Recognition Research

Research date: 2026-09-29

## Recommendation

Add enrolled-speaker identification after diarization and before export, as proposed by
[ADR 0001](../adr/0001-speaker-recognition-boundary.md). For the first backend, ask the existing
`pyannote/speaker-diarization-3.1` pipeline for `return_embeddings=True` and match the returned
representative embedding for each recording-local speaker against enrolled profiles. In
`pyannote.audio` 3.3.1, the returned row `embeddings[i]` is explicitly aligned with
`diarization.labels()[i]`; the pipeline constructs those rows from its clustering centroids and
reorders them after assigning human-readable local labels. This avoids a second embedding pass and
keeps diarization and enrollment in one embedding space. See the official
[3.3.1 diarization implementation](https://github.com/pyannote/pyannote-audio/blob/3.3.1/pyannote/audio/pipelines/speaker_diarization.py).

Enroll only deliberately provided samples that a person has reviewed and confirmed contain one
known speaker. An unconstrained diarization pass can flag likely contamination, but forcing
`num_speakers=1` only requests one cluster; it cannot validate the sample. Accept a known identity
only when the best score passes an acceptance threshold and is sufficiently separated from the
runner-up. Unknown, ambiguous, and insufficient-speech outcomes retain the recording-local label
in Markdown and have distinct statuses in metadata. Threshold and margin values need calibration
on representative recordings; the primary sources supply no universal values.

## Diarization and identification are different stages

Speaker diarization partitions a recording into turns and returns recording-local speaker labels.
The `speaker-diarization-3.1` card describes an `Annotation` result and the 3.3.1 implementation
generates labels such as `SPEAKER_00`; it does not attach enrolled identities. See the official
[pipeline card](https://huggingface.co/pyannote/speaker-diarization-3.1) and
[3.3.1 source](https://github.com/pyannote/pyannote-audio/blob/3.3.1/pyannote/audio/pipelines/speaker_diarization.py).

Enrolled identification adds a gallery of known people, compares a diarized speaker's embedding
with that gallery, and permits rejection when the evidence is insufficient. AudioScribe should
therefore preserve the local cluster label independently from any recognized identity. A rejected
or ambiguous match remains a valid diarization result.

This fits the current repository boundary: [`run_diarization`](../../src/audio_transcriber/diarizer.py)
returns local turns, [`assign_speakers`](../../src/audio_transcriber/diarizer.py) aligns transcript
segments to those turns, and export currently writes the local labels. Recognition should produce
a local-label-to-match mapping without changing transcript timing or clustering.

## Embedding extraction choices

### Preferred v1: reuse diarization centroids

`pyannote.audio` 3.3.1 supports `return_embeddings=True` on the speaker diarization pipeline and
documents the returned values as representative embeddings aligned with the final label order.
The same source shows that embedding extraction can exclude overlap and that clustering produces
one centroid per detected cluster. It also documents a rare edge case in which missing centroids
are padded with zero vectors, so the application must reject zero-norm and non-finite embeddings.
See the official
[speaker diarization implementation](https://github.com/pyannote/pyannote-audio/blob/3.3.1/pyannote/audio/pipelines/speaker_diarization.py).

Enrollment must produce vectors in exactly that same embedding space. Require explicit human
confirmation that the selected clip contains only the named person, then run an unconstrained
diarization check and reject detected additional speakers. After those checks, run the same pinned
pipeline with `num_speakers=1` and `return_embeddings=True` to extract one representative vector.
The forced one-speaker run is an extraction step, not proof of sample purity. A later optimization
may invoke the pipeline's embedding model directly only after the model identity, preprocessing,
and output contract are included in the same fingerprint.

### Alternative: standalone pyannote embedding

The `pyannote/embedding` model supports whole-input inference with
`Inference(model, window="whole")` and excerpt inference with `Inference.crop`. Its model card uses
cosine distance for verification and also demonstrates three-second sliding windows. See the
official [embedding model card](https://huggingface.co/pyannote/embedding). In 3.3.1,
`Inference.crop` also accepts a list of segments; with whole-window inference it concatenates those
segments into one artificial chunk. See the official
[3.3.1 inference implementation](https://github.com/pyannote/pyannote-audio/blob/3.3.1/pyannote/audio/core/inference.py).

This is useful if AudioScribe later needs explicit turn selection, overlap removal, or multiple
embeddings per cluster. It should be a separate adapter and a separate profile namespace. The
`pyannote/embedding` card states that it relies on pyannote.audio 2.1, while this repository pins
3.3.1, so loading and scoring it in the pinned environment remains a compatibility spike rather
than an assumed capability. See the same official
[model card](https://huggingface.co/pyannote/embedding).

### Alternative: SpeechBrain ECAPA

SpeechBrain's `spkrec-ecapa-voxceleb` model can extract embeddings and perform verification with a
cosine score. It was trained on VoxCeleb1 and VoxCeleb2, expects 16 kHz single-channel tensors for
batch embedding calls, reports 0.80% EER on the cleaned VoxCeleb1 test set, and explicitly gives no
performance warranty for other datasets. See the official
[SpeechBrain model card](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb).

The pyannote 3.3.1 embedding adapter recognizes SpeechBrain models, supplies masks, exposes cosine
as their metric, and detects a model-specific minimum number of samples. See the official
[3.3.1 speaker verification implementation](https://github.com/pyannote/pyannote-audio/blob/3.3.1/pyannote/audio/pipelines/speaker_verification.py).
SpeechBrain is viable as a later backend, but it should not be introduced in v1 unless it
outperforms the existing pipeline's embedding space on AudioScribe's calibration set.

## Scoring and open-set policy

Normalize every backend to one application convention: **cosine similarity, higher is better**.
This prevents threshold inversion because the pyannote embedding example uses cosine distance,
while SpeechBrain's implementation computes `CosineSimilarity` and accepts when `score >
threshold`. See the official [pyannote model card](https://huggingface.co/pyannote/embedding) and
[SpeechBrain source](https://github.com/speechbrain/speechbrain/blob/develop/speechbrain/inference/speaker.py).

For each probe cluster:

1. Report a configuration error before processing when the registry has no compatible profile.
2. Normalize each enrollment sample vector, then normalize their mean into one profile centroid.
3. Score the probe against each compatible profile centroid.
4. Let `best` and `runner_up` be the two highest profile scores.
5. Return `Known` only when `best >= acceptance_threshold` and
   `best - runner_up >= ambiguity_margin`.
6. Return `Unknown` when the acceptance threshold fails and `Ambiguous` when only the margin fails.

Require an explicit model-specific acceptance threshold and ambiguity margin for v1 until a
calibrated policy artifact supplies defaults. SpeechBrain's official verification recipe computes positive and negative
scores, derives a threshold with its EER utility, and optionally performs cohort normalization;
the ECAPA model card also warns that performance on other datasets is not guaranteed. See the
official
[verification recipe](https://github.com/speechbrain/speechbrain/blob/develop/recipes/VoxCeleb/SpeakerRec/speaker_verification_cosine.py)
and [model card](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb).

To publish a default, first define a false-accept target and a consented labeled dataset containing
same-person and impostor trials from the intended microphones, rooms, languages, compression, and
noise conditions. Select the operating threshold for that target, then tune the runner-up margin
separately for multi-profile ambiguity. Record the dataset revision, trial counts, target, measured
error rates, and resulting policy version. A single positive and negative smoke trial verifies
compatibility only; it cannot establish a default threshold.

## Speech quality and duration

The official sources do not define a universal minimum enrollment duration or number of clips.
The SpeechBrain verification recipe prepares three-second segments, and the pyannote embedding
card demonstrates three-second sliding windows, but these are examples rather than quality
guarantees. See the official
[SpeechBrain recipe](https://github.com/speechbrain/speechbrain/blob/develop/recipes/VoxCeleb/SpeakerRec/speaker_verification_cosine.py)
and [pyannote model card](https://huggingface.co/pyannote/embedding).

Keep two separate limits:

- The **runtime floor** is the backend's minimum processable sample count. The pyannote 3.3.1
  adapters discover this per model and return invalid embeddings for shorter input. See the
  official
  [speaker verification implementation](https://github.com/pyannote/pyannote-audio/blob/3.3.1/pyannote/audio/pipelines/speaker_verification.py).
- The **quality floor** is an AudioScribe policy based on clean voiced duration and enrollment
  diversity. Start it as configuration, report why a sample failed, and set its final value from
  calibration rather than treating the runtime floor as evidence of reliable recognition.

Prefer single-speaker, non-overlapped material for enrollment and capture more than one sample per
person across realistic conditions. For recording probes, the existing pyannote pipeline can
exclude overlap during embedding extraction and falls back to all speech when non-overlapped
speech is too short. See the official
[3.3.1 diarization source](https://github.com/pyannote/pyannote-audio/blob/3.3.1/pyannote/audio/pipelines/speaker_diarization.py).

## Compatibility with the pinned stack

The repository pins `pyannote.audio==3.3.1`, `torch==2.4.1`, and `torchaudio==2.4.1` in
[`pyproject.toml`](../../pyproject.toml). The `speaker-diarization-3.1` card requires
pyannote.audio 3.1 or newer and states that segmentation and embedding run in pure PyTorch, so the
pinned pyannote version meets the model's documented minimum. See the official
[pipeline card](https://huggingface.co/pyannote/speaker-diarization-3.1).

The official 3.3.1 dependency list requires Torch 2.0 or newer, torchaudio 2.2 or newer, and
SpeechBrain 1.0 or newer; the project's Torch and torchaudio pins satisfy those declared ranges.
The 3.3.1 release also records a fix for SpeechBrain 1.x support. See the official
[3.3.1 requirements](https://raw.githubusercontent.com/pyannote/pyannote-audio/3.3.1/requirements.txt)
and [release notes](https://github.com/pyannote/pyannote-audio/releases/tag/3.3.1).

Declared compatibility is not a substitute for a local smoke check. Before implementation is
accepted, verify `return_embeddings=True` on CPU and the project's supported accelerator, confirm
finite nonzero centroids, and compare one same-person and one different-person trial as a smoke
check. These two trials do not calibrate an acceptance threshold.
The gated pyannote repositories also require accepted model conditions and a Hugging Face token for
download, as documented by the official
[diarization card](https://huggingface.co/pyannote/speaker-diarization-3.1).

## Intended architecture

Keep model-specific extraction behind an adapter and keep matching pure:

| Component | Responsibility | Stable contract |
| --- | --- | --- |
| Diarization adapter | Produce speaker turns plus one probe embedding per local label | `turns`, `label -> vector`, embedding-space fingerprint |
| Enrollment service | Require reviewed single-speaker confirmation, screen for extra speakers, and extract one vector | identity, vector, quality metadata, source hash |
| Profile registry | Store identities and sample vectors grouped by embedding space | versioned transactional SQLite registry; no cross-space lookup |
| Matcher | Score a probe against compatible profile centroids and apply open-set policy | `Known`, `Unknown`, `Ambiguous`, or `Insufficient Speech` with scores and reason |
| Transcript integration | Apply the local-label mapping before Markdown and metadata export | unresolved labels remain visible and auditable |

The embedding-space fingerprint should include backend type, immutable commit or content digests
for the pipeline configuration and every loaded segmentation and embedding model, library version,
vector dimension, sample rate, preprocessing policy, score convention, and schema version. Resolve
all components before enrollment or identification; fail closed when a component cannot be
fingerprinted. A profile from a different fingerprint is incompatible rather than merely a
low-scoring candidate.

Store multiple enrollment sample vectors and their quality/source metadata, plus the derived
normalized profile centroid. Score the centroid in v1 and rebuild it whenever samples change.
Keep the registry in one transactional SQLite database so profile data and its content-hash
revision commit together. Store model and policy revisions with every recognition result. Markdown
displays the accepted name or the original `SPEAKER_nn` label; JSON metadata retains that local
label, decision (`known`, `unknown`, `ambiguous`, or `insufficient_speech`), best and runner-up
scores, margin, thresholds, probe duration, registry revision, and decision reason.

The high Recording Processing seam should return stage statuses and artifacts to the CLI. Backend
adapters can then be replaced in tests, while the matcher and registry remain independent of
pyannote objects. Cache identities follow the dependency graph: transcription depends on media
and transcription settings; diarization depends on media and its model settings; attribution
depends on transcript segments and diarization; identification depends on cached Voiceprints,
registry revision, and matching policy; final export depends on attribution and identification.
An edited transcript therefore reattributes text without rerunning pyannote.

## Unresolved risks

- The primary sources provide no universal open-set threshold, ambiguity margin, reliable minimum
  enrollment duration, or required enrollment count. V1 requires explicit thresholds until a
  representative calibration artifact supports defaults.
- VoxCeleb EER values do not predict AudioScribe's meeting, lecture, microphone, language, or noise
  conditions; the SpeechBrain model card explicitly disclaims performance guarantees on other
  datasets.
- A remote model or pipeline revision can change the embedding space. Enrollment and identification
  fail when immutable revisions or content digests for loaded components cannot be resolved.
- Very short clusters, overlap, reverberation, noise, compression, channel mismatch, and voice
  changes can shift scores. These conditions need representation in calibration data.
- The diarization implementation can return a zero-padded centroid in a rare speaker-count edge
  case; validation must reject it rather than allow undefined cosine scoring.
- The standalone `pyannote/embedding` card targets pyannote.audio 2.1. Its use with the pinned 3.3.1
  environment remains unverified.
- CPU, CUDA, ROCm/HIP, torchaudio binary compatibility, and gated-model download behavior still
  require an environment smoke check.
