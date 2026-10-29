# AudioScribe

AudioScribe turns recordings into timestamped transcripts and can attribute speech to recording-local or enrolled speakers.

## Language

### Transcription

**Recording**:
An audio or video source submitted to AudioScribe for processing.
_Avoid_: Media file, input file

**Transcript**:
The complete text derived from one recording, including its ordered transcript segments.
_Avoid_: Transcription, document

**Transcript Segment**:
A time-bounded passage of text within a transcript.
_Avoid_: Chunk, line

### Speaker attribution

**Diarization**:
The division of a recording into speaker turns grouped by recording-local speakers.
_Avoid_: Speaker recognition, speaker identification

**Speaker Turn**:
A time range during which one local speaker is active.
_Avoid_: Utterance, speaker segment

**Local Speaker**:
A speaker cluster that is meaningful only within one recording, represented by a label such as `SPEAKER_00`.
_Avoid_: Speaker identity, person

**Speaker Attribution**:
The association of a transcript segment with a local speaker and, when available, a speaker profile.
_Avoid_: Speaker labeling

### Speaker identity

**Speaker Profile**:
A stable identity enrolled in AudioScribe, with a speaker ID, display name, and voiceprints produced by one embedding model.
_Avoid_: User, contact, account

**Speaker ID**:
The immutable machine identifier of a speaker profile.
_Avoid_: Speaker label, display name

**Display Name**:
The mutable human-readable name of a speaker profile shown in recognized transcripts.
_Avoid_: Speaker ID, local speaker label

**Enrollment Sample**:
A reviewed recording that contains speech from exactly one known person and contributes evidence to a speaker profile.
_Avoid_: Training sample, reference speaker

**Voiceprint**:
A biometric embedding derived from an enrollment sample or from the turns of one local speaker.
_Avoid_: Voice ID, voice signature

**Speaker Enrollment**:
The creation or update of a speaker profile from one or more enrollment samples.
_Avoid_: Training, registration

**Speaker Identification**:
Open-set comparison of a local speaker's voiceprint with compatible speaker profiles.
_Avoid_: Diarization, closed-set classification

**Speaker Match**:
The result of speaker identification, including the status and comparison evidence for one local speaker.
_Avoid_: Guess, classification

**Known Speaker**:
A speaker match that satisfies the recognition policy for one speaker profile.
_Avoid_: Recognized person

**Unknown Speaker**:
A speaker match for which no profile satisfies the recognition policy.
_Avoid_: Unregistered user

**Ambiguous Speaker**:
A speaker match where competing profiles are too close for a reliable identity decision.
_Avoid_: Unknown speaker

**Insufficient-Speech Speaker**:
A local speaker whose usable turns do not contain enough speech to produce a reliable voiceprint.
_Avoid_: Unknown speaker

**Recognition Policy**:
The model-specific acceptance threshold, ambiguity margin, and usable-speech requirements applied to speaker matches.
_Avoid_: Matching config, confidence setting

**Speaker Registry**:
The local collection of speaker profiles used for enrollment and identification.
_Avoid_: Speaker database, address book

**Registry Revision**:
A value that changes whenever speaker profile data or display names change, invalidating recognition outputs derived from an older registry state.
_Avoid_: Schema version

