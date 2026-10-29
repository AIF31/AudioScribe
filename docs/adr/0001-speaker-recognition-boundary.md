---
status: accepted
---

# Identify Enrolled Speakers After Diarization

AudioScribe identifies deliberately enrolled people after diarization and before speaker attribution and export. It builds one voiceprint per local speaker, compares that voiceprint with compatible speaker profiles under an open-set recognition policy, and preserves unknown, ambiguous, and insufficient-speech outcomes. This keeps identity consistent within a recording while retaining diarization's ability to discover speakers who have no profile.

## Considered Options

- Matching each transcript segment repeats embedding work and can assign conflicting identities to segments from the same local speaker.
- Replacing diarization with profile classification cannot discover unknown speakers or determine speaker turns.
- Cross-recording clustering into anonymous global identities introduces identity lifecycle and merge/split behavior that v1 does not need.

## Consequences

- The combined workflow needs a local speaker registry and speaker enrollment.
- Enrollment requires a reviewed single-speaker source; forcing diarization to one cluster cannot establish that the source contains only one person.
- Recognition outputs depend on the embedding model, recognition policy, and registry revision.
- Coarse transcript segments can still misattribute short interruptions; word-level attribution remains separate work.
- Backends without usable transcript timestamps cannot participate in combined speaker attribution.
