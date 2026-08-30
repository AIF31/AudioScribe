"""Compare Seminar.m4a transcripts from large-v3 vs large-v3-turbo."""
import json
import re
from difflib import SequenceMatcher

BASE = "/home/alan1/WSL-Projects/Fast_Whisper/data/comparison"


def load(dir_name):
    meta = json.load(open(f"{BASE}/{dir_name}/Seminar/Seminar_metadata.json"))
    text = open(f"{BASE}/{dir_name}/Seminar/Seminar_transcript.md", encoding="utf-8").read()
    blocks = re.findall(r"\[(\d{2}:\d{2}:\d{2}) - (\d{2}:\d{2}:\d{2})\]\n(.*?)(?=\n\[|\Z)", text, re.S)
    segments = [(a, b, seg.strip()) for a, b, seg in blocks if seg.strip()]
    return meta, segments


meta_l, segs_l = load("large-v3")
meta_t, segs_t = load("turbo")

print("=== METADATA ===")
for label, m in (("large-v3", meta_l), ("large-v3-turbo", meta_t)):
    print(f"{label:15} detected={m.get('detected_language')}({m.get('language_probability')}) "
          f"segments={m['segment_count']} duration={round(m['duration'],1)}s")

print("\n=== TIMING (measured wall time, incl. model load) ===")
print("large-v3-turbo: 55.3s   (~50x realtime)")
print("large-v3      : 119.0s  (~23x realtime)  -> turbo 2.2x faster")

print(f"\n=== SEGMENTS: large-v3={len(segs_l)}, turbo={len(segs_t)} ===")

words_l = " ".join(s for _, _, s in segs_l).split()
words_t = " ".join(s for _, _, s in segs_t).split()
print(f"word counts: large-v3={len(words_l)}, turbo={len(words_t)}")

sm = SequenceMatcher(None, words_l, words_t, autojunk=False)
match_pct = sm.ratio() * 100
print(f"word-level similarity (SequenceMatcher ratio): {match_pct:.2f}%")

diff_words = sum(1 for tag, *_ in sm.get_opcodes() if tag != "equal"
                 for _ in range(0, 1))
inserts = sum(b2 - b1 for tag, b1, b2, _, _ in sm.get_opcodes() if tag in {"insert", "replace"})
deletes = sum(c2 - c1 for tag, _, _, c1, c2 in sm.get_opcodes() if tag in {"delete", "replace"})
print(f"differing words: ~{inserts + deletes} of {max(len(words_l), len(words_t))} "
      f"({(inserts + deletes) / max(len(words_l), len(words_t)) * 100:.2f}%)")

print("\n=== SEGMENT-BY-SEGMENT ALIGNMENT (first differences) ===")
shown = 0
n = min(len(segs_l), len(segs_t))
for i in range(n):
    a, b, ta = segs_l[i]
    _, _, tb = segs_t[i]
    if ta != tb and shown < 5:
        ratio = SequenceMatcher(None, ta.split(), tb.split()).ratio() * 100
        print(f"\n--- segment {i} [{a} - {b}] (similarity {ratio:.1f}%) ---")
        print(f"large-v3 : {ta[:220]}")
        print(f"turbo    : {tb[:220]}")
        shown += 1

identical = sum(1 for i in range(n) if segs_l[i][2] == segs_t[i][2])
print(f"\n=== SUMMARY ===")
print(f"identical segments: {identical}/{n} ({identical / n * 100:.1f}%)")
