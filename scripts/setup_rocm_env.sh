#!/usr/bin/env bash
set -euo pipefail

echo "Checking AMD ROCm/HIP diagnostics..."

if command -v rocminfo >/dev/null 2>&1; then
  rocminfo | head -n 40 || true
else
  echo "rocminfo not found. Install ROCm/HIP runtime or ensure it is on PATH."
fi

if command -v rocm-smi >/dev/null 2>&1; then
  rocm-smi || true
else
  echo "rocm-smi not found. This may be normal on some Windows HIP SDK setups."
fi

python - <<'PY'
import importlib

for name in ("ctranslate2", "faster_whisper"):
    try:
        mod = importlib.import_module(name)
        print(f"{name}: OK", getattr(mod, "__version__", "unknown"))
    except Exception as exc:
        print(f"{name}: FAILED: {exc}")

try:
    from faster_whisper import WhisperModel

    WhisperModel("tiny", device="cuda", compute_type="float16")
    print("faster-whisper ROCm/HIP smoke test: OK")
except Exception as exc:
    print(f"faster-whisper ROCm/HIP smoke test: FAILED: {exc}")
    raise SystemExit(1)
PY
