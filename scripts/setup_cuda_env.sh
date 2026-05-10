#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"

CUDA_LIBRARY_PATH="$("$PYTHON_BIN" - <<'PY'
import nvidia.cublas.lib
import nvidia.cudnn.lib

paths = [
    next(iter(nvidia.cublas.lib.__path__)),
    next(iter(nvidia.cudnn.lib.__path__)),
]

try:
    import nvidia.cuda_nvrtc.lib

    paths.append(next(iter(nvidia.cuda_nvrtc.lib.__path__)))
except Exception:
    pass

print(":".join(paths))
PY
)"

export LD_LIBRARY_PATH="${CUDA_LIBRARY_PATH}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
