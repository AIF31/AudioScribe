# AMD ROCm/HIP Setup

AudioScribe can experimentally run local `faster-whisper` transcription on AMD GPUs through CTranslate2 ROCm/HIP builds.

> AMD ROCm/HIP support is experimental. Treat this as a best-effort local acceleration path, validate it with a small smoke test, and keep CPU fallback available for production batches until your exact GPU/runtime combination is proven stable.

This feature is experimental because success depends on the AMD GPU, operating system, ROCm/HIP runtime version, and whether the installed CTranslate2 package was built with HIP support.

For AMD GPUs supported by the Windows HIP SDK, such as the Radeon RX 6950 XT, see [Windows HIP SDK GPU Setup](windows-hip.md). That path is required when WSL does not expose an AMD compute device to Linux.

## Which AMD Path Should I Use?

| Environment | Recommended path |
|---|---|
| Native Linux with ROCm-supported AMD GPU | Use this guide with a ROCm/HIP-enabled CTranslate2 wheel or source build. |
| WSL2 with AMD GPU exposed to Linux | Use this guide only if ROCm/HIP tools and device nodes are visible inside WSL. |
| Windows with HIP SDK-supported AMD GPU | Use [Windows HIP SDK GPU Setup](windows-hip.md). |
| WSL2 with no AMD device nodes visible | Use [Windows HIP SDK GPU Setup](windows-hip.md) or native Linux. |

## Important Naming Detail

Use:

```env
WHISPER_ACCELERATOR=rocm
WHISPER_DEVICE=cuda
```

CTranslate2/faster-whisper use `device="cuda"` as the GPU device string even when the backend is ROCm/HIP.

Do not set `WHISPER_DEVICE=rocm`. That is not a CTranslate2/faster-whisper device name.

## Option A: CTranslate2 ROCm Wheel

Install AudioScribe without CUDA extras:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

Install the ROCm/HIP CTranslate2 wheel matching your Python, OS, and ROCm version from:

```text
https://github.com/OpenNMT/CTranslate2/releases
```

```bash
python -m pip install --force-reinstall /path/to/ctranslate2-*-rocm*.whl
cp .env.rocm.example .env
audio-transcribe check-accelerator
```

If the install replaces the ROCm wheel with a normal PyPI wheel later, reinstall the ROCm wheel and rerun `audio-transcribe check-accelerator`.

## Option B: Build CTranslate2 From Source

```bash
git clone --recursive https://github.com/OpenNMT/CTranslate2.git
cd CTranslate2
git checkout v4.7.1

mkdir build
cd build
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DWITH_HIP=ON \
  -DWITH_OPENBLAS=ON
make -j"$(nproc)"
sudo make install
sudo ldconfig

cd ../python
python -m pip install -r install_requirements.txt
python setup.py bdist_wheel
python -m pip install --force-reinstall dist/*.whl
```

If installed to a custom prefix, set `CTRANSLATE2_ROOT` during wrapper build and add the CTranslate2 library path to `LD_LIBRARY_PATH`.

Use this option when no matching ROCm/HIP wheel exists for your Python, OS, or ROCm/HIP runtime.

## Recommended Initial Settings

```env
WHISPER_MODEL_NAME=large-v3
WHISPER_ACCELERATOR=rocm
WHISPER_DEVICE=cuda
WHISPER_COMPUTE_TYPE=float16
WHISPER_BATCH_SIZE=4
WHISPER_ALLOW_CPU_FALLBACK=true
```

If memory or ROCm errors occur:

```env
WHISPER_COMPUTE_TYPE=int8_float16
WHISPER_BATCH_SIZE=2
```

If still unstable:

```env
WHISPER_ACCELERATOR=cpu
WHISPER_DEVICE=cpu
WHISPER_COMPUTE_TYPE=int8
WHISPER_BATCH_SIZE=1
```

## Smoke Test

```bash
cp .env.rocm.example .env
audio-transcribe inspect-config
audio-transcribe check-accelerator
audio-transcribe transcribe-file ./data/audio_raw/example.m4a
```

For a strict GPU-only smoke test, disable CPU fallback:

```bash
WHISPER_ALLOW_CPU_FALLBACK=false audio-transcribe transcribe-file ./data/audio_raw/example.m4a
```

Successful AMD GPU metadata should show:

```json
{
  "requested_accelerator": "rocm",
  "requested_device": "cuda",
  "effective_device": "cuda",
  "accelerator": "rocm",
  "device": "cuda"
}
```

If metadata shows `accelerator: cpu`, ROCm initialization failed and AudioScribe used CPU fallback.

## Notes

- Linux support depends on AMD's current ROCm support matrix for your GPU.
- Windows AMD HIP SDK support depends on GPU and HIP SDK version, and CTranslate2 ROCm wheel compatibility may lag. Use the dedicated [Windows HIP guide](windows-hip.md) for RX 6950 XT style setups.
- `rocminfo`, `rocm-smi`, `amd-smi`, and `hipcc` are useful diagnostics but may not all be available on every supported OS.
- Do not install `.[cuda]` for AMD. It installs NVIDIA libraries.
- Do not use PyTorch ROCm checks to validate AudioScribe. The local backend is CTranslate2/faster-whisper.

## Troubleshooting

- `rocminfo` missing: install ROCm/HIP runtime or confirm PATH setup.
- `rocm-smi` missing: this may be normal on some Windows HIP SDK setups.
- CTranslate2 GPU device count is `0`: the installed CTranslate2 package cannot see a GPU. Check ROCm/HIP runtime visibility and confirm the package is a ROCm/HIP build.
- `hipblas` or `rocblas` errors: the installed CTranslate2 wheel likely does not match your ROCm/HIP runtime.
- GPU memory access fault: reduce batch size, test with a smaller model first, and check upstream CTranslate2 ROCm issues.
