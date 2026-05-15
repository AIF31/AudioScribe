# Windows HIP SDK GPU Setup

This path is for AMD GPUs that are supported by the Windows HIP SDK, such as the Radeon RX 6950 XT (`gfx1030`). It is separate from WSL2 ROCm. If WSL does not expose `/dev/kfd`, `/dev/dri`, or `/dev/dxg`, run the GPU test from Windows PowerShell instead of Linux.

> Windows HIP support in AudioScribe is experimental. It depends on the AMD driver, HIP SDK version, Python version, and CTranslate2 ROCm/HIP wheel or local build compatibility.

## Requirements

- Windows 11 with a current AMD Adrenalin driver.
- AMD HIP SDK for Windows installed.
- Python 3.12 x64 for Windows. Python 3.14 is not supported by AudioScribe because the project requires Python `<3.13`.
- A CTranslate2 ROCm/HIP Windows wheel for Python 3.12, or a local CTranslate2 build made with HIP enabled.

## Setup

Open PowerShell in your AudioScribe clone:

```powershell
cd C:\path\to\AudioScribe
```

Install Python 3.12 if `py -3.12 --version` does not work.

Install AMD HIP SDK for Windows and make sure its tools are available:

```powershell
hipcc --version
rocminfo
```

Download the CTranslate2 ROCm Windows wheel bundle from the CTranslate2 v4.7.1 GitHub release. Then run one of:

```powershell
$env:CT2_ROCM_ZIP = "C:\path\to\rocm-python-wheels-Windows.zip"
.\scripts\setup_windows_hip_env.ps1
```

or:

```powershell
$env:CT2_ROCM_WHEEL = "C:\path\to\ctranslate2-4.7.1-cp312-cp312-win_amd64.whl"
.\scripts\setup_windows_hip_env.ps1
```

The script creates `.venv-win-hip`, installs AudioScribe, installs the ROCm-enabled CTranslate2 wheel when provided, and prints CTranslate2 GPU diagnostics.

## GPU Smoke Test

Run the test with CPU fallback disabled:

```powershell
.\scripts\windows_hip_smoke_test.ps1 -AudioPath "data\audio_raw\sample.m4a"
```

The script requires `-AudioPath` so it does not assume any private local recording exists in the repository.

Expected successful metadata:

```json
{
  "requested_accelerator": "rocm",
  "requested_device": "cuda",
  "effective_device": "cuda",
  "accelerator": "rocm",
  "device": "cuda"
}
```

If the metadata says `accelerator: cpu`, the test did not pass. The smoke script disables CPU fallback and fails in that case.

## Troubleshooting

- `py -3.12` missing: install Windows Python 3.12 x64.
- `hipcc` missing: install AMD HIP SDK for Windows or load the HIP SDK environment in PowerShell.
- CTranslate2 GPU count is `0`: the installed CTranslate2 package is not a ROCm/HIP build, the HIP SDK is not visible, or the wheel/runtime versions do not match.
- `hipblas` or related DLL errors: the CTranslate2 ROCm wheel does not match the installed HIP SDK. Build CTranslate2 locally against the installed SDK.
- WSL sees no AMD device nodes: use this Windows HIP path instead of WSL.
