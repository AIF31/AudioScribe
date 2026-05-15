param(
    [string]$Python = "py -3.12",
    [string]$VenvPath = ".venv-win-hip",
    [string]$CTranslate2RocmWheel = $env:CT2_ROCM_WHEEL,
    [string]$CTranslate2RocmZip = $env:CT2_ROCM_ZIP
)

$ErrorActionPreference = "Stop"

function Invoke-Python {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Arguments)
    $parts = $Python -split " "
    $exe = $parts[0]
    $baseArgs = @()
    if ($parts.Length -gt 1) {
        $baseArgs = $parts[1..($parts.Length - 1)] | Where-Object { $_ }
    }
    & $exe @baseArgs @Arguments
}

function Get-VenvPython {
    Join-Path $VenvPath "Scripts\python.exe"
}

Write-Host "Checking Python..."
Invoke-Python --version

if (-not (Test-Path $VenvPath)) {
    Invoke-Python -m venv $VenvPath
}

$venvPython = Get-VenvPython
& $venvPython -m pip install --upgrade pip
& $venvPython -m pip install -e ".[dev]"

if ($CTranslate2RocmZip) {
    if (-not (Test-Path $CTranslate2RocmZip)) {
        throw "CT2_ROCM_ZIP was set, but the file does not exist: $CTranslate2RocmZip"
    }
    $extractDir = Join-Path $env:TEMP "audioscribe-ct2-rocm"
    if (Test-Path $extractDir) {
        Remove-Item -Recurse -Force $extractDir
    }
    Expand-Archive -Path $CTranslate2RocmZip -DestinationPath $extractDir
    $wheel = Get-ChildItem -Path $extractDir -Recurse -Filter "ctranslate2-*-cp312-*-win_amd64.whl" |
        Select-Object -First 1
    if (-not $wheel) {
        throw "No Python 3.12 Windows CTranslate2 wheel found in $CTranslate2RocmZip"
    }
    $CTranslate2RocmWheel = $wheel.FullName
}

if ($CTranslate2RocmWheel) {
    if (-not (Test-Path $CTranslate2RocmWheel)) {
        throw "CT2_ROCM_WHEEL was set, but the file does not exist: $CTranslate2RocmWheel"
    }
    & $venvPython -m pip install --force-reinstall $CTranslate2RocmWheel
} else {
    Write-Warning "No CTranslate2 ROCm wheel was provided. Set CT2_ROCM_WHEEL or CT2_ROCM_ZIP before expecting GPU execution."
}

Write-Host "Checking HIP/ROCm tools on PATH..."
foreach ($tool in @("hipcc", "rocminfo", "amd-smi")) {
    $command = Get-Command $tool -ErrorAction SilentlyContinue
    if ($command) {
        Write-Host "$tool: $($command.Source)"
    } else {
        Write-Warning "$tool not found on PATH"
    }
}

Write-Host "Checking CTranslate2..."
$ct2Check = @'
import ctranslate2

print("ctranslate2", ctranslate2.__version__)
print("module", ctranslate2.__file__)
try:
    print("gpu_count", ctranslate2.get_cuda_device_count())
except Exception as exc:
    print("gpu_count_error", type(exc).__name__, exc)
try:
    print("cuda_compute_types", ctranslate2.get_supported_compute_types("cuda"))
except Exception as exc:
    print("cuda_compute_types_error", type(exc).__name__, exc)
'@
$ct2Check | & $venvPython -

Write-Host "Windows HIP environment created at $VenvPath"
Write-Host "Use $venvPython to run scripts/windows_hip_smoke_test.ps1"
