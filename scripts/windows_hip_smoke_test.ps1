param(
    [string]$Python = ".venv-win-hip\Scripts\python.exe",
    [Parameter(Mandatory = $true)]
    [string]$AudioPath,
    [string]$OutputDir = "data\transcripts",
    [string]$ComputeType = "float16",
    [int]$BatchSize = 4
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $Python)) {
    throw "Python not found: $Python. Run scripts/setup_windows_hip_env.ps1 first."
}

if (-not (Test-Path $AudioPath)) {
    throw "Audio file not found: $AudioPath"
}

$env:TRANSCRIPTION_BACKEND = "faster-whisper"
$env:WHISPER_ACCELERATOR = "rocm"
$env:WHISPER_DEVICE = "cuda"
$env:WHISPER_COMPUTE_TYPE = $ComputeType
$env:WHISPER_BATCH_SIZE = [string]$BatchSize
$env:WHISPER_ALLOW_CPU_FALLBACK = "false"

Write-Host "Python:"
& $Python --version

Write-Host "HIP/ROCm tools:"
foreach ($tool in @("hipcc", "rocminfo", "amd-smi")) {
    $command = Get-Command $tool -ErrorAction SilentlyContinue
    if ($command) {
        Write-Host "$tool: $($command.Source)"
    } else {
        Write-Warning "$tool not found on PATH"
    }
}

Write-Host "CTranslate2:"
$ct2Check = @'
import ctranslate2

print("ctranslate2", ctranslate2.__version__)
print("module", ctranslate2.__file__)
print("gpu_count", ctranslate2.get_cuda_device_count())
print("cuda_compute_types", ctranslate2.get_supported_compute_types("cuda"))
'@
$ct2Check | & $Python -

Write-Host "AudioScribe accelerator check:"
& $Python -m audio_transcriber.cli check-accelerator

Write-Host "Transcribing with CPU fallback disabled:"
& $Python -m audio_transcriber.cli transcribe-file $AudioPath --output-dir $OutputDir

$stem = [System.IO.Path]::GetFileNameWithoutExtension($AudioPath)
$metadataPath = Join-Path (Join-Path $OutputDir $stem) "$stem`_metadata.json"
if (-not (Test-Path $metadataPath)) {
    throw "Metadata was not generated: $metadataPath"
}

$metadata = Get-Content $metadataPath -Raw | ConvertFrom-Json
if ($metadata.accelerator -ne "rocm" -or $metadata.device -ne "cuda") {
    throw "Expected ROCm GPU metadata, got accelerator=$($metadata.accelerator), device=$($metadata.device)"
}

Write-Host "ROCm GPU smoke test passed. Metadata: $metadataPath"
