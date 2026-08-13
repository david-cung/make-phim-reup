param(
    [string]$Python = ".venv-worker\Scripts\python.exe"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $Python)) {
    throw "Worker Python not found at '$Python'. Create .venv-worker first."
}

Write-Host "Installing the pinned local F5-TTS Vietnamese runtime..."
& $Python -m pip install -e "python[f5]"

Write-Host ""
Write-Host "F5 runtime installed."
Write-Host "Install the 5.45 GB ViVoice model explicitly from Voice Over > QUALITY."
Write-Host "For CUDA acceleration, install the PyTorch build matching your CUDA driver."
