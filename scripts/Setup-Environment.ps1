param(
    [string]$Python = 'python'
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path $PSScriptRoot -Parent
Push-Location $projectRoot
try {
    . "$PSScriptRoot\Enter-Project.ps1"
    if (-not (Test-Path '.venv\Scripts\python.exe')) {
        & $Python -m venv .venv
        if ($LASTEXITCODE -ne 0) { throw 'Python virtual environment creation failed.' }
    }
    $projectPython = Join-Path $projectRoot '.venv\Scripts\python.exe'
    & $projectPython -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
    if ($LASTEXITCODE -ne 0) { throw 'CUDA PyTorch installation failed.' }
    & $projectPython -m pip install -r requirements-windows.lock.txt
    if ($LASTEXITCODE -ne 0) { throw 'Training dependency installation failed.' }
    & $projectPython -m pip check
    if ($LASTEXITCODE -ne 0) { throw 'Dependency validation failed.' }
    & $projectPython scripts/check_gpu.py
    if ($LASTEXITCODE -ne 0) { throw 'GPU validation failed.' }
    & $projectPython scripts/smoke_training.py
    if ($LASTEXITCODE -ne 0) { throw 'Training validation failed.' }
} finally {
    Pop-Location
}
