# Dot-source this file: . .\scripts\Enter-Project.ps1
$projectRoot = Split-Path $PSScriptRoot -Parent
$env:HF_HOME = Join-Path $projectRoot '.cache\huggingface'
$env:PIP_CACHE_DIR = Join-Path $projectRoot '.cache\pip'
$env:TEMP = Join-Path $projectRoot '.cache\tmp'
$env:TMP = $env:TEMP
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'
$env:HF_HUB_DISABLE_TELEMETRY = '1'
$env:DO_NOT_TRACK = '1'
New-Item -ItemType Directory -Force $env:TEMP | Out-Null
Write-Host "Project Python: $projectRoot\.venv\Scripts\python.exe"
