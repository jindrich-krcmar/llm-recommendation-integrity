#!/usr/bin/env bash
# Untested Linux/CUDA bootstrap; this is not a validated Linux environment lock.
set -euo pipefail
if [[ "$(uname -s)" != Linux ]]; then
    echo 'This script is a Linux/CUDA starting point; use Setup-Environment.ps1 on Windows.' >&2
    exit 1
fi
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
python_bin="${PYTHON_BIN:-python3.12}"
"$python_bin" -c 'import sys; assert sys.version_info[:2] == (3, 12), "Use Python 3.12 for this recipe"'
if [[ -d .venv && ! -x .venv/bin/python ]]; then
    echo 'Existing .venv is not a Linux environment; use a separate checkout.' >&2
    exit 1
fi
export HF_HOME="$project_root/.cache/huggingface"
export PIP_CACHE_DIR="$project_root/.cache/pip"
export HF_HUB_DISABLE_TELEMETRY=1
export DO_NOT_TRACK=1
"$python_bin" -m venv .venv
.venv/bin/python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
.venv/bin/python -m pip install -r requirements-linux.in
.venv/bin/python -m pip check
.venv/bin/python -c 'import torch; assert torch.cuda.is_available(), "CUDA GPU unavailable"; assert torch.cuda.is_bf16_supported(), "BF16 support required"'
.venv/bin/python scripts/check_gpu.py
.venv/bin/python scripts/smoke_training.py
echo 'Setup and smoke checks finished. These checks are not experimental results.'
