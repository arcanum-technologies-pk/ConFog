#!/usr/bin/env bash
# Creates .venv with the pinned packages (PyTorch with CUDA 12.4 for the RTX 3080) and the blockchain tools.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-$(command -v python3.11 || command -v python3.12 || command -v python3)}
echo "[setup] using $($PY --version)"
$PY -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install --extra-index-url https://download.pytorch.org/whl/cu124 -r requirements.txt
.venv/bin/python -c "import torch; print('[setup] torch', torch.__version__, '| CUDA available:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no GPU')"

if command -v npm >/dev/null; then
  (cd blockchain && npm install --no-audit --no-fund && node compile.js)
else
  echo "[setup] Node.js >= 18 not found: install it (sudo apt install nodejs npm) for the blockchain stage"
fi
command -v docker >/dev/null || echo "[setup] Docker not found: the bench stage will pin fog nodes to single CPU cores instead"
echo "[setup] done. Activate with: source .venv/bin/activate"
