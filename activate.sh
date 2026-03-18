#!/bin/bash
# ── Quick activate — load modules + venv ──
# MUST be sourced, not executed:  source activate.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/venv"

if [ ! -d "$VENV_DIR" ]; then
    echo "ERROR: venv not found at $VENV_DIR"
    echo "Run setup.sh first: bash setup.sh"
    return 1 2>/dev/null || exit 1
fi

# Load modules FIRST (provides libffi, libssl, etc.)
module purge
module load GCC/12.3.0
module load CUDA/12.1
module load cuDNN/8.9.2.26-CUDA-12.1.1
module load Python/3.11.3-GCCcore-12.3.0

export PYTHONNOUSERSITE=1

# Activate venv AFTER modules are loaded
source "$VENV_DIR/bin/activate"

echo "Ready. GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"
echo ""
echo "  Quick test:  python main.py --quick --device cuda --wandb --project evo-inject"
echo "  Full run:    python main.py --gens 200 --pop 30 --steps 30 --device cuda --wandb"
