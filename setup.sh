#!/bin/bash
# ── One-time setup — run BEFORE first SLURM job ──
# Creates venv, installs all deps, logs into wandb, downloads models
#
# Usage:
#   bash setup.sh              # Full setup
#   bash setup.sh --no-model   # Skip model download (if already cached)
#
# Run this on a login node or interactive session with GPU access

set -e

SKIP_MODEL=false
if [ "$1" = "--no-model" ]; then
    SKIP_MODEL=true
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$SCRIPT_DIR/venv"

echo "================================================"
echo "  EVO-INJECT SETUP"
echo "================================================"
echo ""

# ── Load modules ──
echo "[1/6] Loading modules..."
module purge
module load cuda/12.1
module load python/3.11  # Adjust to your cluster's available version
echo "  CUDA: $(nvcc --version 2>/dev/null | grep release | awk '{print $6}' || echo 'module loaded')"
echo "  Python: $(python3 --version)"
echo ""

# ── Create venv ──
echo "[2/6] Creating virtual environment at $VENV_DIR ..."
if [ -d "$VENV_DIR" ]; then
    echo "  Existing venv found. Removing and recreating..."
    rm -rf "$VENV_DIR"
fi
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
echo "  venv activated: $(which python)"
echo ""

# ── Install dependencies ──
echo "[3/6] Installing dependencies..."
pip install --upgrade pip setuptools wheel -q

echo "  Installing PyTorch (CUDA 12.1)..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 -q

echo "  Installing HuggingFace + embeddings..."
pip install transformers accelerate sentence-transformers -q

echo "  Installing numpy..."
pip install numpy -q

echo "  Installing wandb..."
pip install wandb -q

echo ""
pip list | grep -E "torch|transformers|sentence|wandb|numpy|accelerate"
echo ""

# ── Verify imports ──
echo "[4/6] Verifying installation..."
python -c "
import torch
print(f'  PyTorch {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')

import transformers
print(f'  Transformers {transformers.__version__}')

import sentence_transformers
print(f'  Sentence-transformers {sentence_transformers.__version__}')

import wandb
print(f'  Wandb {wandb.__version__}')

import numpy
print(f'  Numpy {numpy.__version__}')

print('  All imports OK')
"
echo ""

# ── Wandb login ──
echo "[5/6] Setting up Weights & Biases..."
echo "  If you haven't logged in before, paste your API key from:"
echo "  https://wandb.ai/authorize"
echo ""
wandb login
echo ""

# ── Download models ──
if [ "$SKIP_MODEL" = false ]; then
    echo "[6/6] Pre-downloading models (saves GPU job time)..."
    echo "  This downloads ~16GB for Llama 3 8B + ~80MB for embeddings."
    echo "  Make sure you have accepted the Llama 3 license on HuggingFace:"
    echo "  https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct"
    echo ""

    python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM

print('  Downloading Llama 3 8B tokenizer...')
AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B-Instruct')

print('  Downloading Llama 3 8B model weights (~16GB)...')
AutoModelForCausalLM.from_pretrained(
    'meta-llama/Meta-Llama-3-8B-Instruct',
    torch_dtype='auto',
)
print('  Llama 3 8B cached.')

from sentence_transformers import SentenceTransformer
print('  Downloading embedding model (all-MiniLM-L6-v2)...')
SentenceTransformer('all-MiniLM-L6-v2')
print('  Embedding model cached.')
"
    echo ""
else
    echo "[6/6] Skipping model download (--no-model flag set)"
    echo ""
fi

# ── Create logs dir ──
mkdir -p "$SCRIPT_DIR/logs"

echo "================================================"
echo "  SETUP COMPLETE"
echo "================================================"
echo ""
echo "  Venv:     $VENV_DIR"
echo "  Activate: source $VENV_DIR/bin/activate"
echo ""
echo "  To run:"
echo "    sbatch run_slurm.sh                    # Default (200 gens, 30 pop)"
echo "    sbatch run_slurm.sh --export=QUICK=1   # Quick test (3 gens, 5 pop)"
echo ""
echo "  To monitor:"
echo "    tail -f logs/slurm_<jobid>.out"
echo "    wandb dashboard: https://wandb.ai"
echo ""
