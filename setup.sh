#!/bin/bash
# Setup script — run once before first SLURM job
# Usage: bash setup.sh

set -e

echo "Setting up parasite environment..."

# Load modules
module purge
module load cuda/12.1
module load python/3.11  # Adjust to your cluster

# Create venv
VENV_DIR="$HOME/.venvs/parasite"
echo "Creating venv at $VENV_DIR"
python -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

pip install --upgrade pip

# PyTorch with CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# HuggingFace + embeddings
pip install transformers accelerate sentence-transformers

# Core
pip install numpy

# Optional
pip install wandb

echo ""
echo "Verifying..."
python -c "
import torch
print(f'PyTorch {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')

import transformers
print(f'Transformers {transformers.__version__}')

import sentence_transformers
print(f'Sentence-transformers {sentence_transformers.__version__}')

print('All dependencies OK')
"

echo ""
echo "Pre-downloading models (saves time on first run)..."
python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
print('Downloading Llama 3 8B tokenizer...')
AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B-Instruct')
print('Downloading Llama 3 8B model...')
AutoModelForCausalLM.from_pretrained('meta-llama/Meta-Llama-3-8B-Instruct', torch_dtype='auto')
print('Llama 3 8B cached.')

from sentence_transformers import SentenceTransformer
print('Downloading embedding model...')
SentenceTransformer('all-MiniLM-L6-v2')
print('Embedding model cached.')
"

echo ""
echo "Setup complete. Run with: sbatch run_slurm.sh"
