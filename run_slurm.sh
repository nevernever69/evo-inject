#!/bin/bash
#SBATCH --job-name=parasite-llm
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1

# ── GP-Evolving Prompt Injection Fuzzer ──
# Runs on HPC with L40S GPU (48GB VRAM)
# Llama 3 8B (~16GB in fp16) + sentence-transformers (~80MB)

echo "================================================"
echo "  PARASITE - GP Prompt Injection Fuzzer"
echo "  Job ID: $SLURM_JOB_ID"
echo "  Node:   $SLURM_NODELIST"
echo "  GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "  Time:   $(date)"
echo "================================================"

# ── Load modules ──
module purge
module load cuda/12.1
module load python/3.11  # Adjust to your cluster's available version

# ── Activate venv (create if needed) ──
VENV_DIR="$HOME/.venvs/parasite"
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python -m venv "$VENV_DIR"
    source "$VENV_DIR/bin/activate"
    pip install --upgrade pip
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    pip install transformers accelerate sentence-transformers numpy
    pip install wandb  # optional
else
    source "$VENV_DIR/bin/activate"
fi

echo "Python: $(python --version)"
echo "Torch:  $(python -c 'import torch; print(torch.__version__, "CUDA:", torch.cuda.is_available())')"
echo ""

# ── Create logs dir ──
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$SCRIPT_DIR/logs"
cd "$SCRIPT_DIR"

# ── Cache HuggingFace models on local scratch if available ──
if [ -n "$TMPDIR" ]; then
    export HF_HOME="$TMPDIR/hf_cache"
    mkdir -p "$HF_HOME"
    echo "HF cache: $HF_HOME"
fi

# ── Run ──
# Default: 200 generations, 30 population, 30 steps/lifetime
# With mutator enabled (LLM-as-mutator shares the target model)
# Estimated: ~2-3 hours on L40S with fp16

python main.py \
    --gens 200 \
    --pop 30 \
    --steps 30 \
    --device cuda \
    --wandb \
    --project parasite-llm \
    2>&1

echo ""
echo "Job finished at $(date)"
echo "Exit code: $?"
