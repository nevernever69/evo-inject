#!/bin/bash
#SBATCH --job-name=evo-inject
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --constraint=gpul40s
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --mail-type=ALL

# ── GP-Evolving Prompt Injection Fuzzer ──
# Runs on HPC with L40S GPU (48GB VRAM)
# Llama 3 8B (~16GB in fp16) + sentence-transformers (~80MB)
#
# Prerequisites: run setup.sh first (creates venv, installs deps, wandb login)
#
# Usage:
#   sbatch run_slurm.sh                       # Default (200 gens, 30 pop)
#   sbatch run_slurm.sh --export=QUICK=1      # Quick test (3 gens, 5 pop)
#   sbatch run_slurm.sh --export=NO_MUTATOR=1 # Raw GP only, no LLM mutator

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$SCRIPT_DIR/venv"

echo "================================================"
echo "  EVO-INJECT - GP Prompt Injection Fuzzer"
echo "  Job ID:  $SLURM_JOB_ID"
echo "  Node:    $SLURM_NODELIST"
echo "  Time:    $(date)"
echo "================================================"

# ── Load modules ──
module purge
module load cuda/12.1
module load python/3.11

# ── Activate venv ──
if [ ! -d "$VENV_DIR" ]; then
    echo "ERROR: venv not found at $VENV_DIR"
    echo "Run setup.sh first: bash setup.sh"
    exit 1
fi
source "$VENV_DIR/bin/activate"

echo "  Python:  $(python --version)"
echo "  Torch:   $(python -c 'import torch; print(torch.__version__)' 2>/dev/null)"
echo "  GPU:     $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo ""

# ── Verify CUDA ──
python -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available!'
print(f'CUDA device: {torch.cuda.get_device_name(0)}')
print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
"
echo ""

# ── Create logs dir ──
mkdir -p "$SCRIPT_DIR/logs"
cd "$SCRIPT_DIR"

# ── Wandb ──
# wandb credentials persist from setup.sh login
# Set offline mode if no internet on compute nodes:
# export WANDB_MODE=offline
echo "  wandb: $(python -c 'import wandb; print(wandb.__version__)' 2>/dev/null || echo 'not installed')"
echo ""

# ── Build run command ──
CMD="python main.py --device cuda --wandb --project evo-inject"

if [ "${QUICK:-0}" = "1" ]; then
    echo "  MODE: Quick test (3 gens, 5 pop, 10 steps)"
    CMD="$CMD --quick"
else
    CMD="$CMD --gens 200 --pop 30 --steps 30"
    echo "  MODE: Full run (200 gens, 30 pop, 30 steps)"
fi

if [ "${NO_MUTATOR:-0}" = "1" ]; then
    echo "  MUTATOR: Disabled (raw GP only)"
    CMD="$CMD --no-mutator"
else
    echo "  MUTATOR: Enabled (LLM-as-mutator)"
fi

echo ""
echo "  Command: $CMD"
echo "================================================"
echo ""

# ── Run ──
$CMD 2>&1

EXIT_CODE=$?
echo ""
echo "================================================"
echo "  Job finished at $(date)"
echo "  Exit code: $EXIT_CODE"
echo "================================================"

# ── Sync wandb if offline ──
if [ "${WANDB_MODE}" = "offline" ]; then
    echo "Syncing wandb offline runs..."
    wandb sync logs/wandb/
fi

exit $EXIT_CODE
