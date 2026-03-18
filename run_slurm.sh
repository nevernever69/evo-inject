#!/bin/bash
#SBATCH --job-name=evo-inject
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --constraint=gpul40s
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=/scratch/pioneer/users/axb2032/DeepQn/evo-inject/logs/%j.out
#SBATCH --error=/scratch/pioneer/users/axb2032/DeepQn/evo-inject/logs/%j.err
#SBATCH --mail-user=axb2032@case.edu
#SBATCH --mail-type=ALL

# ── GP-Evolving Prompt Injection Fuzzer ──
# Prerequisites: run setup.sh first (creates venv, installs deps, wandb login)
#
# Usage:
#   sbatch run_slurm.sh                  # Full run (200 gens, 30 pop)
#   QUICK=1 sbatch run_slurm.sh          # Quick test (3 gens, 5 pop)

# ── Project directory ──
PROJECT_DIR="/scratch/pioneer/users/axb2032/DeepQn/evo-inject"
cd "$PROJECT_DIR"
mkdir -p logs

echo "================================================"
echo "  EVO-INJECT - GP Prompt Injection Fuzzer"
echo "  Job ID:  $SLURM_JOB_ID"
echo "  Node:    $SLURM_NODELIST"
echo "  Time:    $(date)"
echo "================================================"

# ── Load modules (Case Western HPC) ──
module purge
module load GCC/12.3.0
module load CUDA/12.1
module load cuDNN/8.9.2.26-CUDA-12.1.1
module load Python/3.11.3-GCCcore-12.3.0

# Ignore ~/.local user-site packages
export PYTHONNOUSERSITE=1

# ── Activate venv ──
source "$PROJECT_DIR/venv/bin/activate"

echo "  Python:  $(python --version)"
echo "  Torch:   $(python -c 'import torch; print(torch.__version__)' 2>/dev/null)"
echo "  GPU:     $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo ""

# ── Verify CUDA ──
python -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available!'
print(f'CUDA device: {torch.cuda.get_device_name(0)}')
print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"
echo ""

# ── Wandb ──
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

exit $EXIT_CODE
