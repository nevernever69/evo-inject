#!/bin/bash
#SBATCH --job-name=evo-inject-v3
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --constraint=gpul40s
#SBATCH --time=48:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=/scratch/pioneer/users/axb2032/DeepQn/evo-inject/logs/%j.out
#SBATCH --error=/scratch/pioneer/users/axb2032/DeepQn/evo-inject/logs/%j.err
#SBATCH --mail-user=axb2032@case.edu
#SBATCH --mail-type=ALL

# ── Compositional Evolutionary Attack Search v3 ──
# Two-level architecture:
#   Level 1: GP evolves attack structure (phrases + token blocks)
#   Level 2: Loss-guided hill climbing refines token blocks
#   MAP-Elites archive maintains diverse attack portfolio
#
# Prerequisites: run setup.sh first (creates venv, installs deps, wandb login)
#
# Usage:
#   sbatch run_slurm.sh                  # Full run (300 gens, 20 pop)
#   QUICK=1 sbatch run_slurm.sh          # Quick test (3 gens, 5 pop)
#   NO_LOSS=1 sbatch run_slurm.sh        # Disable loss computation (faster)
#   NO_REFINE=1 sbatch run_slurm.sh      # Disable token block refinement

# ── Project directory ──
PROJECT_DIR="/scratch/pioneer/users/axb2032/DeepQn/evo-inject"
cd "$PROJECT_DIR"
mkdir -p logs

echo "================================================"
echo "  EVO-INJECT v3 - Compositional Evolutionary Search"
echo "  Job ID:  $SLURM_JOB_ID"
echo "  Node:    $SLURM_NODELIST"
echo "  Time:    $(date)"
echo "  Wall:    48 hours"
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
print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
"
echo ""

# ── Wandb ──
echo "  wandb: $(python -c 'import wandb; print(wandb.__version__)' 2>/dev/null || echo 'not installed')"
echo ""

# ── Build run command ──
CMD="python main.py --device cuda --wandb --project evo-inject-compositional"

if [ "${QUICK:-0}" = "1" ]; then
    echo "  MODE: Quick test (3 gens, 5 pop, 10 steps)"
    CMD="$CMD --quick"
else
    CMD="$CMD --gens 300 --pop 20 --steps 20"
    echo "  MODE: Full run (300 gens, 20 pop, 20 steps/life)"
fi

if [ "${NO_LOSS:-0}" = "1" ]; then
    echo "  LOSS: Disabled (faster, embedding-only signal)"
    CMD="$CMD --no-loss"
else
    echo "  LOSS: Enabled (cross-entropy on target tokens)"
fi

if [ "${NO_REFINE:-0}" = "1" ]; then
    echo "  REFINE: Disabled (no hill climbing on token blocks)"
    CMD="$CMD --no-refine"
else
    echo "  REFINE: Enabled (loss-guided token block refinement)"
fi

echo ""
echo "  Command: $CMD"
echo "================================================"
echo ""

# ── Run with SIGTERM handling for checkpointing ──
# SLURM sends SIGTERM before killing; python catches KeyboardInterrupt
# and saves a final checkpoint + report
trap 'echo "SIGTERM received, sending SIGINT to python..."; kill -INT $PID; wait $PID' TERM

$CMD 2>&1 &
PID=$!
wait $PID

EXIT_CODE=$?
echo ""
echo "================================================"
echo "  Job finished at $(date)"
echo "  Exit code: $EXIT_CODE"
echo "================================================"

exit $EXIT_CODE
