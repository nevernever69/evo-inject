#!/bin/bash
#SBATCH --job-name=evo-const
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --constraint=gpu4090
#SBATCH --time=23:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --output=/scratch/pioneer/users/axb2032/DeepQn/evo-inject/logs/%j.out
#SBATCH --error=/scratch/pioneer/users/axb2032/DeepQn/evo-inject/logs/%j.err
#SBATCH --mail-user=axb2032@case.edu
#SBATCH --mail-type=ALL

# ── Constrained Vocab Zero-Seed: The Key Experiment ──
#
# No attack seeds + token search LIMITED to ~5K interesting/separator/special
# tokens instead of the full 128K vocabulary.
#
# Hypothesis: loss-guided evolutionary search discovers adversarial tokens
# without gradients when the search space is appropriately constrained.
#
# Three-way comparison:
#   1. main.py             → seeded phrases, full vocab
#   2. main_noseed.py      → no seeds, full vocab (blind search)
#   3. main_constrained.py → no seeds, constrained vocab (THIS)
#
# Targeting 4090 nodes (24GB VRAM, sufficient for Llama 3 8B)
#
# Usage:
#   sbatch run_slurm_constrained.sh                  # Full run
#   QUICK=1 sbatch run_slurm_constrained.sh          # Quick test

# ── Project directory ──
PROJECT_DIR="/scratch/pioneer/users/axb2032/DeepQn/evo-inject"
cd "$PROJECT_DIR"
mkdir -p logs

echo "================================================"
echo "  CONSTRAINED VOCAB ZERO-SEED EXPERIMENT"
echo "  NO seeds + ~5K token search space"
echo "  Job ID:  $SLURM_JOB_ID"
echo "  Node:    $SLURM_NODELIST"
echo "  Time:    $(date)"
echo "  Wall:    23 hours"
echo "================================================"

# ── Load modules (Case Western HPC) ──
module purge
module load GCC/12.3.0
module load CUDA/12.1
module load cuDNN/8.9.2.26-CUDA-12.1.1
module load Python/3.11.3-GCCcore-12.3.0

export PYTHONNOUSERSITE=1
source "$PROJECT_DIR/venv/bin/activate"

echo "  Python:  $(python --version)"
echo "  Torch:   $(python -c 'import torch; print(torch.__version__)' 2>/dev/null)"
echo "  GPU:     $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo ""

# ── Build run command ──
CMD="python main_constrained.py --device cuda --wandb --project evo-inject-compositional"

if [ "${QUICK:-0}" = "1" ]; then
    echo "  MODE: Quick test (3 gens, 5 pop, 10 steps)"
    CMD="$CMD --quick"
else
    CMD="$CMD --gens 300 --pop 20 --steps 20"
    echo "  MODE: Full run (300 gens, 20 pop, 20 steps/life)"
fi

echo "  LOSS: Enabled"
echo "  REFINE: Enabled"
echo "  SEED PHRASES: NONE (filler only)"
echo "  TOKEN VOCAB: CONSTRAINED (~5K interesting/separator/special)"
echo ""
echo "  Command: $CMD"
echo "================================================"
echo ""

# ── Run with SIGTERM handling ──
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
