#!/bin/bash
#SBATCH --job-name=evo-noseed
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --constraint=gpul40s
#SBATCH --time=23:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=/scratch/pioneer/users/axb2032/DeepQn/evo-inject/logs/%j.out
#SBATCH --error=/scratch/pioneer/users/axb2032/DeepQn/evo-inject/logs/%j.err
#SBATCH --mail-user=axb2032@case.edu
#SBATCH --mail-type=ALL

# ── Zero-Seed Ablation: Can evolution discover attacks from scratch? ──
#
# Same architecture as v3, but phrase library has ZERO attack seeds.
# Only neutral fillers ("Please", "Now", "Then", etc.)
# All attack strategies must emerge through token block evolution
# + loss-guided refinement + MAP-Elites diversity pressure.
#
# This is THE key experiment for the NeurIPS paper.
#
# Usage:
#   sbatch run_slurm_noseed.sh                  # Full run
#   QUICK=1 sbatch run_slurm_noseed.sh          # Quick test

# ── Project directory ──
PROJECT_DIR="/scratch/pioneer/users/axb2032/DeepQn/evo-inject"
cd "$PROJECT_DIR"
mkdir -p logs

echo "================================================"
echo "  ZERO-SEED ABLATION EXPERIMENT"
echo "  NO attack seeds — discovery from scratch"
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
CMD="python main_noseed.py --device cuda --wandb --project evo-inject-compositional"

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
