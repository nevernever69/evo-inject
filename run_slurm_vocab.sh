#!/bin/bash
#SBATCH --job-name=evo-vocab
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

# ── Vocabulary Ablation Experiment ──
#
# Parameterized SLURM script for Option B experiments.
# Controls vocab size and pool composition via env vars.
#
# Environment variables:
#   VOCAB_SIZE  — target pool size (default 5000)
#   POOL_TYPE   — curated|random|interesting|separator|special|int_sep (default curated)
#   GENS        — generations (default 100)
#   RUN_SEED    — random seed for reproducibility (default 42)
#
# Usage:
#   sbatch run_slurm_vocab.sh                                          # defaults
#   VOCAB_SIZE=2000 POOL_TYPE=curated GENS=100 sbatch run_slurm_vocab.sh
#   VOCAB_SIZE=5000 POOL_TYPE=random RUN_SEED=42 sbatch run_slurm_vocab.sh
#   POOL_TYPE=interesting sbatch run_slurm_vocab.sh
#   QUICK=1 sbatch run_slurm_vocab.sh                                  # quick test

# ── Parameters with defaults ──
VOCAB_SIZE="${VOCAB_SIZE:-5000}"
POOL_TYPE="${POOL_TYPE:-curated}"
GENS="${GENS:-80}"
RUN_SEED="${RUN_SEED:-42}"

# ── Project directory ──
PROJECT_DIR="/scratch/pioneer/users/axb2032/DeepQn/evo-inject"
cd "$PROJECT_DIR"
mkdir -p logs

echo "================================================"
echo "  VOCABULARY ABLATION EXPERIMENT"
echo "  Pool: ${POOL_TYPE}-${VOCAB_SIZE} (seed=${RUN_SEED})"
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
CMD="python main_vocab_ablation.py --device cuda --wandb --project evo-inject-compositional"
CMD="$CMD --vocab-size $VOCAB_SIZE --pool $POOL_TYPE --seed $RUN_SEED"

if [ "${QUICK:-0}" = "1" ]; then
    echo "  MODE: Quick test (3 gens, 5 pop, 10 steps)"
    CMD="$CMD --quick"
else
    CMD="$CMD --gens $GENS --pop 20 --steps 20"
    echo "  MODE: Full run ($GENS gens, 20 pop, 20 steps/life)"
fi

echo "  LOSS: Enabled"
echo "  REFINE: Enabled"
echo "  SEED PHRASES: NONE (filler only)"
echo "  POOL TYPE: $POOL_TYPE"
echo "  VOCAB SIZE: $VOCAB_SIZE"
echo "  RNG SEED: $RUN_SEED"
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
