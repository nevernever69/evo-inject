#!/bin/bash
# ── Interactive GPU session for evo-inject ──
# Requests an L40S GPU node and drops you into a shell
#
# Usage:
#   bash interactive.sh          # Get interactive GPU shell
#   bash interactive.sh cpu      # Get interactive CPU shell (for setup only)

NODE="${1:-gpu}"
USERID="${USER}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Requesting interactive $NODE node..."

if [ "$NODE" == "gpu" ]; then
    srun -p gpu \
        -C "gpul40s" \
        --gres=gpu:1 \
        --time=04:00:00 \
        --mem=64G \
        --cpus-per-task=8 \
        --mail-user=${USERID}@case.edu \
        --mail-type=ALL \
        --pty bash
elif [ "$NODE" == "cpu" ]; then
    srun --time=02:00:00 \
        --mem=64G \
        --cpus-per-task=8 \
        --mail-user=${USERID}@case.edu \
        --mail-type=ALL \
        --pty bash
else
    echo "Unknown node type: $NODE. Use 'gpu' or 'cpu'."
    exit 1
fi
