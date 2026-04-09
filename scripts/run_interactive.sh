#!/bin/bash
# =============================================================================
# run_interactive.sh — salloc interactive GPU session for quick sample tests
#
# Usage:
#   bash scripts/run_interactive.sh                   # defaults: --source hf --sample 100
#   bash scripts/run_interactive.sh --sample 500
#   bash scripts/run_interactive.sh --source local --file my_data.csv --sample 200
#
# All extra arguments are forwarded to main.py.
# =============================================================================

# Forward any extra CLI args to main.py (e.g. --sample 200 --mode low-fpr)
EXTRA_ARGS="${@}"

# Default to 100-row sample if --sample is not in EXTRA_ARGS
if [[ "$EXTRA_ARGS" != *"--sample"* ]]; then
    EXTRA_ARGS="--sample 100 $EXTRA_ARGS"
fi

salloc \
    --partition=gpu \
    --gres=gpu:a100:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=01:00:00 \
    --account=your_project_account \
    bash -c "
        module purge
        module load conda
        conda activate binoculars
        cd $(pwd)
        python main.py --source hf $EXTRA_ARGS
    "
