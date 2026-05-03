#!/bin/bash
#SBATCH --job-name=bino-full
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/full_%j.out
#SBATCH --error=logs/full_%j.err
## Account: edit the line below or override on the command line with
##   sbatch --account=<your_carc_account> job.sl
## (For reproduction by the original authors on USC CARC, use the lab account.)
#SBATCH --account=your_carc_account

# =============================================================================
# job.sl — sbatch job for full dataset runs
#
# Submit with:
#   sbatch --account=<acct> job.sl                              # full dataset, post_2022 split
#   SPLIT=pre_2022 sbatch --account=<acct> job.sl               # pre_2022 split
#   MODE=low-fpr sbatch --account=<acct> job.sl                 # low-fpr threshold
#   OUTPUT=my_results.json sbatch --account=<acct> job.sl       # custom output name
# =============================================================================

# Path to the python interpreter inside your conda env.
# Override on submission with:  BINO_PYTHON=/path/to/python sbatch ... job.sl
PYTHON="${BINO_PYTHON:-$(which python)}"

SPLIT=${SPLIT:-post_2022}
MODE=${MODE:-accuracy}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT=${OUTPUT:-full_${SPLIT}_${MODE}_${TIMESTAMP}_job${SLURM_JOB_ID}.json}

echo "Job ID       : $SLURM_JOB_ID"
echo "Node         : $SLURMD_NODENAME"
echo "Split        : $SPLIT"
echo "Mode         : $MODE"
echo "Output file  : $OUTPUT"
echo "Started at   : $(date)"
echo "----------------------------------------"

$PYTHON main.py \
    --source hf \
    --split "$SPLIT" \
    --mode "$MODE" \
    --output "$OUTPUT"

echo "----------------------------------------"
echo "Finished at  : $(date)"
