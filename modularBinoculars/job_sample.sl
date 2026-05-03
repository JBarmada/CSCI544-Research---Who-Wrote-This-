#!/bin/bash
#SBATCH --job-name=bino-sample
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/sample_%j.out
#SBATCH --error=logs/sample_%j.err
## Account: edit the line below or override on the command line with
##   sbatch --account=<your_carc_account> job_sample.sl
## (For reproduction by the original authors on USC CARC, use the lab account.)
#SBATCH --account=your_carc_account

# =============================================================================
# job_sample.sl — sbatch job for sample runs
#
# Submit with:
#   sbatch --account=<acct> job_sample.sl                          # 100-row sample (default)
#   SAMPLE=500 sbatch --account=<acct> job_sample.sl               # 500-row sample
#   SAMPLE=100 SPLIT=pre_2022 sbatch --account=<acct> job_sample.sl
#   SAMPLE=1000 MODE=low-fpr sbatch --account=<acct> job_sample.sl
# =============================================================================

# Path to the python interpreter inside your conda env.
# Override on submission with:  BINO_PYTHON=/path/to/python sbatch ... job_sample.sl
PYTHON="${BINO_PYTHON:-$(which python)}"

# Configurable via env vars; fall back to defaults
SAMPLE=${SAMPLE:-100}
SPLIT=${SPLIT:-post_2022}
MODE=${MODE:-accuracy}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT=${OUTPUT:-sample_${SPLIT}_n${SAMPLE}_${MODE}_${TIMESTAMP}_job${SLURM_JOB_ID}.json}

echo "Job ID       : $SLURM_JOB_ID"
echo "Node         : $SLURMD_NODENAME"
echo "Sample size  : $SAMPLE"
echo "Split        : $SPLIT"
echo "Mode         : $MODE"
echo "Output file  : $OUTPUT"
echo "Started at   : $(date)"
echo "----------------------------------------"

$PYTHON main.py \
    --source hf \
    --split "$SPLIT" \
    --sample "$SAMPLE" \
    --mode "$MODE" \
    --output "$OUTPUT"

echo "----------------------------------------"
echo "Finished at  : $(date)"
