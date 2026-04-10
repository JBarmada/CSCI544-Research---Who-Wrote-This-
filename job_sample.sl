#!/bin/bash
#SBATCH --job-name=bino-sample
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/sample_%j.out
#SBATCH --error=logs/sample_%j.err
#SBATCH --account=snazaria_1817

# =============================================================================
# job_sample.sl — sbatch job for sample runs
#
# Submit with:
#   sbatch job_sample.sl                          # 100-row sample (default)
#   SAMPLE=500 sbatch job_sample.sl               # 500-row sample
#   SAMPLE=100 SPLIT=pre_2022 sbatch job_sample.sl
#   SAMPLE=1000 MODE=low-fpr sbatch job_sample.sl
# =============================================================================

PYTHON="/home1/barmada/.conda/envs/binoculars/bin/python"

# Configurable via env vars; fall back to defaults
SAMPLE=${SAMPLE:-100}
SPLIT=${SPLIT:-post_2022}
MODE=${MODE:-accuracy}
OUTPUT=${OUTPUT:-sample_results.json}

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
