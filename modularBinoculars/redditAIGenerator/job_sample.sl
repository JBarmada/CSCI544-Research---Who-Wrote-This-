#!/bin/bash
#SBATCH --job-name=reddit-ai-sample
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/reddit_ai_sample_%j.out
#SBATCH --error=logs/reddit_ai_sample_%j.err
#SBATCH --account=your_carc_account

# =============================================================================
# job_sample.sl — sbatch job for sample Reddit AI generation runs
#
# Submit with:
#   sbatch job_sample.sl
#   SAMPLE=8 sbatch job_sample.sl
#   SAMPLE=32 SOURCE=local FILE=sample_reddit_pre_2022.csv sbatch job_sample.sl
# =============================================================================

PYTHON=${PYTHON:-$(which python)}

SOURCE=${SOURCE:-hf}
DATASET=${DATASET:-validname/reddit-ai-detection-english-80k}
SPLIT=${SPLIT:-pre_2022}
FILE=${FILE:-sample_reddit_pre_2022.csv}
SAMPLE=${SAMPLE:-8}
MODEL_NAME=${MODEL_NAME:-meta-llama/Llama-3.1-8B-Instruct}
OUTPUT_PREFIX=${OUTPUT_PREFIX:-reddit_ai_sample}
CHECKPOINT_EVERY=${CHECKPOINT_EVERY:-5}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_PREFIX=${OUTPUT_PREFIX}_${SPLIT}_n${SAMPLE}_${TIMESTAMP}_job${SLURM_JOB_ID}

echo "Job ID         : $SLURM_JOB_ID"
echo "Node           : $SLURMD_NODENAME"
echo "Python         : $PYTHON"
echo "Source         : $SOURCE"
echo "Split          : $SPLIT"
echo "Dataset        : $DATASET"
echo "File           : $FILE"
echo "Sample size    : $SAMPLE"
echo "Model          : $MODEL_NAME"
echo "Output prefix  : $OUTPUT_PREFIX"
echo "Checkpoint n   : $CHECKPOINT_EVERY"
echo "Started at     : $(date)"
echo "----------------------------------------"

$PYTHON main.py \
    --source "$SOURCE" \
    --dataset "$DATASET" \
    --split "$SPLIT" \
    --file "$FILE" \
    --sample "$SAMPLE" \
    --model "$MODEL_NAME" \
    --output-prefix "$OUTPUT_PREFIX" \
    --checkpoint-every "$CHECKPOINT_EVERY"

echo "----------------------------------------"
echo "Finished at    : $(date)"
