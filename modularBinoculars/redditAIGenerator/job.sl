#!/bin/bash
#SBATCH --job-name=reddit-ai-gen
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/reddit_ai_gen_%j.out
#SBATCH --error=logs/reddit_ai_gen_%j.err
#SBATCH --account=your_carc_account

# =============================================================================
# job.sl — sbatch job for full Reddit AI generation runs
#
# Submit with:
#   sbatch job.sl
#   SAMPLE=128 sbatch job.sl
#   SOURCE=local FILE=sample_reddit_pre_2022.csv sbatch job.sl
#   MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct sbatch job.sl
# =============================================================================

PYTHON=${PYTHON:-$(which python)}

SOURCE=${SOURCE:-hf}
DATASET=${DATASET:-validname/reddit-ai-detection-english-80k}
SPLIT=${SPLIT:-pre_2022}
FILE=${FILE:-sample_reddit_pre_2022.csv}
SAMPLE=${SAMPLE:-}
MODEL_NAME=${MODEL_NAME:-meta-llama/Llama-3.1-8B-Instruct}
OUTPUT_PREFIX=${OUTPUT_PREFIX:-reddit_ai_generated}
CHECKPOINT_EVERY=${CHECKPOINT_EVERY:-10}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_PREFIX=${OUTPUT_PREFIX}_${SPLIT}_${TIMESTAMP}_job${SLURM_JOB_ID}

echo "Job ID         : $SLURM_JOB_ID"
echo "Node           : $SLURMD_NODENAME"
echo "Python         : $PYTHON"
echo "Source         : $SOURCE"
echo "Split          : $SPLIT"
echo "Dataset        : $DATASET"
echo "File           : $FILE"
echo "Sample size    : ${SAMPLE:-full}"
echo "Model          : $MODEL_NAME"
echo "Output prefix  : $OUTPUT_PREFIX"
echo "Checkpoint n   : $CHECKPOINT_EVERY"
echo "Started at     : $(date)"
echo "----------------------------------------"

EXTRA_ARGS=""
if [[ -n "$SAMPLE" ]]; then
    EXTRA_ARGS="$EXTRA_ARGS --sample $SAMPLE"
fi
if [[ -n "$CHECKPOINT_EVERY" ]]; then
    EXTRA_ARGS="$EXTRA_ARGS --checkpoint-every $CHECKPOINT_EVERY"
fi

$PYTHON main.py \
    --source "$SOURCE" \
    --dataset "$DATASET" \
    --split "$SPLIT" \
    --file "$FILE" \
    --model "$MODEL_NAME" \
    --output-prefix "$OUTPUT_PREFIX" \
    $EXTRA_ARGS

echo "----------------------------------------"
echo "Finished at    : $(date)"
