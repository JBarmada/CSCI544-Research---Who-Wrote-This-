#!/bin/bash
#SBATCH --job-name=reddit-ai-gen
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/reddit_ai_gen_%j.out
#SBATCH --error=logs/reddit_ai_gen_%j.err
#SBATCH --account=snazaria_1817

SOURCE=${SOURCE:-hf}
DATASET=${DATASET:-validname/reddit-ai-detection-english-80k}
SPLIT=${SPLIT:-pre_2022}
FILE=${FILE:-sample_reddit_pre_2022.csv}
SAMPLE=${SAMPLE:-}
MODEL_NAME=${MODEL_NAME:-meta-llama/Llama-3.1-8B-Instruct}
CONDA_ENV=${CONDA_ENV:-reddit-ai-gen}
OUTPUT_PREFIX=${OUTPUT_PREFIX:-reddit_ai_generated}

echo "Job ID        : $SLURM_JOB_ID"
echo "Node          : $SLURMD_NODENAME"
echo "Source        : $SOURCE"
echo "Split         : $SPLIT"
echo "Dataset       : $DATASET"
echo "File          : $FILE"
echo "Model         : $MODEL_NAME"
echo "Conda env     : $CONDA_ENV"
echo "Started at    : $(date)"
echo "----------------------------------------"

EXTRA_ARGS=""
if [[ -n "$SAMPLE" ]]; then
    EXTRA_ARGS="$EXTRA_ARGS --sample $SAMPLE"
fi

module purge
module load conda

conda run -n "$CONDA_ENV" python main.py \
    --source "$SOURCE" \
    --dataset "$DATASET" \
    --split "$SPLIT" \
    --file "$FILE" \
    --model "$MODEL_NAME" \
    --output-prefix "$OUTPUT_PREFIX" \
    $EXTRA_ARGS

echo "----------------------------------------"
echo "Finished at   : $(date)"
