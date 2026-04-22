#!/bin/bash

SOURCE=${SOURCE:-hf}
DATASET=${DATASET:-validname/reddit-ai-detection-english-80k}
SPLIT=${SPLIT:-pre_2022}
FILE=${FILE:-sample_reddit_pre_2022.csv}
MODEL_NAME=${MODEL_NAME:-meta-llama/Llama-3.1-8B-Instruct}
CONDA_ENV=${CONDA_ENV:-reddit-ai-gen}
EXTRA_ARGS="${@}"

if [[ "$EXTRA_ARGS" != *"--sample"* ]]; then
    EXTRA_ARGS="--sample 8 $EXTRA_ARGS"
fi

salloc \
    --partition=gpu \
    --gres=gpu:a40:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=01:00:00 \
    --account=snazaria_1817 \
    bash -c "
        module purge
        module load conda
        conda run -n $CONDA_ENV python $(pwd)/main.py \
            --source $SOURCE \
            --dataset $DATASET \
            --split $SPLIT \
            --file $FILE \
            --model $MODEL_NAME \
            $EXTRA_ARGS
    "
