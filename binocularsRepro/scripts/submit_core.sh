#!/bin/bash

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UPSTREAM_DIR="${UPSTREAM_DIR:-$PROJECT_DIR/.upstream/Binoculars}"

sbatch "$PROJECT_DIR/job.sl"

DATASET_PATH="$UPSTREAM_DIR/datasets/core/cnn/cnn-llama2_13.jsonl" \
DATASET_NAME="CNN" \
HUMAN_SAMPLE_KEY="article" \
MACHINE_SAMPLE_KEY="meta-llama-Llama-2-13b-hf_generated_text_wo_prompt" \
MACHINE_TEXT_SOURCE="LLaMA-2-13B" \
sbatch "$PROJECT_DIR/job.sl"

DATASET_PATH="$UPSTREAM_DIR/datasets/core/pubmed/pubmed-llama2_13.jsonl" \
DATASET_NAME="PubMed" \
HUMAN_SAMPLE_KEY="article" \
MACHINE_SAMPLE_KEY="meta-llama-Llama-2-13b-hf_generated_text_wo_prompt" \
MACHINE_TEXT_SOURCE="LLaMA-2-13B" \
sbatch "$PROJECT_DIR/job.sl"
