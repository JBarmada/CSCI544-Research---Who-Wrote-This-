#!/bin/bash

set -euo pipefail

PINNED_COMMIT="c8ae2f90d50ee696418bc71d8d9e5020e5f9d7b8"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UPSTREAM_DIR="${UPSTREAM_DIR:-$PROJECT_DIR/.upstream/Binoculars}"
ENV_NAME="${ENV_NAME:-binoculars}"

module purge
module load conda
module load git

mkdir -p "$(dirname "$UPSTREAM_DIR")"

if [[ ! -d "$UPSTREAM_DIR/.git" ]]; then
    git clone https://github.com/ahans30/Binoculars.git "$UPSTREAM_DIR"
fi

git -C "$UPSTREAM_DIR" fetch --all --tags
git -C "$UPSTREAM_DIR" checkout "$PINNED_COMMIT"

source "$(conda info --base)/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    conda create -n "$ENV_NAME" python=3.9 -y
fi

conda activate "$ENV_NAME"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
unset PYTHONPATH
unset PYTHONHOME

python -m pip install --upgrade pip
python -m pip install --upgrade "numpy<2"
python -m pip install -e "$UPSTREAM_DIR"
python -m pip install -r "$PROJECT_DIR/requirements.txt"

echo
echo "Setup complete."
echo "Project dir : $PROJECT_DIR"
echo "Upstream dir: $UPSTREAM_DIR"
echo "Pinned commit: $PINNED_COMMIT"
echo "Conda env   : $ENV_NAME"
