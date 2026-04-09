#!/bin/bash
# =============================================================================
# setup_env.sh — One-time environment setup on USC CARC (discovery)
#
# Run this ONCE from the project root before submitting any jobs:
#   bash scripts/setup_env.sh
# =============================================================================

set -e

module purge
module load conda

# Create the conda env (Python 3.9 matches the paper's dev environment)
conda create -n binoculars python=3.9 -y
source activate binoculars

# Install the paper's Binoculars library directly from GitHub
# NOTE: the PyPI package named 'binoculars' is unrelated — do NOT use pip install binoculars
pip install git+https://github.com/ahans30/Binoculars.git

# Install all other project dependencies
pip install -r requirements.txt

echo ""
echo "Setup complete. Activate the env with:"
echo "  module load conda && conda activate binoculars"
