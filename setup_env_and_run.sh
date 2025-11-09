#!/bin/bash

# Ensure conda is initialized
if [ -f ~/.bashrc ]; then
    source ~/.bashrc
fi

# Initialize conda for this shell
CONDA_BASE=$(conda info --base 2>/dev/null)
if [ -z "$CONDA_BASE" ]; then
    # Try to find conda from module
    module load miniforge/24.3.0-py3.11
    CONDA_BASE=$(conda info --base)
fi

source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate expanded_gametime

# Verify
python -c "import torch; print('PyTorch OK')" || exit 1

# Run the actual evaluation
python "$@"
