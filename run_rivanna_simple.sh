#!/bin/bash
#SBATCH --job-name=gametime_simple
#SBATCH --output=logs/experiment_%j.out
#SBATCH --error=logs/experiment_%j.err
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=gpu
#SBATCH --account=uvasrg_paid

echo "=== Simple GAMETime Experiment ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start Time: $(date)"

# Load required modules
echo "Loading modules..."
module load miniforge/24.3.0-py3.11

# Initialize conda
echo "Activating conda environment..."
eval "$(conda shell.bash hook)"
conda activate expanded_gametime

# Verify environment
echo "Environment verification:"
python -c "
import torch
print('Python version check')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
"

# Show GPU status
echo ""
echo "GPU Status:"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv

# Check data availability
echo ""
echo "Data Status:"
if [ -d "data/expanded_experiments/nba" ]; then
    echo "✓ Training data found"
    ls -lh data/expanded_experiments/nba/*.json
else
    echo "✗ Training data missing"
    exit 1
fi

# Run the experiment
echo ""
echo "=== Starting Experiment ==="
./run_gametime_simple.sh

echo ""
echo "=== Job Complete at $(date) ==="
