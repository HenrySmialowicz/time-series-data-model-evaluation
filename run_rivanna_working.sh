#!/bin/bash
#SBATCH --job-name=expanded_gametime
#SBATCH --output=logs/experiment_%j.out
#SBATCH --error=logs/experiment_%j.err
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80GB
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=gpu
#SBATCH --account=uvasrg_paid

echo "=== GAMETime Experiment Started ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Time: $(date)"

echo "Loading modules..."
module purge

# Load the modules that actually exist on Rivanna
module load gcc/9.2.0 2>/dev/null || module load gcc/11.4.0 2>/dev/null
module load cuda/11.8 2>/dev/null || module load cuda/12.1.1 2>/dev/null
module load miniforge/24.3.0-py3.11

echo "Successfully loaded modules:"
module list

# Check GPU
echo "GPU Status:"
nvidia-smi

# Initialize conda properly
echo "Initializing conda environment..."
eval "$(conda shell.bash hook)"

echo "Available conda environments:"
conda env list

echo "Activating expanded_gametime environment..."
conda activate expanded_gametime

# Verify Python/PyTorch setup
echo "Verifying Python environment:"
python -c "
import torch
print(f'Python location: {__import__('sys').executable}')
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU count: {torch.cuda.device_count()}')
    print(f'GPU name: {torch.cuda.get_device_name(0)}')
else:
    print('CUDA not available - this will cause training to fail')
"

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export CUDA_VISIBLE_DEVICES=0

echo "=== Starting GAMETime Experiment ==="
if [ ! -f "run_expanded_gametime_experiment.sh" ]; then
    echo "ERROR: run_expanded_gametime_experiment.sh not found!"
    ls -la *.sh
    exit 1
fi

chmod +x run_expanded_gametime_experiment.sh
./run_expanded_gametime_experiment.sh

echo "=== Experiment Complete at $(date) ==="
