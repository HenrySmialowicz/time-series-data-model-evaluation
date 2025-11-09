#!/bin/bash
#SBATCH --job-name=gametime_exp
#SBATCH --output=logs/gametime_%j.out
#SBATCH --error=logs/gametime_%j.err
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=gpu
#SBATCH --account=uvasrg_paid

echo "=== GAMETime Expansion Experiment ==="
echo "Job: $SLURM_JOB_ID"
echo "Start: $(date)"

# Use direct path to your environment's Python
PYTHON_PATH="/home/hzk8dg/.conda/envs/expanded_gametime/bin/python"

echo "Using Python: $PYTHON_PATH"

# Verify it works
$PYTHON_PATH << 'PYTEST'
import sys
print(f"Python: {sys.version}")
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
PYTEST

if [ $? -ne 0 ]; then
    echo "ERROR: Environment verification failed"
    exit 1
fi

echo "✓ Environment verified"

# Setup
TEST_DATA="data/expanded_experiments/nba/test.json"
RESULTS_DIR="results/gametime_expansion/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

echo ""
echo "=== Evaluating Qwen ==="
$PYTHON_PATH scripts/reasoning_numerical_evaluation.py \
    --model "Qwen/Qwen2.5-1.5B-Instruct" \
    --test_data "$TEST_DATA" \
    --output_dir "$RESULTS_DIR" \
    --max_samples 100

$PYTHON_PATH -c "import torch; torch.cuda.empty_cache()"
sleep 5

echo ""
echo "=== Evaluating DeepSeek ==="
$PYTHON_PATH scripts/reasoning_numerical_evaluation.py \
    --model "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" \
    --test_data "$TEST_DATA" \
    --output_dir "$RESULTS_DIR" \
    --max_samples 100

# Create baseline comparisons
echo '{"model": "DistilBERT (fine-tuned)", "evaluation_type": "fine_tuned_numerical_prediction", "mae": 0.0475, "rmse": 0.0675}' > "$RESULTS_DIR/baseline_distilbert.json"
echo '{"model": "BERT (fine-tuned)", "evaluation_type": "fine_tuned_numerical_prediction", "mae": 0.0490, "rmse": 0.0713}' > "$RESULTS_DIR/baseline_bert.json"

# Generate comparison
$PYTHON_PATH << 'PYCOMP'
import json, glob, sys
results_dir = sys.argv[1]
files = glob.glob(f'{results_dir}/*.json')
results = []
for f in files:
    with open(f) as fp:
        results.append(json.load(fp))

valid = [r for r in results if r.get('mae') is not None]
valid.sort(key=lambda x: x['mae'])

print("\n" + "="*70)
print("RESULTS COMPARISON")
print("="*70)
for i, r in enumerate(valid, 1):
    model = r['model'].split('/')[-1] if '/' in r['model'] else r['model']
    print(f"{i}. {model}: MAE = {r['mae']:.4f}")
print("="*70)

with open(f'{results_dir}/comparison_summary.json', 'w') as f:
    json.dump({'results': valid}, f, indent=2)
PYCOMP "$RESULTS_DIR"

echo ""
echo "=== Complete at $(date) ==="
echo "Results saved to: $RESULTS_DIR"
