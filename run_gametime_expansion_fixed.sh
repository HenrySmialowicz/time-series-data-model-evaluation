#!/bin/bash
#SBATCH --job-name=gametime_expansion
#SBATCH --output=logs/gametime_expansion_%j.out
#SBATCH --error=logs/gametime_expansion_%j.err
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=gpu
#SBATCH --account=uvasrg_paid

echo "========================================================================"
echo "GAMETime Expansion: Reasoning Models vs Fine-tuned Models"
echo "========================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Start: $(date)"

# More robust environment loading
echo "Loading environment..."
module purge
module load miniforge/24.3.0-py3.11

# Activate conda with multiple methods
source ~/.bashrc 2>/dev/null || true
eval "$(conda shell.bash hook)"
conda activate expanded_gametime

# Verify PyTorch is available
echo "Verifying PyTorch..."
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')" || {
    echo "ERROR: PyTorch not available"
    exit 1
}

# Setup
TEST_DATA="data/expanded_experiments/nba/test.json"
RESULTS_DIR="results/gametime_expansion/$(date +%Y%m%d_%H%M%S)_job_${SLURM_JOB_ID}"
mkdir -p "$RESULTS_DIR"

echo "Test Data: $TEST_DATA"
echo "Results: $RESULTS_DIR"

# Evaluate Qwen
echo ""
echo "[1/2] Evaluating Qwen..."
python scripts/reasoning_numerical_evaluation.py \
    --model "Qwen/Qwen2.5-1.5B-Instruct" \
    --test_data "$TEST_DATA" \
    --output_dir "$RESULTS_DIR" \
    --max_samples 100

python -c "import torch; torch.cuda.empty_cache()"
sleep 5

# Evaluate DeepSeek
echo ""
echo "[2/2] Evaluating DeepSeek..."
python scripts/reasoning_numerical_evaluation.py \
    --model "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" \
    --test_data "$TEST_DATA" \
    --output_dir "$RESULTS_DIR" \
    --max_samples 100

# Create baseline files
cat > "$RESULTS_DIR/finetuned_baseline.json" << BASELINE
{
  "model": "DistilBERT-base-uncased (fine-tuned)",
  "evaluation_type": "fine_tuned_numerical_prediction",
  "mae": 0.0475,
  "rmse": 0.0675
}
BASELINE

cat > "$RESULTS_DIR/finetuned_bert.json" << BASELINE2
{
  "model": "BERT-base-uncased (fine-tuned)",
  "evaluation_type": "fine_tuned_numerical_prediction",
  "mae": 0.0490,
  "rmse": 0.0713
}
BASELINE2

# Generate comparison
python << 'PYEND'
import json, glob
results_dir = "$RESULTS_DIR"
files = glob.glob(f'{results_dir}/*.json')
results = []
for f in files:
    with open(f) as fp:
        results.append(json.load(fp))
valid = [r for r in results if r.get('mae')]
valid.sort(key=lambda x: x['mae'])
print("\n=== RESULTS ===")
for i, r in enumerate(valid, 1):
    print(f"{i}. {r['model']}: MAE {r['mae']:.4f}")
with open(f'{results_dir}/summary.json', 'w') as f:
    json.dump({'results': valid}, f, indent=2)
PYEND

echo ""
echo "Complete at $(date)"
echo "Results: $RESULTS_DIR"
