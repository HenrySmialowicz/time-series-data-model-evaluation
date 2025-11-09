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
echo "Comparing MAE on ESPN Win Probability Predictions"
echo "========================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start: $(date)"
echo ""

# Load environment
module load miniforge/24.3.0-py3.11
eval "$(conda shell.bash hook)"
conda activate expanded_gametime

# Verify GPU
nvidia-smi --query-gpu=name,memory.total --format=csv

# Setup
TEST_DATA="data/expanded_experiments/nba/test.json"
RESULTS_DIR="results/gametime_expansion/$(date +%Y%m%d_%H%M%S)_job_${SLURM_JOB_ID}"
mkdir -p "$RESULTS_DIR"

echo ""
echo "Test Data: $TEST_DATA"
echo "Results: $RESULTS_DIR"
echo ""

# Test Reasoning Models (Numerical Prediction - No Fine-tuning)
echo "========================================================================"
echo "PART 1: Testing Reasoning Models (Zero-shot Numerical Prediction)"
echo "========================================================================"

echo ""
echo "[1/2] Evaluating Qwen/Qwen2.5-1.5B-Instruct..."
python scripts/reasoning_numerical_evaluation.py \
    --model "Qwen/Qwen2.5-1.5B-Instruct" \
    --test_data "$TEST_DATA" \
    --output_dir "$RESULTS_DIR" \
    --max_samples 100

python -c "import torch; torch.cuda.empty_cache()"
sleep 5

echo ""
echo "[2/2] Evaluating deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B..."
python scripts/reasoning_numerical_evaluation.py \
    --model "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" \
    --test_data "$TEST_DATA" \
    --output_dir "$RESULTS_DIR" \
    --max_samples 100

python -c "import torch; torch.cuda.empty_cache()"

# Compare with existing fine-tuned models
echo ""
echo "========================================================================"
echo "PART 2: Comparing with Fine-tuned Models (for context)"
echo "========================================================================"

echo ""
echo "Loading fine-tuned model results..."
cat > "$RESULTS_DIR/finetuned_baseline.json" << BASELINE
{
  "model": "DistilBERT-base-uncased (fine-tuned, direct mode)",
  "evaluation_type": "fine_tuned_numerical_prediction",
  "mae": 0.0475,
  "rmse": 0.0675,
  "training_samples": 123971,
  "note": "Fine-tuned on NBA data - 2 epochs"
}
BASELINE

cat > "$RESULTS_DIR/finetuned_baseline_bert.json" << BASELINE2
{
  "model": "BERT-base-uncased (fine-tuned, direct mode)",
  "evaluation_type": "fine_tuned_numerical_prediction",
  "mae": 0.0490,
  "rmse": 0.0713,
  "training_samples": 123971,
  "note": "Fine-tuned on NBA data - 2 epochs"
}
BASELINE2

# Generate comprehensive comparison report
echo ""
echo "========================================================================"
echo "PART 3: Generating Comparison Report"
echo "========================================================================"

python << PYTHON
import json
import glob

results_dir = "$RESULTS_DIR"

print("\n" + "="*80)
print("GAMETime EXPANSION: REASONING vs FINE-TUNING COMPARISON")
print("="*80 + "\n")

# Load all results
all_results = []

# Load reasoning model results
reasoning_files = glob.glob(f'{results_dir}/numerical_reasoning_*.json')
for file in reasoning_files:
    with open(file) as f:
        data = json.load(f)
        all_results.append(data)

# Load fine-tuned baselines
baseline_files = glob.glob(f'{results_dir}/finetuned_*.json')
for file in baseline_files:
    with open(file) as f:
        data = json.load(f)
        all_results.append(data)

# Sort by MAE (if available)
valid_results = [r for r in all_results if r.get('mae') is not None]
invalid_results = [r for r in all_results if r.get('mae') is None]

valid_results.sort(key=lambda x: x['mae'])

print("Performance Ranking (by MAE - lower is better):")
print("-"*80)
print(f"{'Rank':<6} {'Model':<45} {'Type':<20} {'MAE':<10}")
print("-"*80)

for i, data in enumerate(valid_results, 1):
    model_name = data['model'].split('/')[-1] if '/' in data['model'] else data['model']
    eval_type = "Fine-tuned" if data.get('evaluation_type') == 'fine_tuned_numerical_prediction' else "Zero-shot Reasoning"
    mae = data.get('mae', 'N/A')
    
    if isinstance(mae, float):
        print(f"{i:<6} {model_name:<45} {eval_type:<20} {mae:.4f}")
    else:
        print(f"{i:<6} {model_name:<45} {eval_type:<20} {mae}")

if invalid_results:
    print("\nModels with failed extraction:")
    for data in invalid_results:
        model_name = data['model'].split('/')[-1] if '/' in data['model'] else data['model']
        print(f"  - {model_name}: Could not extract numerical predictions")

print("\n" + "="*80)
print("DETAILED BREAKDOWN")
print("="*80 + "\n")

for data in valid_results:
    print(f"Model: {data['model']}")
    print(f"Type: {data.get('evaluation_type', 'unknown')}")
    
    if 'reasoning' in data.get('evaluation_type', ''):
        print(f"Extraction Success: {data.get('extraction_success_rate', 0):.1%}")
        print(f"Valid Predictions: {data.get('valid_predictions', 0)}/{data.get('total_samples', 0)}")
    
    mae = data.get('mae')
    rmse = data.get('rmse')
    if mae is not None:
        print(f"MAE: {mae:.4f}")
    if rmse is not None:
        print(f"RMSE: {rmse:.4f}")
    
    if data.get('direction_accuracy'):
        print(f"Direction Accuracy: {data['direction_accuracy']:.1%}")
    
    if data.get('difficulty_breakdown'):
        print("\nBy Difficulty:")
        for diff, metrics in data['difficulty_breakdown'].items():
            print(f"  {diff.capitalize()}: MAE {metrics['mae']:.4f}")
    
    print("-"*80 + "\n")

# Save summary
with open(f'{results_dir}/comparison_summary.json', 'w') as f:
    json.dump({
        'reasoning_models': [r for r in valid_results if 'reasoning' in r.get('evaluation_type', '')],
        'finetuned_models': [r for r in valid_results if 'fine_tuned' in r.get('evaluation_type', '')],
        'failed_models': invalid_results
    }, f, indent=2)

print(f"\nResults saved to: {results_dir}")
print(f"Summary saved to: {results_dir}/comparison_summary.json")
PYTHON

echo ""
echo "========================================================================"
echo "Experiment Complete at $(date)"
echo "========================================================================"
echo "Results directory: $RESULTS_DIR"
