#!/bin/bash
#SBATCH --job-name=expanded_gametime
#SBATCH --output=logs/experiment_%j.out
#SBATCH --error=logs/experiment_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

# =============================================================================
# Complete Expanded GAMETime Experiment Master Script
# Processes CSV data, trains models, and evaluates results
# =============================================================================

set -e  # Exit on error

# Configuration
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export CUDA_VISIBLE_DEVICES=0

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# =============================================================================
# Phase 1: Environment Verification
# =============================================================================

log "Starting Expanded GAMETime Experiment"
log "Phase 1: Environment Verification"

# Check allocation
if [ -z "$SLURM_JOB_ACCOUNT" ]; then
    error "No SLURM allocation detected. Please run with sbatch or salloc."
    exit 1
fi

log "Using allocation: $SLURM_JOB_ACCOUNT"

# Load modules
log "Loading required modules..."
module load miniforge/22.11.1-4
module load cuda/11.8
module load gcc/9.2.0

# Activate environment
log "Activating conda environment..."
eval "$(conda shell.bash hook)"
conda activate expanded_gametime

# Verify environment
log "Verifying environment..."
python -c "
import torch
import transformers
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'Current GPU: {torch.cuda.current_device()}')
    print(f'GPU name: {torch.cuda.get_device_name()}')
"

# =============================================================================
# Phase 2: CSV Data Processing
# =============================================================================

log "Phase 2: CSV Data Processing"

# Check if CSV files exist
CSV_DIR="data/raw/csv_files"
if [ ! -d "$CSV_DIR" ] || [ -z "$(ls -A $CSV_DIR 2>/dev/null)" ]; then
    error "No CSV files found in $CSV_DIR"
    error "Please upload your NBA CSV files to this directory first"
    exit 1
fi

CSV_COUNT=$(ls $CSV_DIR/*.csv 2>/dev/null | wc -l)
log "Found $CSV_COUNT CSV files to process"

# Process CSV files if not already done
PROCESSED_FILE="data/processed/csv_output/nba_2024_25_processed.json"
if [ ! -f "$PROCESSED_FILE" ]; then
    log "Processing CSV files..."
    python scripts/csv_data_processor.py \
        --csv_folder "$CSV_DIR" \
        --output_path "data/processed/csv_output" \
        2>&1 | tee logs/csv_processing.log
    
    if [ $? -eq 0 ]; then
        success "CSV processing completed"
    else
        error "CSV processing failed. Check logs/csv_processing.log"
        exit 1
    fi
else
    success "CSV files already processed"
fi

# =============================================================================
# Phase 3: Dataset Creation
# =============================================================================

log "Phase 3: Creating Expanded Dataset"

DATASET_DIR="data/expanded_experiments/nba"
if [ ! -f "$DATASET_DIR/train.json" ]; then
    log "Creating expanded prediction dataset..."
    python scripts/create_csv_expanded_dataset.py \
        --processed_data "$PROCESSED_FILE" \
        --output_path "data/expanded_experiments" \
        2>&1 | tee logs/dataset_creation.log
    
    if [ $? -eq 0 ]; then
        success "Dataset creation completed"
    else
        error "Dataset creation failed. Check logs/dataset_creation.log"
        exit 1
    fi
else
    success "Expanded dataset already exists"
fi

# Verify dataset
log "Dataset statistics:"
python -c "
import json
with open('$DATASET_DIR/train.json', 'r') as f: train = json.load(f)
with open('$DATASET_DIR/val.json', 'r') as f: val = json.load(f)
with open('$DATASET_DIR/test.json', 'r') as f: test = json.load(f)
print(f'Train samples: {len(train):,}')
print(f'Validation samples: {len(val):,}')
print(f'Test samples: {len(test):,}')
print(f'Total samples: {len(train) + len(val) + len(test):,}')
"

# =============================================================================
# Phase 4: Model Training
# =============================================================================

log "Phase 4: Model Training"

# Define models and modes to train
declare -a MODELS=(
    "microsoft/DialoGPT-medium"
    "Qwen/Qwen2.5-1.5B-Instruct"
)

declare -a MODES=("direct" "change")

TRAIN_DATA="$DATASET_DIR/train.json"
VAL_DATA="$DATASET_DIR/val.json"

# Train all model combinations
for MODEL in "${MODELS[@]}"; do
    for MODE in "${MODES[@]}"; do
        MODEL_SAFE=$(echo "$MODEL" | sed 's/\//_/g')
        OUTPUT_DIR="models/probability_prediction/$MODE/${MODEL_SAFE}_nba_2024"
        
        if [ ! -f "$OUTPUT_DIR/best_model.pth" ]; then
            log "Training $MODEL ($MODE mode)..."
            
            mkdir -p "$OUTPUT_DIR"
            
            python scripts/probability_prediction_model.py \
                --model_name "$MODEL" \
                --prediction_mode "$MODE" \
                --train_data "$TRAIN_DATA" \
                --val_data "$VAL_DATA" \
                --output_dir "$OUTPUT_DIR" \
                --epochs 3 \
                --batch_size 6 \
                2>&1 | tee "logs/training_${MODEL_SAFE}_${MODE}.log"
            
            if [ $? -eq 0 ]; then
                success "Training completed: $MODEL ($MODE)"
            else
                warning "Training failed: $MODEL ($MODE). Check logs/training_${MODEL_SAFE}_${MODE}.log"
            fi
        else
            success "Model already trained: $MODEL ($MODE)"
        fi
    done
done

# =============================================================================
# Phase 5: Comprehensive Evaluation
# =============================================================================

log "Phase 5: Comprehensive Evaluation"

TEST_DATA="$DATASET_DIR/test.json"
RESULTS_DIR="results/csv_experiments/$(date +%Y%m%d_%H%M%S)"

log "Running comprehensive evaluation..."
python scripts/comprehensive_evaluation.py \
    --test_data "$TEST_DATA" \
    --models_dir "models/probability_prediction" \
    --output_dir "$RESULTS_DIR" \
    --max_samples 1000 \
    2>&1 | tee logs/evaluation.log

if [ $? -eq 0 ]; then
    success "Evaluation completed"
    log "Results saved to: $RESULTS_DIR"
else
    error "Evaluation failed. Check logs/evaluation.log"
    exit 1
fi

# =============================================================================
# Phase 6: Generate Final Report
# =============================================================================

log "Phase 6: Generating Final Report"

# Create comprehensive summary
cat > "$RESULTS_DIR/experiment_summary.md" << EOL
# Expanded GAMETime CSV Experiment Summary

## Experiment Details
- **Date**: $(date)
- **Job ID**: $SLURM_JOB_ID
- **Allocation**: $SLURM_JOB_ACCOUNT
- **Node**: $SLURMD_NODENAME
- **CSV Files Processed**: $CSV_COUNT
- **GPU Used**: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits)

## Dataset Statistics
$(cat $DATASET_DIR/dataset_statistics.json 2>/dev/null || echo "Statistics not available")

## Models Trained
$(find models/probability_prediction -name "model_info.json" -exec echo "- {}" \; -exec cat {} \; -exec echo "" \; 2>/dev/null | head -20)

## Evaluation Results
See evaluation_report.md for detailed results.

## Files Generated
- Processed CSV data: data/processed/csv_output/
- Training datasets: data/expanded_experiments/nba/
- Trained models: models/probability_prediction/
- Evaluation results: $RESULTS_DIR/
- Logs: logs/

## Next Steps
1. Review evaluation_report.md for model performance comparison
2. Check individual model plots in the results directory  
3. Examine training logs for any issues or improvements
4. Consider hyperparameter tuning for best-performing models
EOL

# =============================================================================
# Final Summary
# =============================================================================

log "Experiment completed successfully!"
log "=================================================="
log "EXPERIMENT SUMMARY"
log "=================================================="

# Count successful models
SUCCESSFUL_MODELS=$(find models/probability_prediction -name "best_model.pth" | wc -l)
log "Models trained: $SUCCESSFUL_MODELS"

# Show results directory
log "Results directory: $RESULTS_DIR"

# Show key files
if [ -f "$RESULTS_DIR/evaluation_report.md" ]; then
    log "Main report: $RESULTS_DIR/evaluation_report.md"
fi

if [ -f "$RESULTS_DIR/detailed_results.json" ]; then
    log "Detailed results: $RESULTS_DIR/detailed_results.json"
fi

# Show best model if evaluation completed
if [ -f "$RESULTS_DIR/detailed_results.json" ]; then
    BEST_MODEL=$(python -c "
import json
with open('$RESULTS_DIR/detailed_results.json', 'r') as f:
    results = json.load(f)
if results:
    best = min(results.keys(), key=lambda x: results[x]['metrics']['basic_metrics']['mae'])
    mae = results[best]['metrics']['basic_metrics']['mae']
    print(f'{best}: MAE {mae:.4f}')
")
    if [ -n "$BEST_MODEL" ]; then
        log "Best model: $BEST_MODEL"
    fi
fi

log "Check the results directory for detailed analysis and plots"
success "All phases completed successfully!"

