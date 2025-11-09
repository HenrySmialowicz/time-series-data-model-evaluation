#!/bin/bash

# Simple GAMETime experiment that focuses on training and evaluation
# set -e  # Commented out to prevent early exit

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Simple GAMETime Training"

# Configuration
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export CUDA_VISIBLE_DEVICES=0

# Data paths
TRAIN_DATA="data/expanded_experiments/nba/train.json"
VAL_DATA="data/expanded_experiments/nba/val.json"
TEST_DATA="data/expanded_experiments/nba/test.json"

# Verify data exists
if [ ! -f "$TRAIN_DATA" ]; then
    echo "ERROR: Training data not found at $TRAIN_DATA"
    exit 1
fi

echo "Training data found:"
echo "  Train: $(wc -c < $TRAIN_DATA | numfmt --to=iec)"
echo "  Val: $(wc -c < $VAL_DATA | numfmt --to=iec)"
echo "  Test: $(wc -c < $TEST_DATA | numfmt --to=iec)"

# Define models (using compatible models that work with your PyTorch version)
declare -a MODELS=(
    "distilbert-base-uncased"
    "bert-base-uncased"
)

declare -a MODES=("direct" "change")

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Model Training"
echo "Training $(( ${#MODELS[@]} * ${#MODES[@]} )) model combinations..."

# Train all combinations
SUCCESSFUL_MODELS=0
echo "DEBUG: About to start training loop with models: ${MODELS[*]}"
for MODEL in "${MODELS[@]}"; do
    for MODE in "${MODES[@]}"; do
        MODEL_SAFE=$(echo "$MODEL" | sed 's/\//_/g')
        OUTPUT_DIR="models/probability_prediction/$MODE/${MODEL_SAFE}_nba_2024"
        
        echo ""
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Training $MODEL ($MODE mode)..."
        mkdir -p "$OUTPUT_DIR"
        
        # Train with reasonable parameters for quick completion
        python scripts/probability_prediction_model.py \
            --model_name "$MODEL" \
            --prediction_mode "$MODE" \
            --train_data "$TRAIN_DATA" \
            --val_data "$VAL_DATA" \
            --output_dir "$OUTPUT_DIR" \
            --epochs 2 \
            --batch_size 8 \
            2>&1 | tee "logs/training_${MODEL_SAFE}_${MODE}.log"
        
        if [ $? -eq 0 ] && [ -f "$OUTPUT_DIR/best_model.pth" ]; then
            echo "[SUCCESS] Training completed: $MODEL ($MODE)"
            ((SUCCESSFUL_MODELS++))
        else
            echo "[WARNING] Training failed: $MODEL ($MODE)"
        fi
    done
echo "DEBUG: Completed training loop"
done
echo "DEBUG: Completed training loop"

echo ""
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Training Complete"
echo "Successfully trained: $SUCCESSFUL_MODELS models"

# Only run evaluation if we have trained models
if [ $SUCCESSFUL_MODELS -gt 0 ]; then
    echo ""
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Evaluation"
    
    RESULTS_DIR="results/csv_experiments/$(date +%Y%m%d_%H%M%S)_job_${SLURM_JOB_ID}"
    mkdir -p "$RESULTS_DIR"
    
    python scripts/comprehensive_evaluation.py \
        --test_data "$TEST_DATA" \
        --models_dir "models/probability_prediction" \
        --output_dir "$RESULTS_DIR" \
        --max_samples 500 \
        2>&1 | tee logs/evaluation_${SLURM_JOB_ID}.log
    
    if [ $? -eq 0 ]; then
        echo "[SUCCESS] Evaluation completed"
        echo "Results saved to: $RESULTS_DIR"
        
        # Generate summary
        cat > "$RESULTS_DIR/experiment_summary.txt" << EOL
GAMETime Simple Experiment Summary
================================
Job ID: ${SLURM_JOB_ID}
Completion Time: $(date)
Models Trained: $SUCCESSFUL_MODELS
Data Samples: 177,102 total
Results Directory: $RESULTS_DIR

Files Generated:
- evaluation_report.md: Main performance comparison
- detailed_results.json: Complete metrics data
- Model plots: Individual performance visualizations
EOL
        
        echo ""
        echo "=== EXPERIMENT SUMMARY ==="
        cat "$RESULTS_DIR/experiment_summary.txt"
        
        # Show quick results if available
        if [ -f "$RESULTS_DIR/evaluation_report.md" ]; then
            echo ""
            echo "=== QUICK RESULTS ==="
            head -20 "$RESULTS_DIR/evaluation_report.md"
        fi
        
    else
        echo "[WARNING] Evaluation failed"
    fi
else
    echo "[ERROR] No models were successfully trained - skipping evaluation"
fi

echo ""
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Experiment Complete!"
