#!/bin/bash
echo "=== Submitting Simple GAMETime Experiment ==="

# Basic checks
if [ ! -f "run_gametime_simple.sh" ] || [ ! -f "run_rivanna_simple.sh" ]; then
    echo "ERROR: Required scripts not found"
    exit 1
fi

if [ ! -d "data/expanded_experiments/nba" ]; then
    echo "ERROR: Training data not found"
    echo "Expected: data/expanded_experiments/nba/"
    exit 1
fi

# Show configuration
echo ""
echo "Configuration:"
echo "  - Time limit: 2 hours"
echo "  - GPU: A100"
echo "  - Memory: 32GB"
echo "  - Models: DistilBERT + BERT"
echo "  - Modes: direct + change"
echo "  - Training samples: ~124k"
echo ""

read -p "Submit experiment? (y/N): " -r
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

# Make sure logs directory exists
mkdir -p logs

# Submit job
echo "Submitting to Rivanna..."
JOB_ID=$(sbatch run_rivanna_simple.sh | awk '{print $4}')

if [ -n "$JOB_ID" ]; then
    echo ""
    echo "✓ Job submitted successfully!"
    echo "Job ID: $JOB_ID"
    echo ""
    echo "Monitor progress:"
    echo "  squeue -j $JOB_ID"
    echo "  tail -f logs/experiment_${JOB_ID}.out"
    echo ""
    echo "Expected runtime: 30-90 minutes"
    echo "Results will be in: results/csv_experiments/"
else
    echo "✗ Job submission failed"
fi
