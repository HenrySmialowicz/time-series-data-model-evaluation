#!/bin/bash
# Submit the complete expanded GAMETime experiment

echo "=== Submitting Expanded GAMETime Experiment ==="

# -------------------------------------------------
# Detect allocation (account) dynamically
# -------------------------------------------------
ALLOCATIONS=$(sacctmgr show associations user=$USER format=Account%30 -P 2>/dev/null | tail -n +2 | sort | uniq)

if [ -z "$ALLOCATIONS" ]; then
    echo "ERROR: Could not detect allocation automatically."
    echo "Please provide your allocation name manually in this script."
    exit 1
fi

if [ $(echo "$ALLOCATIONS" | wc -l) -eq 1 ]; then
    ALLOCATION="$ALLOCATIONS"
    echo "Using allocation: $ALLOCATION"
else
    echo "Multiple allocations found:"
    echo "$ALLOCATIONS"
    read -p "Enter allocation to use: " ALLOCATION
fi

# -------------------------------------------------
# Detect partition (gpu preferred, fallback standard)
# -------------------------------------------------
if sinfo -o "%P" | grep -qw "gpu"; then
    PARTITION="gpu"
else
    PARTITION="standard"
fi
echo "Using partition: $PARTITION"

# Default time (can adjust)
TIME="12:00:00"

# -------------------------------------------------
# Verify setup before submitting
# -------------------------------------------------
echo "1. Verifying setup..."
./verify_setup.sh

# Check CSV files
CSV_COUNT=$(ls data/raw/csv_files/*.csv 2>/dev/null | wc -l)
if [ $CSV_COUNT -eq 0 ]; then
    echo "ERROR: No CSV files found!"
    echo "Please upload your CSV files to data/raw/csv_files/ first"
    exit 1
fi

echo "Found $CSV_COUNT CSV files to process"

# Ask for confirmation
echo ""
echo "Experiment configuration:"
echo "  - Allocation: $ALLOCATION"
echo "  - Partition: $PARTITION"
echo "  - Time limit: $TIME"
echo "  - CSV files: $CSV_COUNT"
echo ""

read -p "Submit experiment? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Experiment cancelled."
    exit 1
fi

# -------------------------------------------------
# Submit job
# -------------------------------------------------
echo "Submitting job..."
JOB_ID=$(sbatch \
    --account="$ALLOCATION" \
    --partition="$PARTITION" \
    --time="$TIME" \
    --job-name="expanded_gametime" \
    --output="logs/experiment_%j.out" \
    --error="logs/experiment_%j.err" \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=8 \
    --mem=32GB \
    --gres=gpu:1 \
    run_expanded_gametime_experiment.sh | awk '{print $4}')

if [ -z "$JOB_ID" ]; then
    echo "ERROR: Job submission failed. Check your allocation/partition."
    exit 1
fi

echo "Job submitted with ID: $JOB_ID"
echo ""
echo "Monitor progress with:"
echo "  squeue -u $USER"
echo "  tail -f logs/experiment_${JOB_ID}.out"
echo ""
echo "Cancel if needed with:"
echo "  scancel $JOB_ID"
