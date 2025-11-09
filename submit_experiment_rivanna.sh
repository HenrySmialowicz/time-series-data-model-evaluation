#!/bin/bash
# Submit your existing GAMETime experiment with proper Rivanna GPU allocation

echo "=== Submitting GAMETime Experiment to Rivanna GPU ==="

# -------------------------------------------------
# IMPORTANT: Update this with your actual allocation
# -------------------------------------------------
ALLOCATION="uvasrg_paid"  # Change this to your allocation

# -------------------------------------------------
# Verify setup before submission
# -------------------------------------------------

echo "1. Verifying setup..."

# Check if your existing script exists
if [ ! -f "run_expanded_gametime_experiment.sh" ]; then
    echo "ERROR: run_expanded_gametime_experiment.sh not found!"
    echo "This script should be in your current directory."
    exit 1
fi

# Run your existing verification
if [ -f "verify_setup.sh" ]; then
    echo "Running your setup verification..."
    ./verify_setup.sh
else
    echo "verify_setup.sh not found, doing basic checks..."
    
    # Basic checks
    if [ ! -d "data/raw/csv_files" ]; then
        echo "ERROR: data/raw/csv_files directory not found"
        exit 1
    fi
    
    CSV_COUNT=$(ls data/raw/csv_files/*.csv 2>/dev/null | wc -l)
    if [ $CSV_COUNT -eq 0 ]; then
        echo "ERROR: No CSV files found in data/raw/csv_files/"
        exit 1
    fi
    echo "Found $CSV_COUNT CSV files"
fi

# Check conda environment
echo "Checking conda environment..."
# Try to load conda module first
if module avail 2>&1 | grep -q "miniforge"; then
    module load $(module avail miniforge 2>&1 | grep miniforge | head -1 | awk '{print $1}') 2>/dev/null
elif module avail 2>&1 | grep -q "anaconda"; then
    module load $(module avail anaconda 2>&1 | grep anaconda | tail -1 | awk '{print $1}') 2>/dev/null
fi

if command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)" 2>/dev/null
    if conda env list | grep -q expanded_gametime; then
        echo "✓ expanded_gametime environment found"
    else
        echo "ERROR: expanded_gametime conda environment not found"
        exit 1
    fi
else
    echo "✓ conda will be loaded during job execution (skipping environment check)"
fi

# -------------------------------------------------
# Show configuration and get confirmation
# -------------------------------------------------

echo ""
echo "Experiment will be submitted with:"
echo "  - Allocation: $ALLOCATION" 
echo "  - Time limit: 2 days"
echo "  - GPU: A100 (1 GPU)"
echo "  - Memory: 80GB"
echo "  - CPUs: 8"
echo "  - Partition: gpu"
echo ""

read -p "Submit experiment with these settings? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Submission cancelled."
    exit 1
fi

# -------------------------------------------------
# Prepare and submit
# -------------------------------------------------

# Make sure scripts are executable
chmod +x run_rivanna_gametime.sh
chmod +x run_expanded_gametime_experiment.sh

# Update allocation in the script

# Create logs directory
mkdir -p logs

echo "Submitting job..."

# Submit with sbatch
JOB_ID=$(sbatch run_rivanna_gametime.sh | awk '{print $4}')

if [ -z "$JOB_ID" ]; then
    echo "ERROR: Job submission failed!"
    echo ""
    echo "Debug information:"
    echo "Available partitions:"
    sinfo -o "%P %a %l %F" 2>/dev/null || echo "Cannot access sinfo"
    echo ""
    echo "Your allocations:"
    sacctmgr show associations user=$USER format=Account%30 -P 2>/dev/null || echo "Cannot access sacctmgr"
    exit 1
fi

echo ""
echo "✓ Job submitted successfully!"
echo ""
echo "Job Details:"
echo "  Job ID: $JOB_ID"
echo "  Allocation: $ALLOCATION"
echo "  Submitted: $(date)"
echo ""
echo "Monitor your job:"
echo "  squeue -u $USER"
echo "  tail -f logs/experiment_${JOB_ID}.out"
echo "  watch -n 30 'squeue -u $USER'"
echo ""
echo "Job will run uninterrupted until completion."
echo "Check results in results/csv_experiments/ when done."
echo ""
echo "To cancel if needed:"
echo "  scancel $JOB_ID"
