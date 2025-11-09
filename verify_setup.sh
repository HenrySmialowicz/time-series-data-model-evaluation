#!/bin/bash
# Quick setup verification

echo "=== Expanded GAMETime Setup Verification ==="

# Check directory structure
echo "1. Directory structure:"
for dir in data/raw/csv_files data/processed scripts models results logs; do
    if [ -d "$dir" ]; then
        echo "  ✓ $dir"
    else
        echo "  ✗ $dir (missing)"
    fi
done

# Check CSV files
CSV_COUNT=$(ls data/raw/csv_files/*.csv 2>/dev/null | wc -l)
echo "2. CSV files: $CSV_COUNT found"

# Check scripts
echo "3. Required scripts:"
for script in scripts/csv_data_processor.py scripts/create_csv_expanded_dataset.py scripts/probability_prediction_model.py scripts/comprehensive_evaluation.py; do
    if [ -f "$script" ]; then
        echo "  ✓ $script"
    else
        echo "  ✗ $script (missing)"
    fi
done

# Check conda environment
echo "4. Conda environment:"
# Try to load conda module first
if module avail 2>&1 | grep -q "miniforge"; then
    module load $(module avail miniforge 2>&1 | grep miniforge | head -1 | awk '{print $1}') 2>/dev/null
elif module avail 2>&1 | grep -q "anaconda"; then
    module load $(module avail anaconda 2>&1 | grep anaconda | tail -1 | awk '{print $1}') 2>/dev/null
fi

# Initialize conda
if command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)" 2>/dev/null
    if conda env list | grep -q expanded_gametime; then
        echo "  ✓ expanded_gametime environment exists"
    else
        echo "  ✗ expanded_gametime environment not found"
    fi
else
    echo "  ✗ conda command not available (will be loaded during job execution)"
fi

# Check modules (if on Rivanna)
if command -v module &> /dev/null; then
    echo "5. Modules available:"
    if module avail miniforge 2>&1 | grep -q miniforge; then
        echo "  ✓ miniforge"
    fi
    if module avail cuda 2>&1 | grep -q cuda; then
        echo "  ✓ cuda"
    fi
fi

echo "=== Setup verification complete ==="
