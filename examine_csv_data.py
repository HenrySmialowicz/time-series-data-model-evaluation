#!/usr/bin/env python3
"""
Interactive CSV data examiner
"""
import pandas as pd
import sys
from pathlib import Path

def examine_csv_files(csv_dir="data/raw/csv_files"):
    csv_dir = Path(csv_dir)
    csv_files = list(csv_dir.glob("*.csv"))
    
    if not csv_files:
        print(f"No CSV files found in {csv_dir}")
        return
    
    print(f"Found {len(csv_files)} CSV files")
    print("\nExamining first file:", csv_files[0].name)
    
    try:
        df = pd.read_csv(csv_files[0])
        print(f"\nShape: {df.shape}")
        print(f"\nColumns ({len(df.columns)}):")
        for i, col in enumerate(df.columns, 1):
            print(f"  {i:2d}. {col}")
        
        print(f"\nFirst 3 rows:")
        print(df.head(3))
        
        print(f"\nData types:")
        print(df.dtypes)
        
        print(f"\nMissing values:")
        print(df.isnull().sum())
        
    except Exception as e:
        print(f"Error reading CSV: {e}")

if __name__ == "__main__":
    csv_dir = sys.argv[1] if len(sys.argv) > 1 else "data/raw/csv_files"
    examine_csv_files(csv_dir)
