#!/usr/bin/env python3
import argparse
import pandas as pd
import json
from pathlib import Path
from tqdm import tqdm

def process_csv(file_path):
    try:
        df = pd.read_csv(file_path)
        required_cols = [col for col in df.columns if col.startswith("wp_")]
        if not required_cols:
            print(f"❌ Skipping {file_path.name}: No win probability columns found")
            print(f"   -> Columns in file: {df.columns.tolist()}\n")
            return None

        game_data = {
            "file": file_path.name,
            "columns": df.columns.tolist(),
            "events": df.to_dict(orient="records"),
        }
        return game_data

    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")
        return None

def main(csv_folder, output_path):
    csv_folder = Path(csv_folder)
    output_path = Path(output_path)

    all_games = []
    failed_files = []

    files = list(csv_folder.glob("*.csv"))
    for file in tqdm(files, desc="Processing CSV files"):
        game_data = process_csv(file)
        if game_data:
            all_games.append(game_data)
        else:
            failed_files.append(file.name)

    output_path.mkdir(parents=True, exist_ok=True)
    out_file = output_path / "nba_2024_25_processed.json"

    with open(out_file, "w") as f:
        json.dump(all_games, f)

    print("\n=== Processing Complete ===")
    print(f"Successfully processed: {len(all_games)} files")
    print(f"Failed to process: {len(failed_files)} files")
    if failed_files:
        print(f"Failed files: {failed_files[:5]}...")
    print(f"Saved {len(all_games)} games to {out_file}\n")

    if all_games:
        total_events = sum(len(g["events"]) for g in all_games)
        avg_events = total_events / len(all_games)
        dates = set()
        for g in all_games:
            for ev in g["events"]:
                if "date" in ev:
                    dates.add(ev["date"])
        print("=== Dataset Statistics ===")
        print(f"Total games: {len(all_games)}")
        print(f"Total events: {total_events}")
        print(f"Average events per game: {avg_events:.1f}")
        print(f"Date range: {len(dates)} unique dates")
    print("\n✓ CSV processing completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_folder", required=True)
    parser.add_argument("--output_path", required=True)
    args = parser.parse_args()
    main(args.csv_folder, args.output_path)
