#!/usr/bin/env python3
"""
Create expanded prediction dataset from processed CSV data
"""
import json
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
from typing import List, Dict

class CSVExpandedDatasetCreator:
    def __init__(self, processed_data_file: str, output_path: str):
        self.processed_data_file = processed_data_file
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
    
    def create_prediction_sequences(self, game_data: Dict) -> List[Dict]:
        """Create prediction sequences from a game"""
        events = game_data['events']
        win_probs = game_data['win_probabilities']
        
        if len(events) < 15:  # Need enough events for meaningful sequences
            return []
        
        sequences = []
        
        # Different window sizes and prediction horizons
        window_sizes = [5, 10, 15]  # Look back this many events
        prediction_horizons = [1, 3, 5]  # Predict this many events ahead
        
        for window_size in window_sizes:
            for horizon in prediction_horizons:
                # Create overlapping sequences
                for i in range(window_size, len(events) - horizon):
                    sequence = {
                        "game_id": game_data['game_id'],
                        "sequence_id": f"w{window_size}_h{horizon}_i{i}",
                        "sport": "nba",
                        "season": "2024-25",
                        "window_size": window_size,
                        "prediction_horizon": horizon,
                        
                        # Historical context (what the model sees)
                        "historical_events": [
                            {
                                "timestamp": events[j]['timestamp'],
                                "event": events[j]['description'],
                                "win_prob_after": win_probs[j]
                            }
                            for j in range(i-window_size, i)
                        ],
                        
                        # What to predict
                        "future_events": [events[j]['description'] for j in range(i, i+horizon)],
                        "target_win_probs": win_probs[i:i+horizon],
                        
                        # Additional context
                        "current_win_prob": win_probs[i-1],
                        "difficulty": self.calculate_difficulty(win_probs[i-window_size:i]),
                        "game_phase": self.determine_game_phase(i, len(events)),
                        "date": game_data.get('date', 'unknown'),
                        "teams": game_data.get('teams', {})
                    }
                    sequences.append(sequence)
        
        return sequences
    
    def calculate_difficulty(self, prob_sequence: List[float]) -> str:
        """Categorize prediction difficulty based on probability variance"""
        variance = np.var(prob_sequence)
        if variance < 0.005:
            return "easy"      # Stable game
        elif variance < 0.02:
            return "medium"    # Some volatility
        else:
            return "hard"      # High volatility
    
    def determine_game_phase(self, current_event: int, total_events: int) -> str:
        """Determine game phase"""
        progress = current_event / total_events
        if progress < 0.33:
            return "early"
        elif progress < 0.67:
            return "middle"
        else:
            return "late"
    
    def create_dataset(self):
        """Main function to create the dataset"""
        print("Loading processed CSV data...")
        with open(self.processed_data_file, 'r') as f:
            games_data = json.load(f)
        
        print(f"Found {len(games_data)} games")
        
        # Create prediction sequences for all games
        all_sequences = []
        for game_data in tqdm(games_data, desc="Creating prediction sequences"):
            sequences = self.create_prediction_sequences(game_data)
            all_sequences.extend(sequences)
        
        print(f"Created {len(all_sequences)} prediction sequences")
        
        # Split chronologically to avoid data leakage
        all_sequences.sort(key=lambda x: x['date'])
        
        n = len(all_sequences)
        train_end = int(0.7 * n)
        val_end = int(0.85 * n)
        
        train_data = all_sequences[:train_end]
        val_data = all_sequences[train_end:val_end]
        test_data = all_sequences[val_end:]
        
        # Save splits
        output_dir = self.output_path / "nba"
        output_dir.mkdir(exist_ok=True)
        
        with open(output_dir / "train.json", 'w') as f:
            json.dump(train_data, f, indent=2)
        
        with open(output_dir / "val.json", 'w') as f:
            json.dump(val_data, f, indent=2)
        
        with open(output_dir / "test.json", 'w') as f:
            json.dump(test_data, f, indent=2)
        
        # Generate statistics
        stats = {
            "total_sequences": len(all_sequences),
            "train_size": len(train_data),
            "val_size": len(val_data),
            "test_size": len(test_data),
            "difficulty_distribution": {
                "easy": sum(1 for s in all_sequences if s["difficulty"] == "easy"),
                "medium": sum(1 for s in all_sequences if s["difficulty"] == "medium"),
                "hard": sum(1 for s in all_sequences if s["difficulty"] == "hard")
            }
        }
        
        with open(output_dir / "dataset_statistics.json", 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"\n=== Dataset Created Successfully ===")
        print(f"Train samples: {len(train_data):,}")
        print(f"Validation samples: {len(val_data):,}")
        print(f"Test samples: {len(test_data):,}")
        print(f"Difficulty distribution:")
        for diff, count in stats["difficulty_distribution"].items():
            print(f"  {diff}: {count:,} ({count/len(all_sequences)*100:.1f}%)")
        
        return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_data", required=True, help="Path to processed CSV data JSON file")
    parser.add_argument("--output_path", required=True, help="Output path for expanded dataset")
    
    args = parser.parse_args()
    
    creator = CSVExpandedDatasetCreator(args.processed_data, args.output_path)
    success = creator.create_dataset()
    
    if success:
        print("✓ Expanded dataset creation completed successfully!")
    else:
        print("✗ Dataset creation failed!")
