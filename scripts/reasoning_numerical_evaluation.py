#!/usr/bin/env python3
"""
GAMETime-style numerical reasoning evaluation
Models predict actual probability values, errors calculated against ESPN data
"""
import json
import random
import argparse
import os
import re
from typing import List, Dict, Optional, Tuple
import numpy as np
import sys
from pathlib import Path

# Add scripts to path
sys.path.append(str(Path(__file__).parent))
from local_model_client import LocalModelClient

class NumericalReasoningEvaluator:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.client = LocalModelClient(model_name)
    
    def create_numerical_prompt(self, sequence_data: Dict) -> Tuple[str, float]:
        """Create prompt for numerical prediction"""
        historical = sequence_data['historical_events']
        current_prob = sequence_data['current_win_prob']
        target_prob = sequence_data['target_win_probs'][0]  # ESPN actual value
        next_event = sequence_data['future_events'][0]
        game_phase = sequence_data.get('game_phase', 'unknown')
        
        # Build contextual prompt
        prompt = f"""Analyze this NBA game sequence and predict the exact win probability.

Game Context:
- Phase: {game_phase}
- Current Win Probability (Team A): {current_prob:.3f}

Recent Event Sequence (showing actual probability changes):
"""
        
        # Show last 5-8 events with probabilities
        num_events = min(8, len(historical))
        for i, event in enumerate(historical[-num_events:], 1):
            prompt += f"{i}. {event['event']}\n"
            prompt += f"   Win Prob: {event['win_prob_after']:.3f}\n"
        
        prompt += f"\nNext Event to Analyze:\n{next_event}\n\n"
        prompt += "Based on the pattern of probability changes and the nature of this event, "
        prompt += "predict Team A's win probability after this event occurs.\n\n"
        prompt += "Think through:\n"
        prompt += "1. Is this event favorable or unfavorable for Team A?\n"
        prompt += "2. How significant is this type of event?\n"
        prompt += "3. What is the game situation (close game, blowout, etc.)?\n\n"
        prompt += "Provide your prediction as a decimal number between 0.000 and 1.000\n"
        prompt += "Answer format: 0.XXX"
        
        return prompt, target_prob
    
    def extract_probability(self, response: str) -> Optional[float]:
        """Extract probability prediction from model response"""
        # Try different patterns
        patterns = [
            r'(?:^|\s)0\.\d{3}(?:\s|$)',  # 0.XXX format
            r'(?:^|\s)0\.\d{2}(?:\s|$)',   # 0.XX format
            r'(?:^|\s)0\.\d{1}(?:\s|$)',   # 0.X format
            r'(\d{1,2})\.?\d*\s*%',         # XX% format
            r'(?:probability|prob|chance).*?(\d+\.?\d*)',  # "probability of 0.65"
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, response.lower())
            if matches:
                try:
                    value = float(matches[-1])
                    # Convert percentage to decimal if needed
                    if value > 1.0:
                        value = value / 100.0
                    # Ensure valid probability range
                    if 0.0 <= value <= 1.0:
                        return value
                except ValueError:
                    continue
        
        return None
    
    def evaluate_sample(self, sample: Dict) -> Dict:
        """Evaluate one sample with numerical prediction"""
        prompt, target_prob = self.create_numerical_prompt(sample)
        response = self.client.generate_response(prompt, max_tokens=200)
        predicted_prob = self.extract_probability(response)
        
        # Calculate error metrics if prediction was extracted
        if predicted_prob is not None:
            error = abs(predicted_prob - target_prob)
            squared_error = (predicted_prob - target_prob) ** 2
            
            # Direction accuracy (for change prediction)
            current = sample['current_win_prob']
            actual_direction = target_prob - current
            pred_direction = predicted_prob - current
            direction_correct = (actual_direction * pred_direction) > 0
        else:
            error = None
            squared_error = None
            direction_correct = False
        
        return {
            'prompt_preview': prompt[:300] + "...",
            'response': response,
            'predicted_probability': predicted_prob,
            'actual_probability': target_prob,
            'current_probability': sample['current_win_prob'],
            'absolute_error': error,
            'squared_error': squared_error,
            'direction_correct': direction_correct,
            'difficulty': sample.get('difficulty', 'unknown'),
            'game_phase': sample.get('game_phase', 'unknown'),
            'extraction_failed': predicted_prob is None
        }
    
    def run_evaluation(self, test_data: List[Dict], max_samples: int = 100) -> Dict:
        """Run complete numerical evaluation"""
        print(f"\nEvaluating {self.model_name} on {max_samples} samples...")
        print("Predicting actual probability values (GAMETime-style)\n")
        
        sample_data = random.sample(test_data, min(max_samples, len(test_data)))
        
        results = []
        valid_predictions = []
        
        for i, sample in enumerate(sample_data):
            if (i + 1) % 20 == 0:
                print(f"Progress: {i+1}/{len(sample_data)}")
            
            result = self.evaluate_sample(sample)
            results.append(result)
            
            if not result['extraction_failed']:
                valid_predictions.append(result)
        
        # Calculate metrics (only on valid predictions)
        if valid_predictions:
            errors = [r['absolute_error'] for r in valid_predictions]
            squared_errors = [r['squared_error'] for r in valid_predictions]
            
            mae = np.mean(errors)
            mse = np.mean(squared_errors)
            rmse = np.sqrt(mse)
            
            # Direction accuracy
            direction_correct = sum(1 for r in valid_predictions if r['direction_correct'])
            direction_accuracy = direction_correct / len(valid_predictions)
            
            # Breakdown by difficulty
            difficulty_breakdown = {}
            for difficulty in ['easy', 'medium', 'hard']:
                diff_results = [r for r in valid_predictions if r['difficulty'] == difficulty]
                if diff_results:
                    diff_errors = [r['absolute_error'] for r in diff_results]
                    difficulty_breakdown[difficulty] = {
                        'mae': float(np.mean(diff_errors)),
                        'count': len(diff_results)
                    }
            
            # Breakdown by game phase
            phase_breakdown = {}
            for phase in ['early', 'middle', 'late']:
                phase_results = [r for r in valid_predictions if r['game_phase'] == phase]
                if phase_results:
                    phase_errors = [r['absolute_error'] for r in phase_results]
                    phase_breakdown[phase] = {
                        'mae': float(np.mean(phase_errors)),
                        'count': len(phase_results)
                    }
        else:
            mae = mse = rmse = direction_accuracy = None
            difficulty_breakdown = {}
            phase_breakdown = {}
        
        extraction_success_rate = len(valid_predictions) / len(results)
        
        return {
            'model': self.model_name,
            'evaluation_type': 'numerical_prediction_gametime_style',
            'total_samples': len(results),
            'valid_predictions': len(valid_predictions),
            'extraction_success_rate': extraction_success_rate,
            'mae': float(mae) if mae is not None else None,
            'mse': float(mse) if mse is not None else None,
            'rmse': float(rmse) if rmse is not None else None,
            'direction_accuracy': float(direction_accuracy) if direction_accuracy is not None else None,
            'difficulty_breakdown': difficulty_breakdown,
            'phase_breakdown': phase_breakdown,
            'detailed_results': results
        }
    
    def cleanup(self):
        """Clean up resources"""
        self.client.cleanup()

def main():
    parser = argparse.ArgumentParser(description="GAMETime-style numerical reasoning evaluation")
    parser.add_argument("--model", required=True, help="Model name")
    parser.add_argument("--test_data", required=True, help="Test data JSON")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--max_samples", type=int, default=100, help="Max samples")
    
    args = parser.parse_args()
    
    # Load test data
    print(f"Loading test data from {args.test_data}...")
    with open(args.test_data, 'r') as f:
        test_data = json.load(f)
    print(f"Loaded {len(test_data)} test samples")
    
    # Run evaluation
    evaluator = NumericalReasoningEvaluator(args.model)
    results = evaluator.run_evaluation(test_data, args.max_samples)
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    model_safe = args.model.replace('/', '_').replace('-', '_')
    output_file = f"{args.output_dir}/numerical_reasoning_{model_safe}.json"
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"GAMETime-Style Numerical Evaluation Results")
    print(f"Model: {args.model}")
    print(f"{'='*70}")
    
    if results['valid_predictions'] > 0:
        print(f"\nExtraction Success Rate: {results['extraction_success_rate']:.1%}")
        print(f"Valid Predictions: {results['valid_predictions']}/{results['total_samples']}")
        print(f"\nPerformance Metrics (vs ESPN Win Probabilities):")
        print(f"  MAE (Mean Absolute Error): {results['mae']:.4f}")
        print(f"  RMSE (Root Mean Squared Error): {results['rmse']:.4f}")
        print(f"  Direction Accuracy: {results['direction_accuracy']:.1%}")
        
        if results['difficulty_breakdown']:
            print(f"\nBy Difficulty:")
            for diff, metrics in results['difficulty_breakdown'].items():
                print(f"  {diff.capitalize()}: MAE {metrics['mae']:.4f} ({metrics['count']} samples)")
        
        if results['phase_breakdown']:
            print(f"\nBy Game Phase:")
            for phase, metrics in results['phase_breakdown'].items():
                print(f"  {phase.capitalize()}: MAE {metrics['mae']:.4f} ({metrics['count']} samples)")
    else:
        print(f"\nWARNING: Model failed to produce valid numerical predictions")
        print(f"Extraction failed on all {results['total_samples']} samples")
    
    print(f"\nDetailed results saved to: {output_file}")
    
    # Cleanup
    evaluator.cleanup()

if __name__ == "__main__":
    main()
