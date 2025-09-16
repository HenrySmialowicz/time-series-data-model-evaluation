#!/usr/bin/env python3
"""
Comprehensive evaluation for CSV-based expanded GAMETime
"""
import torch
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pathlib import Path
import argparse
import sys
from typing import Dict, List

# Add scripts to path
sys.path.append(str(Path(__file__).parent))
from probability_prediction_model import ProbabilityPredictor

class ExpandedGameTimeEvaluator:
    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def load_and_evaluate_model(self, model_path: str, model_name: str, 
                               prediction_mode: str, test_data: List[Dict]) -> Dict:
        """Load model and evaluate it"""
        print(f"Evaluating {model_name} ({prediction_mode})...")
        
        try:
            # Load model
            model = ProbabilityPredictor(model_name, prediction_mode)
            model.load_state_dict(torch.load(f"{model_path}/best_model.pth", map_location='cpu'))
            model.eval()
            
            # Make predictions
            predictions = []
            targets = []
            metadata = []
            
            with torch.no_grad():
                batch_size = 8
                for i in range(0, len(test_data), batch_size):
                    batch = test_data[i:i+batch_size]
                    batch_dict = {'sequences': batch}
                    
                    pred = model(batch_dict)
                    predictions.extend(pred.cpu().numpy().flatten())
                    
                    # Get targets and metadata
                    for seq in batch:
                        target = seq['target_win_probs'][0]
                        if prediction_mode == "change":
                            current = seq['current_win_prob']
                            target = target - current
                        targets.append(target)
                        metadata.append(seq)
            
            # Calculate comprehensive metrics
            evaluation_results = self.calculate_metrics(predictions, targets, metadata)
            
            return {
                'model_name': model_name,
                'prediction_mode': prediction_mode,
                'predictions': predictions,
                'targets': targets,
                'metadata': metadata,
                'metrics': evaluation_results,
                'model_path': model_path
            }
            
        except Exception as e:
            print(f"Error evaluating {model_name}: {e}")
            return None
    
    def calculate_metrics(self, predictions: List[float], 
                         targets: List[float], metadata: List[Dict]) -> Dict:
        """Calculate comprehensive evaluation metrics"""
        predictions = np.array(predictions)
        targets = np.array(targets)
        
        # Basic metrics
        mae = mean_absolute_error(targets, predictions)
        mse = mean_squared_error(targets, predictions)
        rmse = np.sqrt(mse)
        
        # Direction accuracy
        direction_accuracy = self.calculate_direction_accuracy(predictions, targets, metadata)
        
        # Performance by game phase
        phase_metrics = self.analyze_by_game_phase(predictions, targets, metadata)
        
        # Performance by difficulty
        difficulty_metrics = self.analyze_by_difficulty(predictions, targets, metadata)
        
        return {
            'basic_metrics': {
                'mae': float(mae),
                'mse': float(mse), 
                'rmse': float(rmse),
                'sample_count': len(predictions)
            },
            'direction_accuracy': float(direction_accuracy),
            'phase_metrics': phase_metrics,
            'difficulty_metrics': difficulty_metrics
        }
    
    def calculate_direction_accuracy(self, predictions: np.ndarray, 
                                   targets: np.ndarray, metadata: List[Dict]) -> float:
        """Calculate direction prediction accuracy"""
        correct = 0
        total = 0
        
        for i, meta in enumerate(metadata):
            if 'current_win_prob' in meta:
                current = meta['current_win_prob']
                
                pred_direction = predictions[i] - current
                actual_direction = targets[i] - current
                
                # Check if directions match (or both are very small)
                if (pred_direction * actual_direction > 0) or \
                   (abs(pred_direction) < 0.01 and abs(actual_direction) < 0.01):
                    correct += 1
                
                total += 1
        
        return correct / max(total, 1)
    
    def analyze_by_game_phase(self, predictions: np.ndarray, 
                            targets: np.ndarray, metadata: List[Dict]) -> Dict:
        """Analyze performance by game phase"""
        phase_results = {}
        
        for phase in ['early', 'middle', 'late']:
            phase_mask = [meta.get('game_phase') == phase for meta in metadata]
            phase_indices = np.where(phase_mask)[0]
            
            if len(phase_indices) > 0:
                phase_pred = predictions[phase_indices]
                phase_targets = targets[phase_indices]
                
                mae = mean_absolute_error(phase_targets, phase_pred)
                phase_results[phase] = {
                    'mae': float(mae),
                    'count': len(phase_indices)
                }
        
        return phase_results
    
    def analyze_by_difficulty(self, predictions: np.ndarray,
                            targets: np.ndarray, metadata: List[Dict]) -> Dict:
        """Analyze performance by difficulty level"""
        difficulty_results = {}
        
        for difficulty in ['easy', 'medium', 'hard']:
            diff_mask = [meta.get('difficulty') == difficulty for meta in metadata]
            diff_indices = np.where(diff_mask)[0]
            
            if len(diff_indices) > 0:
                diff_pred = predictions[diff_indices]
                diff_targets = targets[diff_indices]
                
                mae = mean_absolute_error(diff_targets, diff_pred)
                difficulty_results[difficulty] = {
                    'mae': float(mae),
                    'count': len(diff_indices)
                }
        
        return difficulty_results
    
    def create_evaluation_plots(self, results: Dict, model_key: str):
        """Create evaluation plots"""
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        predictions = np.array(results['predictions'])
        targets = np.array(results['targets'])
        
        # 1. Predictions vs Targets scatter plot
        axes[0, 0].scatter(targets, predictions, alpha=0.6, s=20)
        axes[0, 0].plot([targets.min(), targets.max()], [targets.min(), targets.max()], 'r--', lw=2)
        axes[0, 0].set_xlabel('Actual Values')
        axes[0, 0].set_ylabel('Predicted Values')
        axes[0, 0].set_title(f'{model_key} - Predictions vs Actual')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Error distribution
        errors = predictions - targets
        axes[0, 1].hist(errors, bins=30, alpha=0.7, edgecolor='black')
        axes[0, 1].set_xlabel('Prediction Error')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Error Distribution')
        axes[0, 1].axvline(0, color='red', linestyle='--')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. MAE by game phase
        phase_metrics = results['metrics']['phase_metrics']
        if phase_metrics:
            phases = list(phase_metrics.keys())
            mae_values = [phase_metrics[phase]['mae'] for phase in phases]
            axes[1, 0].bar(phases, mae_values, color=['lightblue', 'lightgreen', 'lightcoral'])
            axes[1, 0].set_ylabel('MAE')
            axes[1, 0].set_title('MAE by Game Phase')
            axes[1, 0].grid(True, alpha=0.3)
        
        # 4. MAE by difficulty
        diff_metrics = results['metrics']['difficulty_metrics']
        if diff_metrics:
            difficulties = list(diff_metrics.keys())
            diff_mae_values = [diff_metrics[diff]['mae'] for diff in difficulties]
            colors = ['green', 'orange', 'red'][:len(difficulties)]
            axes[1, 1].bar(difficulties, diff_mae_values, color=colors)
            axes[1, 1].set_ylabel('MAE')
            axes[1, 1].set_title('MAE by Difficulty')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / f'{model_key}_evaluation.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_comparison_report(self, all_results: Dict) -> str:
        """Generate comprehensive comparison report"""
        report = "# CSV-Based Expanded GAMETime Evaluation Report\n\n"
        report += f"Generated on: {np.datetime64('now')}\n\n"
        
        # Model comparison table
        report += "## Model Performance Comparison\n\n"
        report += "| Model | Mode | MAE | RMSE | Direction Acc | Samples |\n"
        report += "|-------|------|-----|------|---------------|----------|\n"
        
        for model_key, results in all_results.items():
            if results and 'metrics' in results:
                metrics = results['metrics']
                basic = metrics['basic_metrics']
                
                report += f"| {results['model_name']} | {results['prediction_mode']} | "
                report += f"{basic['mae']:.4f} | {basic['rmse']:.4f} | "
                report += f"{metrics['direction_accuracy']:.3f} | {basic['sample_count']} |\n"
        
        # Best performing model
        valid_results = {k: v for k, v in all_results.items() if v and 'metrics' in v}
        if valid_results:
            best_model = min(valid_results.keys(), 
                           key=lambda x: valid_results[x]['metrics']['basic_metrics']['mae'])
            best_mae = valid_results[best_model]['metrics']['basic_metrics']['mae']
            
            report += f"\n## Key Findings\n\n"
            report += f"- **Best Overall Model**: {best_model} (MAE: {best_mae:.4f})\n"
            
            # Performance analysis
            report += f"\n### Performance Analysis\n\n"
            for model_key, results in valid_results.items():
                if results['metrics']['phase_metrics']:
                    report += f"**{model_key} by Game Phase:**\n"
                    for phase, metrics in results['metrics']['phase_metrics'].items():
                        report += f"- {phase.title()}: MAE {metrics['mae']:.4f} ({metrics['count']} samples)\n"
                    report += "\n"
        
        return report

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data", required=True, help="Test data JSON file")
    parser.add_argument("--models_dir", required=True, help="Directory containing trained models")
    parser.add_argument("--output_dir", required=True, help="Output directory for results")
    parser.add_argument("--max_samples", type=int, default=500, help="Max test samples to use")
    
    args = parser.parse_args()
    
    evaluator = ExpandedGameTimeEvaluator(args.output_dir)
    
    # Load test data
    print("Loading test data...")
    with open(args.test_data, 'r') as f:
        test_data = json.load(f)
    
    # Limit test samples if specified
    if args.max_samples and len(test_data) > args.max_samples:
        test_data = test_data[:args.max_samples]
    
    print(f"Using {len(test_data)} test samples")
    
    # Find all trained models
    models_dir = Path(args.models_dir)
    model_results = {}
    
    # Look for model directories
    for prediction_mode in ['direct', 'change']:
        mode_dir = models_dir / prediction_mode
        if mode_dir.exists():
            for model_dir in mode_dir.iterdir():
                if model_dir.is_dir() and (model_dir / "best_model.pth").exists():
                    # Extract model name from directory
                    model_name_parts = model_dir.name.replace('_', '/').split('_nba')[0]
                    
                    # Common model mappings
                    if 'microsoft_DialoGPT-medium' in model_dir.name:
                        model_name = 'microsoft/DialoGPT-medium'
                    elif 'Qwen_Qwen2.5-1.5B-Instruct' in model_dir.name:
                        model_name = 'Qwen/Qwen2.5-1.5B-Instruct'
                    else:
                        model_name = model_name_parts
                    
                    model_key = f"{model_name.replace('/', '_')}_{prediction_mode}"
                    
                    # Evaluate model
                    result = evaluator.load_and_evaluate_model(
                        str(model_dir), model_name, prediction_mode, test_data
                    )
                    
                    if result:
                        model_results[model_key] = result
                        print(f"✓ {model_key}: MAE {result['metrics']['basic_metrics']['mae']:.4f}")
    
    if not model_results:
        print("No trained models found to evaluate!")
        return
    
    print(f"\n=== Evaluation Complete ===")
    print(f"Evaluated {len(model_results)} models")
    
    # Create plots for each model
    for model_key, results in model_results.items():
        evaluator.create_evaluation_plots(results, model_key)
    
    # Generate report
    report = evaluator.generate_comparison_report(model_results)
    
    with open(evaluator.results_dir / "evaluation_report.md", 'w') as f:
        f.write(report)
    
    # Save detailed results
    # Convert numpy arrays to lists for JSON serialization
    json_results = {}
    for key, value in model_results.items():
        json_value = value.copy()
        if 'predictions' in json_value:
            json_value['predictions'] = [float(x) for x in json_value['predictions']]
        if 'targets' in json_value:
            json_value['targets'] = [float(x) for x in json_value['targets']]
        # Remove metadata to reduce file size
        json_value.pop('metadata', None)
        json_results[key] = json_value
    
    with open(evaluator.results_dir / "detailed_results.json", 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"\nResults saved to: {args.output_dir}")
    print(f"Report available at: {args.output_dir}/evaluation_report.md")

if __name__ == "__main__":
    main()
