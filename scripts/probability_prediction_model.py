#!/usr/bin/env python3
"""
Probability Prediction Model for Expanded GAMETime
"""
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, TrainingArguments, Trainer
import json
import numpy as np
from typing import Dict, List
import argparse
from sklearn.metrics import mean_absolute_error
import os
from datetime import datetime

class ProbabilityPredictor(nn.Module):
    def __init__(self, model_name: str, prediction_mode: str = "direct"):
        super().__init__()
        self.model_name = model_name
        self.prediction_mode = prediction_mode
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.language_model = AutoModel.from_pretrained(model_name)
        
        # Probability sequence encoder
        self.prob_encoder = nn.LSTM(
            input_size=1,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )
        
        # Fusion layer
        hidden_size = self.language_model.config.hidden_size
        self.fusion = nn.MultiheadAttention(
            embed_dim=hidden_size + 64,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Prediction head
        if prediction_mode == "direct":
            self.predictor = nn.Sequential(
                nn.Linear(hidden_size + 64, 256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(64, 1),
                nn.Sigmoid()  # Output probability [0,1]
            )
        elif prediction_mode == "change":
            self.predictor = nn.Sequential(
                nn.Linear(hidden_size + 64, 256),
                nn.ReLU(), 
                nn.Dropout(0.2),
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(64, 1),
                nn.Tanh()  # Output change [-1,1]
            )
    
    def create_prompt(self, sequence_data: Dict) -> str:
        """Create prompt for prediction"""
        if self.prediction_mode == "direct":
            return self._direct_prediction_prompt(sequence_data)
        else:
            return self._change_prediction_prompt(sequence_data)
    
    def _direct_prediction_prompt(self, data: Dict) -> str:
        """Direct probability prediction prompt"""
        prompt = "You are an NBA analyst. Predict the win probability after the next event.\n\n"
        prompt += f"Game phase: {data.get('game_phase', 'unknown')}\n"
        prompt += "Recent events and probabilities:\n"
        
        for i, event_data in enumerate(data['historical_events']):
            prompt += f"{i+1}. {event_data['event']} → {event_data['win_prob_after']:.3f}\n"
        
        prompt += f"\nNext event: {data['future_events'][0]}\n"
        prompt += "Predicted win probability:"
        
        return prompt
    
    def _change_prediction_prompt(self, data: Dict) -> str:
        """Probability change prediction prompt"""
        current_prob = data['current_win_prob']
        prompt = f"Current win probability: {current_prob:.3f}\n"
        prompt += f"Next event: {data['future_events'][0]}\n"
        prompt += "Predicted probability change:"
        
        return prompt
    
    def forward(self, input_data: Dict) -> torch.Tensor:
        """Forward pass"""
        # Create prompts and tokenize
        prompts = [self.create_prompt(data) for data in input_data['sequences']]
        tokenized = self.tokenizer(prompts, return_tensors="pt", padding=True, 
                                 truncation=True, max_length=512)
        
        # Move to device
        device = next(self.parameters()).device
        tokenized = {k: v.to(device) for k, v in tokenized.items()}
        
        # Get text embeddings
        text_outputs = self.language_model(**tokenized)
        text_features = text_outputs.last_hidden_state
        
        # Get probability sequences
        prob_sequences = []
        for seq in input_data['sequences']:
            probs = [event['win_prob_after'] for event in seq['historical_events']]
            prob_sequences.append(probs)
        
        # Pad probability sequences to same length
        max_len = max(len(seq) for seq in prob_sequences)
        padded_probs = []
        for seq in prob_sequences:
            padded = seq + [seq[-1]] * (max_len - len(seq))  # Pad with last value
            padded_probs.append(padded)
        
        prob_tensor = torch.FloatTensor(padded_probs).to(device).unsqueeze(-1)
        prob_features, _ = self.prob_encoder(prob_tensor)
        
        # Combine features
        text_pooled = text_features.mean(dim=1).unsqueeze(1)
        prob_pooled = prob_features[:, -1:, :]  # Last hidden state
        
        combined = torch.cat([text_pooled, prob_pooled], dim=-1)
        
        # Apply fusion
        fused, _ = self.fusion(combined, combined, combined)
        
        # Make prediction
        prediction = self.predictor(fused.squeeze(1))
        
        return prediction

class ProbabilityTrainer:
    def __init__(self, model: ProbabilityPredictor, device: str = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = device
        self.model = model.to(device)
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=2, gamma=0.8)
    
    def train_epoch(self, train_data: List[Dict], batch_size: int = 8) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        # Shuffle data
        import random
        random.shuffle(train_data)
        
        for i in range(0, len(train_data), batch_size):
            batch = train_data[i:i+batch_size]
            batch_dict = {'sequences': batch}
            
            self.optimizer.zero_grad()
            
            # Forward pass
            predictions = self.model(batch_dict)
            
            # Get targets
            targets = []
            for seq in batch:
                target = seq['target_win_probs'][0]  # First target
                if self.model.prediction_mode == "change":
                    # Convert to change
                    current = seq['current_win_prob']
                    target = target - current
                targets.append(target)
            
            targets = torch.FloatTensor(targets).to(self.device)
            
            # Calculate loss
            loss = nn.MSELoss()(predictions.squeeze(), targets)
            
            # Add constraint for direct prediction
            if self.model.prediction_mode == "direct":
                constraint_loss = torch.mean(
                    torch.relu(predictions - 1.0) + torch.relu(-predictions)
                )
                loss += 0.1 * constraint_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        self.scheduler.step()
        return total_loss / max(num_batches, 1)
    
    def evaluate(self, eval_data: List[Dict], batch_size: int = 8) -> Dict:
        """Evaluate model"""
        self.model.eval()
        predictions = []
        targets = []
        
        with torch.no_grad():
            for i in range(0, len(eval_data), batch_size):
                batch = eval_data[i:i+batch_size]
                batch_dict = {'sequences': batch}
                
                pred = self.model(batch_dict)
                predictions.extend(pred.cpu().numpy().flatten())
                
                # Get targets
                batch_targets = []
                for seq in batch:
                    target = seq['target_win_probs'][0]
                    if self.model.prediction_mode == "change":
                        current = seq['current_win_prob']
                        target = target - current
                    batch_targets.append(target)
                
                targets.extend(batch_targets)
        
        # Calculate metrics
        predictions = np.array(predictions)
        targets = np.array(targets)
        
        mae = mean_absolute_error(targets, predictions)
        mse = np.mean((predictions - targets) ** 2)
        rmse = np.sqrt(mse)
        
        # Direction accuracy for change prediction
        if self.model.prediction_mode == "change":
            correct_directions = np.sum(np.sign(predictions) == np.sign(targets))
            direction_acc = correct_directions / len(targets)
        else:
            # For direct prediction, check if we predicted increase/decrease correctly
            direction_acc = 0.0  # Would need historical context to compute this
        
        return {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'direction_accuracy': direction_acc,
            'predictions': predictions.tolist(),
            'targets': targets.tolist()
        }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="microsoft/DialoGPT-medium")
    parser.add_argument("--prediction_mode", default="direct", choices=["direct", "change"])
    parser.add_argument("--train_data", required=True)
    parser.add_argument("--val_data", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    
    args = parser.parse_args()
    
    print(f"=== Training {args.model_name} ({args.prediction_mode} mode) ===")
    
    # Load data
    with open(args.train_data, 'r') as f:
        train_data = json.load(f)
    
    with open(args.val_data, 'r') as f:
        val_data = json.load(f)
    
    print(f"Train samples: {len(train_data):,}")
    print(f"Validation samples: {len(val_data):,}")
    
    # Initialize model and trainer
    model = ProbabilityPredictor(args.model_name, args.prediction_mode)
    trainer = ProbabilityTrainer(model)
    
    print(f"Model loaded on device: {trainer.device}")
    
    # Training loop
    best_val_mae = float('inf')
    training_history = []
    
    for epoch in range(args.epochs):
        print(f"\n=== Epoch {epoch + 1}/{args.epochs} ===")
        
        # Train
        train_loss = trainer.train_epoch(train_data, args.batch_size)
        print(f"Train loss: {train_loss:.4f}")
        
        # Validate
        val_metrics = trainer.evaluate(val_data[:200], args.batch_size)  # Sample for speed
        val_mae = val_metrics['mae']
        
        print(f"Validation MAE: {val_mae:.4f}")
        print(f"Validation RMSE: {val_metrics['rmse']:.4f}")
        if args.prediction_mode == "change":
            print(f"Direction Accuracy: {val_metrics['direction_accuracy']:.3f}")
        
        # Save best model
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            os.makedirs(args.output_dir, exist_ok=True)
            torch.save(model.state_dict(), f"{args.output_dir}/best_model.pth")
            
            # Save training info
            model_info = {
                'model_name': args.model_name,
                'prediction_mode': args.prediction_mode,
                'best_val_mae': float(best_val_mae),
                'epoch': epoch + 1,
                'train_samples': len(train_data),
                'val_samples': len(val_data),
                'training_date': datetime.now().isoformat()
            }
            
            with open(f"{args.output_dir}/model_info.json", 'w') as f:
                json.dump(model_info, f, indent=2)
        
        training_history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_mae': val_mae,
            'val_rmse': val_metrics['rmse']
        })
    
    # Save training history
    with open(f"{args.output_dir}/training_history.json", 'w') as f:
        json.dump(training_history, f, indent=2)
    
    print(f"\n=== Training Complete ===")
    print(f"Best validation MAE: {best_val_mae:.4f}")
    print(f"Model saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
