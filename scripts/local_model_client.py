#!/usr/bin/env python3
"""
Client for querying Qwen and DeepSeek models locally
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Optional

class LocalModelClient:
    def __init__(self, model_name: str, device: str = None):
        self.model_name = model_name
        
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        print(f"Loading {model_name} on {self.device}...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True
        )
        
        if self.device == "cpu":
            self.model = self.model.to(self.device)
        
        self.model.eval()
        print(f"✓ Model loaded successfully on {self.device}")
    
    def generate_response(self, prompt: str, max_tokens: int = 200) -> str:
        """Generate response from model"""
        
        # Format prompt for instruction-tuned models
        if "qwen" in self.model_name.lower():
            messages = [{"role": "user", "content": prompt}]
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        elif "deepseek" in self.model_name.lower():
            text = f"User: {prompt}\n\nAssistant:"
        else:
            text = prompt
        
        # Tokenize
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.1,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode only the new tokens
        response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )
        
        return response.strip()
    
    def cleanup(self):
        """Free GPU memory"""
        del self.model
        del self.tokenizer
        torch.cuda.empty_cache()
