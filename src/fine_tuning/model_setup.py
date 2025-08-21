#!/usr/bin/env python3
"""
Model Setup for Pro Roofing AI
Handles model configuration, loading, and optimization
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    BitsAndBytesConfig, PreTrainedTokenizer, PreTrainedModel
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
import accelerate

class ModelSetup:
    """Handle model setup and configuration for training"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Model components
        self.tokenizer: Optional[PreTrainedTokenizer] = None
        self.model: Optional[PreTrainedModel] = None
        self.device = self._get_device()
        
        self.logger.info(f"🔧 ModelSetup initialized with device: {self.device}")

    def _get_device(self) -> str:
        """Determine the best available device"""
        if torch.cuda.is_available():
            device = "cuda"
            self.logger.info(f"🚀 CUDA available: {torch.cuda.get_device_name(0)}")
            self.logger.info(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            device = "cpu"
            self.logger.warning("⚠️  CUDA not available, using CPU")
        
        return device

    def setup_tokenizer(self, model_name: Optional[str] = None) -> PreTrainedTokenizer:
        """Setup and configure tokenizer"""
        if model_name is None:
            model_name = self.config['model']['name']
        
        self.logger.info(f"🔤 Loading tokenizer: {model_name}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=self.config.get('advanced', {}).get('trust_remote_code', True),
                use_fast=self.config.get('advanced', {}).get('use_fast_tokenizer', True),
                cache_dir=self.config['model'].get('cache_dir')
            )
            
            # Configure padding token
            if self.tokenizer.pad_token is None:
                if self.tokenizer.eos_token is not None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                    self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
                    self.logger.info("🔧 Set pad_token to eos_token")
                else:
                    # Add a pad token if none exists
                    self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                    self.logger.info("🔧 Added new pad_token: [PAD]")
            
            # Set padding side for training
            self.tokenizer.padding_side = "right"
            
            # Log tokenizer info
            self.logger.info(f"✅ Tokenizer loaded successfully")
            self.logger.info(f"📊 Vocab size: {self.tokenizer.vocab_size}")
            self.logger.info(f"🔑 Special tokens: pad={self.tokenizer.pad_token}, eos={self.tokenizer.eos_token}")
            
            return self.tokenizer
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load tokenizer: {e}")
            raise

    def setup_quantization_config(self) -> Optional[BitsAndBytesConfig]:
        """Setup quantization configuration"""
        model_config = self.config['model']
        
        if model_config.get('load_in_4bit', False):
            self.logger.info("🔧 Setting up 4-bit quantization")
            
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_storage=torch.uint8
            )
            
            self.logger.info("✅ 4-bit quantization config created")
            return bnb_config
            
        elif model_config.get('load_in_8bit', False):
            self.logger.info("🔧 Setting up 8-bit quantization")
            
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False
            )
            
            self.logger.info("✅ 8-bit quantization config created")
            return bnb_config
        
        return None

    def setup_model(self, model_name: Optional[str] = None) -> PreTrainedModel:
        """Setup and configure the base model"""
        if model_name is None:
            model_name = self.config['model']['name']
        
        self.logger.info(f"🧠 Loading model: {model_name}")
        
        try:
            # Setup model loading arguments
            model_kwargs = {
                'trust_remote_code': self.config.get('advanced', {}).get('trust_remote_code', True),
                'cache_dir': self.config['model'].get('cache_dir'),
                'torch_dtype': getattr(torch, self.config['model'].get('torch_dtype', 'float16')),
                'device_map': self.config['model'].get('device_map', 'auto'),
                'low_cpu_mem_usage': True,
            }
            
            # Add quantization config if specified
            quantization_config = self.setup_quantization_config()
            if quantization_config is not None:
                model_kwargs['quantization_config'] = quantization_config
            
            # Load the model
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                **model_kwargs
            )
            
            # Prepare model for k-bit training if quantized
            if quantization_config is not None:
                self.logger.info("🔧 Preparing model for k-bit training")
                self.model = prepare_model_for_kbit_training(
                    self.model,
                    use_gradient_checkpointing=self.config['training'].get('gradient_checkpointing', True)
                )
            
            # Enable gradient checkpointing
            if self.config['training'].get('gradient_checkpointing', True):
                self.model.gradient_checkpointing_enable()
                self.logger.info("✅ Gradient checkpointing enabled")
            
            # Resize token embeddings if tokenizer was modified
            if self.tokenizer and len(self.tokenizer) != self.model.config.vocab_size:
                self.logger.info("🔧 Resizing token embeddings")
                self.model.resize_token_embeddings(len(self.tokenizer))
            
            # Log model info
            self.logger.info(f"✅ Model loaded successfully")
            self.logger.info(f"📊 Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
            self.logger.info(f"🎯 Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
            
            return self.model
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load model: {e}")
            raise

    def setup_lora(self) -> PreTrainedModel:
        """Setup LoRA (Low-Rank Adaptation) for efficient fine-tuning"""
        if 'lora' not in self.config:
            self.logger.info("⏭️  No LoRA configuration found, skipping LoRA setup")
            return self.model
        
        if self.model is None:
            raise ValueError("Model must be loaded before setting up LoRA")
        
        self.logger.info("🔧 Setting up LoRA configuration")
        
        lora_config = self.config['lora']
        
        # Create LoRA configuration
        peft_config = LoraConfig(
            r=lora_config['r'],
            lora_alpha=lora_config['alpha'],
            target_modules=lora_config['target_modules'],
            lora_dropout=lora_config['dropout'],
            bias=lora_config['bias'],
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
        )
        
        # Apply LoRA to model
        self.model = get_peft_model(self.model, peft_config)
        
        # Print trainable parameters
        self.model.print_trainable_parameters()
        
        self.logger.info("✅ LoRA setup completed")
        
        return self.model

    def setup_complete_model(self, model_name: Optional[str] = None) -> Tuple[PreTrainedTokenizer, PreTrainedModel]:
        """Setup complete model with tokenizer and optional LoRA"""
        self.logger.info("🚀 Starting complete model setup")
        
        # Setup tokenizer
        tokenizer = self.setup_tokenizer(model_name)
        
        # Setup model
        model = self.setup_model(model_name)
        
        # Setup LoRA if configured
        model = self.setup_lora()
        
        self.logger.info("🎉 Complete model setup finished")
        
        return tokenizer, model

    def save_model_config(self, output_dir: str):
        """Save model configuration for later inference"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save model configuration
        config_file = output_path / "model_setup_config.json"
        
        import json
        
        setup_config = {
            "model_name": self.config['model']['name'],
            "torch_dtype": self.config['model'].get('torch_dtype', 'float16'),
            "quantization": {
                "load_in_4bit": self.config['model'].get('load_in_4bit', False),
                "load_in_8bit": self.config['model'].get('load_in_8bit', False)
            },
            "lora_config": self.config.get('lora', {}),
            "device": self.device,
            "setup_timestamp": str(Path().stat().st_mtime)
        }
        
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(setup_config, f, indent=2)
        
        self.logger.info(f"💾 Model setup config saved to: {config_file}")

    def validate_model_setup(self) -> bool:
        """Validate that model and tokenizer are properly set up"""
        issues = []
        
        # Check tokenizer
        if self.tokenizer is None:
            issues.append("Tokenizer not loaded")
        else:
            if self.tokenizer.pad_token is None:
                issues.append("Tokenizer missing pad_token")
        
        # Check model
        if self.model is None:
            issues.append("Model not loaded")
        else:
            # Check if model is on correct device
            if hasattr(self.model, 'device'):
                if str(self.model.device) != self.device and not self.config['model'].get('device_map'):
                    issues.append(f"Model device mismatch: expected {self.device}, got {self.model.device}")
        
        # Check LoRA setup if configured
        if 'lora' in self.config and self.model is not None:
            if not hasattr(self.model, 'peft_config'):
                issues.append("LoRA configured but not applied to model")
        
        if issues:
            self.logger.error(f"❌ Model setup validation failed: {issues}")
            return False
        
        self.logger.info("✅ Model setup validation passed")
        return True

    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics"""
        memory_stats = {}
        
        if torch.cuda.is_available():
            memory_stats['gpu_allocated'] = torch.cuda.memory_allocated() / 1024**3  # GB
            memory_stats['gpu_reserved'] = torch.cuda.memory_reserved() / 1024**3    # GB
            memory_stats['gpu_total'] = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
            memory_stats['gpu_free'] = memory_stats['gpu_total'] - memory_stats['gpu_reserved']
        
        # CPU memory (approximate)
        import psutil
        process = psutil.Process()
        memory_stats['cpu_memory'] = process.memory_info().rss / 1024**3  # GB
        
        return memory_stats

    def optimize_model_for_inference(self):
        """Optimize model for inference (post-training)"""
        if self.model is None:
            raise ValueError("Model must be loaded before optimization")
        
        self.logger.info("🔧 Optimizing model for inference")
        
        # Disable gradient computation
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Enable torch.jit.script compilation if supported
        try:
            if self.config.get('advanced', {}).get('torch_compile', False):
                self.model = torch.compile(self.model)
                self.logger.info("✅ Torch compilation enabled")
        except Exception as e:
            self.logger.warning(f"⚠️  Torch compilation failed: {e}")
        
        self.logger.info("✅ Model optimization completed")

    def cleanup(self):
        """Clean up resources"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        if self.model is not None:
            del self.model
            self.model = None
        
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        
        self.logger.info("🧹 Model setup cleanup completed")


def main():
    """Test model setup"""
    import yaml
    
    # Load config
    with open("config/training_config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    # Test model setup
    model_setup = ModelSetup(config)
    
    try:
        tokenizer, model = model_setup.setup_complete_model()
        
        # Validate setup
        is_valid = model_setup.validate_model_setup()
        
        # Get memory usage
        memory_stats = model_setup.get_memory_usage()
        print(f"Memory usage: {memory_stats}")
        
        if is_valid:
            print("✅ Model setup test passed")
        else:
            print("❌ Model setup test failed")
            
    except Exception as e:
        print(f"❌ Model setup test failed with error: {e}")
    
    finally:
        model_setup.cleanup()


if __name__ == "__main__":
    main()