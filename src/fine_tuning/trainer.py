#!/usr/bin/env python3
"""
Main Training Script for Pro Roofing AI
Fine-tunes language models on roofing industry data
"""

import os
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
import yaml
import torch
import wandb
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    TrainingArguments, Trainer, DataCollatorForLanguageModeling,
    EarlyStoppingCallback
)
from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
import bitsandbytes as bnb

from ..data.enhanced_data_processor import EnhancedRoofingDataProcessor
from .model_setup import ModelSetup
from .training_manager import TrainingManager

class RoofingAITrainer:
    """Main trainer class for Pro Roofing AI models"""
    
    def __init__(self, config_path: str):
        """Initialize trainer with configuration"""
        self.config = self.load_config(config_path)
        self.setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.model_setup = ModelSetup(self.config)
        self.training_manager = TrainingManager(self.config)
        self.data_processor = EnhancedRoofingDataProcessor(self.config)
        
        # Model components
        self.tokenizer = None
        self.model = None
        self.train_dataset = None
        self.eval_dataset = None
        
        self.logger.info("🚀 Pro Roofing AI Trainer initialized")

    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load training configuration"""
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config

    def setup_logging(self):
        """Setup logging configuration"""
        log_level = self.config.get('logging', {}).get('level', 'INFO')
        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('logs/training/trainer.log')
            ]
        )

    def prepare_data(self) -> bool:
        """Prepare training and validation datasets"""
        self.logger.info("📊 Preparing training data...")
        
        try:
            # Check if processed data exists
            train_file = self.config['data']['train_file']
            if not os.path.exists(train_file):
                self.logger.info("📈 Processing raw datasets...")
                self.data_processor.process_all_datasets("data/raw")
            
            # Load datasets
            dataset = load_dataset('json', data_files=train_file, split='train')
            
            # Split into train and validation
            split_ratio = self.config['data'].get('validation_split_percentage', 10) / 100
            dataset = dataset.train_test_split(test_size=split_ratio, seed=42)
            
            self.train_dataset = dataset['train']
            self.eval_dataset = dataset['test']
            
            self.logger.info(f"✅ Training examples: {len(self.train_dataset)}")
            self.logger.info(f"✅ Validation examples: {len(self.eval_dataset)}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Data preparation failed: {e}")
            return False

    def setup_model_and_tokenizer(self) -> bool:
        """Setup model and tokenizer for training"""
        self.logger.info("🧠 Setting up model and tokenizer...")
        
        try:
            # Load tokenizer
            model_name = self.config['model']['name']
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=self.config.get('advanced', {}).get('trust_remote_code', True)
            )
            
            # Add pad token if missing
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
            # Load model with quantization
            model_kwargs = {
                'torch_dtype': getattr(torch, self.config['model'].get('torch_dtype', 'float16')),
                'device_map': self.config['model'].get('device_map', 'auto'),
                'trust_remote_code': self.config.get('advanced', {}).get('trust_remote_code', True)
            }
            
            # Add quantization config if specified
            if self.config['model'].get('load_in_4bit', False):
                from transformers import BitsAndBytesConfig
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True
                )
                model_kwargs['quantization_config'] = bnb_config
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                **model_kwargs
            )
            
            # Prepare model for k-bit training if quantized
            if self.config['model'].get('load_in_4bit', False) or self.config['model'].get('load_in_8bit', False):
                self.model = prepare_model_for_kbit_training(self.model)
            
            # Setup LoRA
            if 'lora' in self.config:
                lora_config = LoraConfig(
                    r=self.config['lora']['r'],
                    lora_alpha=self.config['lora']['alpha'],
                    target_modules=self.config['lora']['target_modules'],
                    lora_dropout=self.config['lora']['dropout'],
                    bias=self.config['lora']['bias'],
                    task_type=TaskType.CAUSAL_LM,
                )
                
                self.model = get_peft_model(self.model, lora_config)
                self.model.print_trainable_parameters()
            
            self.logger.info("✅ Model and tokenizer setup complete")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Model setup failed: {e}")
            return False

    def tokenize_data(self) -> bool:
        """Tokenize the datasets"""
        self.logger.info("🔤 Tokenizing datasets...")
        
        try:
            def tokenize_function(examples):
                # Extract text from conversation format
                texts = []
                for messages in examples['messages']:
                    if isinstance(messages, list):
                        # Convert conversation to single text
                        conversation_text = ""
                        for msg in messages:
                            role = msg.get('role', '')
                            content = msg.get('content', '')
                            conversation_text += f"<|{role}|>{content}<|end|>\n"
                        texts.append(conversation_text)
                    else:
                        texts.append(str(messages))
                
                # Tokenize
                tokenized = self.tokenizer(
                    texts,
                    truncation=True,
                    padding=False,
                    max_length=self.config['training']['max_seq_length'],
                    return_tensors=None
                )
                
                # Set labels for language modeling
                tokenized['labels'] = tokenized['input_ids'].copy()
                
                return tokenized
            
            # Tokenize datasets
            self.train_dataset = self.train_dataset.map(
                tokenize_function,
                batched=True,
                num_proc=self.config['data'].get('preprocessing_num_workers', 4),
                remove_columns=self.train_dataset.column_names
            )
            
            self.eval_dataset = self.eval_dataset.map(
                tokenize_function,
                batched=True,
                num_proc=self.config['data'].get('preprocessing_num_workers', 4),
                remove_columns=self.eval_dataset.column_names
            )
            
            self.logger.info("✅ Tokenization complete")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Tokenization failed: {e}")
            return False

    def setup_training_arguments(self) -> TrainingArguments:
        """Setup training arguments"""
        training_config = self.config['training']
        
        training_args = TrainingArguments(
            output_dir=training_config['output_dir'],
            overwrite_output_dir=training_config.get('overwrite_output_dir', True),
            num_train_epochs=training_config['num_train_epochs'],
            per_device_train_batch_size=training_config['per_device_train_batch_size'],
            per_device_eval_batch_size=training_config['per_device_eval_batch_size'],
            gradient_accumulation_steps=training_config['gradient_accumulation_steps'],
            gradient_checkpointing=training_config.get('gradient_checkpointing', True),
            learning_rate=training_config['learning_rate'],
            lr_scheduler_type=training_config.get('lr_scheduler_type', 'cosine'),
            warmup_ratio=training_config.get('warmup_ratio', 0.1),
            weight_decay=training_config.get('weight_decay', 0.01),
            optim=training_config.get('optim', 'paged_adamw_32bit'),
            max_grad_norm=training_config.get('max_grad_norm', 1.0),
            logging_steps=training_config.get('logging_steps', 10),
            save_steps=training_config['save_steps'],
            eval_steps=training_config['eval_steps'],
            save_total_limit=training_config.get('save_total_limit', 3),
            evaluation_strategy=training_config.get('evaluation_strategy', 'steps'),
            dataloader_num_workers=training_config.get('dataloader_num_workers', 4),
            remove_unused_columns=training_config.get('remove_unused_columns', False),
            fp16=training_config.get('fp16', False),
            bf16=training_config.get('bf16', True),
            tf32=training_config.get('tf32', True),
            dataloader_pin_memory=training_config.get('dataloader_pin_memory', True),
            group_by_length=training_config.get('group_by_length', True),
            report_to="wandb" if 'wandb' in self.config else None,
            run_name=self.config.get('wandb', {}).get('name', 'roofing-ai-training'),
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
        )
        
        return training_args

    def setup_trainer(self) -> Trainer:
        """Setup the Trainer object"""
        training_args = self.setup_training_arguments()
        
        # Data collator for language modeling
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # Causal language modeling
            pad_to_multiple_of=8 if training_args.fp16 else None
        )
        
        # Callbacks
        callbacks = []
        if 'early_stopping' in self.config:
            early_stopping = EarlyStoppingCallback(
                early_stopping_patience=self.config['early_stopping'].get('patience', 3),
                early_stopping_threshold=self.config['early_stopping'].get('threshold', 0.01)
            )
            callbacks.append(early_stopping)
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
            callbacks=callbacks,
        )
        
        return trainer

    def train(self) -> bool:
        """Main training loop"""
        self.logger.info("🎯 Starting training...")
        
        try:
            # Initialize wandb if configured
            if 'wandb' in self.config:
                wandb.init(
                    project=self.config['wandb']['project'],
                    name=self.config['wandb']['name'],
                    tags=self.config['wandb'].get('tags', []),
                    notes=self.config['wandb'].get('notes', ''),
                    config=self.config
                )
            
            # Setup trainer
            trainer = self.setup_trainer()
            
            # Check for resuming from checkpoint
            resume_from_checkpoint = self.config.get('checkpoint', {}).get('resume_from_checkpoint')
            
            # Start training
            train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
            
            # Save model
            trainer.save_model()
            trainer.save_state()
            
            # Log training results
            self.logger.info(f"✅ Training completed!")
            self.logger.info(f"📊 Training loss: {train_result.training_loss:.4f}")
            
            # Save final model if configured
            if self.config.get('export', {}).get('merge_and_unload', False):
                self.save_final_model(trainer)
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Training failed: {e}")
            return False
        finally:
            if 'wandb' in self.config:
                wandb.finish()

    def save_final_model(self, trainer: Trainer):
        """Save the final merged model"""
        self.logger.info("💾 Saving final model...")
        
        try:
            export_config = self.config.get('export', {})
            output_dir = export_config.get('output_dir', './models/final')
            
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            # Merge LoRA weights if using PEFT
            if hasattr(self.model, 'merge_and_unload'):
                merged_model = self.model.merge_and_unload()
                merged_model.save_pretrained(output_dir)
                self.tokenizer.save_pretrained(output_dir)
                self.logger.info(f"✅ Final model saved to {output_dir}")
            else:
                trainer.save_model(output_dir)
                self.logger.info(f"✅ Model saved to {output_dir}")
            
            # Push to hub if configured
            if export_config.get('push_to_hub', False):
                hub_model_id = export_config.get('hub_model_id')
                if hub_model_id:
                    trainer.push_to_hub(hub_model_id)
                    self.logger.info(f"✅ Model pushed to hub: {hub_model_id}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save final model: {e}")

    def run_full_pipeline(self) -> bool:
        """Run the complete training pipeline"""
        self.logger.info("🚀 Starting full training pipeline...")
        
        steps = [
            ("Preparing data", self.prepare_data),
            ("Setting up model", self.setup_model_and_tokenizer),
            ("Tokenizing data", self.tokenize_data),
            ("Training model", self.train),
        ]
        
        for step_name, step_func in steps:
            self.logger.info(f"📍 {step_name}...")
            if not step_func():
                self.logger.error(f"❌ Pipeline failed at: {step_name}")
                return False
        
        self.logger.info("🎉 Training pipeline completed successfully!")
        return True


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Pro Roofing AI Trainer")
    parser.add_argument(
        "--config", 
        type=str, 
        default="config/training_config.yaml",
        help="Path to training configuration file"
    )
    parser.add_argument(
        "--data-only", 
        action="store_true",
        help="Only process data without training"
    )
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = RoofingAITrainer(args.config)
    
    if args.data_only:
        # Only process data
        success = trainer.prepare_data()
        if success:
            print("✅ Data processing completed successfully")
        else:
            print("❌ Data processing failed")
    else:
        # Run full pipeline
        success = trainer.run_full_pipeline()
        if success:
            print("✅ Training completed successfully")
        else:
            print("❌ Training failed")


if __name__ == "__main__":
    main()