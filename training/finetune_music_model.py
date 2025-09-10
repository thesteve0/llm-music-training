#!/usr/bin/env python3
"""
Music Lyrics Understanding Fine-Tuning Script
High-performance LoRA fine-tuning for OpenShift AI deployment
Optimized for 48GB GPU with maximum VRAM utilization
"""

import os
import sys
import json
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional

import yaml
import torch
import pandas as pd
from datasets import Dataset, load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
import numpy as np
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('/shared/models/training.log') if os.path.exists('/shared') else logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def verify_dependencies():
    """Verify all required dependencies are available"""
    logger.info("Verifying dependencies...")
    
    required_packages = [
        'torch', 'transformers', 'peft', 'datasets', 'pandas', 
        'numpy', 'yaml', 'tqdm', 'bitsandbytes'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"Missing required packages: {missing_packages}")
        sys.exit(1)
    
    logger.info("All dependencies verified successfully")
    

def load_config(config_path: str = "/shared/code/training/config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file with environment variable overrides"""
    logger.info(f"Loading configuration from {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Environment variable overrides
    env_overrides = {
        'EPOCHS': ('training', 'num_train_epochs'),
        'BATCH_SIZE': ('training', 'per_device_train_batch_size'),
        'LEARNING_RATE': ('training', 'learning_rate'),
        'LORA_RANK': ('lora', 'r'),
        'DATA_DIR': ('environment', 'DATA_DIR'),
        'OUTPUT_DIR': ('environment', 'OUTPUT_DIR')
    }
    
    for env_var, (section, key) in env_overrides.items():
        if env_var in os.environ:
            value = os.environ[env_var]
            if env_var in ['EPOCHS', 'BATCH_SIZE', 'LORA_RANK']:
                value = int(value)
            elif env_var == 'LEARNING_RATE':
                value = float(value)
            
            config[section][key] = value
            logger.info(f"Override from {env_var}: {section}.{key} = {value}")
    
    return config


def download_and_process_dataset(config: Dict[str, Any]) -> Dataset:
    """Download and process the 5M Songs Lyrics dataset"""
    logger.info("Downloading and processing dataset...")
    
    dataset_config = config['dataset']
    data_processing_config = config['data_processing']
    
    # Download dataset
    dataset = load_dataset(
        dataset_config['name'], 
        split='train',
        streaming=False
    )
    
    logger.info(f"Downloaded dataset with {len(dataset)} total samples")
    
    # Process and filter dataset
    def clean_and_filter(batch):
        # Clean lyrics
        if data_processing_config['clean_lyrics']:
            # Basic cleaning - remove extra whitespace and normalize
            batch['text'] = batch['text'].strip() if batch['text'] else ""
        
        # Filter by length
        text_len = len(batch['text']) if batch['text'] else 0
        if (text_len < data_processing_config['min_lyric_length'] or 
            text_len > data_processing_config['max_lyric_length']):
            return None
            
        return batch
    
    # Apply filtering
    logger.info("Applying data cleaning and filtering...")
    dataset = dataset.filter(lambda x: clean_and_filter(x) is not None)
    
    # Remove duplicates if specified
    if data_processing_config['remove_duplicates']:
        logger.info("Removing duplicates...")
        dataset = dataset.unique('text')
    
    # Limit to specified number of samples
    num_samples = min(dataset_config['num_samples'], len(dataset))
    dataset = dataset.select(range(num_samples))
    
    logger.info(f"Processed dataset: {len(dataset)} samples")
    return dataset


def format_instruction_data(dataset: Dataset, config: Dict[str, Any]) -> Dataset:
    """Format data for instruction following"""
    logger.info("Formatting data for instruction following...")
    
    instruction_config = config['instruction_format']
    system_prompt = instruction_config['system_prompt']
    templates = instruction_config['instruction_templates']
    
    def format_sample(example):
        # Use a random template for variety
        template = np.random.choice(templates)
        
        # Create instruction-following format
        formatted_text = f"<|system|>\n{system_prompt}\n\n<|user|>\n{template}\n\n{example['text']}\n\n<|assistant|>\n"
        
        # For training, we need the full conversation including a response
        # Since we don't have responses in the dataset, we'll use the lyrics as both input and target
        formatted_text += f"These lyrics demonstrate {example.get('artist', 'an artist')} style"
        if 'genre' in example:
            formatted_text += f" in the {example['genre']} genre"
        formatted_text += "."
        
        return {'text': formatted_text}
    
    dataset = dataset.map(format_sample, remove_columns=dataset.column_names)
    return dataset


def setup_model_and_tokenizer(config: Dict[str, Any]):
    """Setup model, tokenizer, and LoRA configuration"""
    logger.info("Setting up model and tokenizer...")
    
    model_config = config['model']
    lora_config = config['lora']
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_config['name'],
        trust_remote_code=model_config['trust_remote_code']
    )
    
    # Add pad token if missing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_config['name'],
        torch_dtype=getattr(torch, model_config['torch_dtype']),
        device_map=model_config['device_map'],
        trust_remote_code=model_config['trust_remote_code']
    )
    
    # Configure LoRA
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=lora_config['inference_mode'],
        r=lora_config['r'],
        lora_alpha=lora_config['lora_alpha'],
        target_modules=lora_config['target_modules'],
        lora_dropout=lora_config['lora_dropout'],
        bias=lora_config['bias']
    )
    
    # Apply LoRA
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    logger.info("Model and LoRA configuration completed")
    return model, tokenizer


def tokenize_dataset(dataset: Dataset, tokenizer, config: Dict[str, Any]) -> Dataset:
    """Tokenize the dataset"""
    logger.info("Tokenizing dataset...")
    
    max_length = config['dataset']['max_length']
    
    def tokenize_function(examples):
        # Tokenize the text
        tokenized = tokenizer(
            examples['text'],
            truncation=True,
            padding=False,
            max_length=max_length,
            return_tensors=None
        )
        
        # For causal LM, labels are the same as input_ids
        tokenized['labels'] = tokenized['input_ids'].copy()
        
        return tokenized
    
    # Tokenize dataset
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing dataset"
    )
    
    logger.info(f"Tokenization completed: {len(tokenized_dataset)} samples")
    return tokenized_dataset


def create_data_splits(dataset: Dataset, config: Dict[str, Any]):
    """Create train/eval splits"""
    dataset_config = config['dataset']
    
    # Split dataset
    train_size = dataset_config['train_split']
    split_dataset = dataset.train_test_split(
        test_size=1-train_size,
        seed=dataset_config['shuffle_seed']
    )
    
    train_dataset = split_dataset['train']
    eval_dataset = split_dataset['test']
    
    logger.info(f"Data splits - Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")
    return train_dataset, eval_dataset


def setup_training_arguments(config: Dict[str, Any]) -> TrainingArguments:
    """Setup training arguments"""
    training_config = config['training']
    
    # Determine output directory
    output_dir = os.environ.get('OUTPUT_DIR', training_config['output_dir'])
    
    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=training_config['num_train_epochs'],
        per_device_train_batch_size=training_config['per_device_train_batch_size'],
        per_device_eval_batch_size=training_config['per_device_eval_batch_size'],
        gradient_accumulation_steps=training_config['gradient_accumulation_steps'],
        learning_rate=training_config['learning_rate'],
        warmup_steps=training_config['warmup_steps'],
        logging_steps=training_config['logging_steps'],
        eval_steps=training_config['eval_steps'],
        save_steps=training_config['save_steps'],
        evaluation_strategy=training_config['evaluation_strategy'],
        fp16=training_config['fp16'],
        dataloader_pin_memory=training_config['dataloader_pin_memory'],
        dataloader_num_workers=training_config['dataloader_num_workers'],
        remove_unused_columns=training_config['remove_unused_columns'],
        load_best_model_at_end=training_config['load_best_model_at_end'],
        metric_for_best_model=training_config['metric_for_best_model'],
        greater_is_better=training_config['greater_is_better'],
        max_grad_norm=training_config['max_grad_norm'],
        lr_scheduler_type=training_config['lr_scheduler_type'],
        save_total_limit=training_config['save_total_limit'],
        seed=training_config['seed'],
        report_to=None,  # Disable wandb/tensorboard for now
    )
    
    return args


def main():
    """Main training function"""
    logger.info("Starting Music Lyrics Fine-tuning Training")
    start_time = time.time()
    
    # Verify dependencies
    verify_dependencies()
    
    # Load configuration
    config = load_config()
    
    # Download and process dataset
    raw_dataset = download_and_process_dataset(config)
    
    # Format for instruction following
    formatted_dataset = format_instruction_data(raw_dataset, config)
    
    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(config)
    
    # Tokenize dataset
    tokenized_dataset = tokenize_dataset(formatted_dataset, tokenizer, config)
    
    # Create data splits
    train_dataset, eval_dataset = create_data_splits(tokenized_dataset, config)
    
    # Setup training arguments
    training_args = setup_training_arguments(config)
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM, not masked LM
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # Start training
    logger.info("Starting training...")
    train_result = trainer.train()
    
    # Save the model
    logger.info("Saving model...")
    trainer.save_model()
    trainer.save_state()
    
    # Save training summary
    output_dir = training_args.output_dir
    training_summary = {
        'config': config,
        'train_result': train_result.metrics,
        'training_time': time.time() - start_time,
        'model_name': config['model']['name'],
        'dataset_size': len(train_dataset),
        'eval_size': len(eval_dataset)
    }
    
    with open(os.path.join(output_dir, 'training_summary.json'), 'w') as f:
        json.dump(training_summary, f, indent=2, default=str)
    
    total_time = time.time() - start_time
    logger.info(f"Training completed successfully in {total_time:.2f} seconds")
    logger.info(f"Final train loss: {train_result.metrics.get('train_loss', 'N/A')}")
    
    return trainer, training_summary


if __name__ == "__main__":
    trainer, summary = main()