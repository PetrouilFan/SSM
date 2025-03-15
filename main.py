"""
Main script for training, evaluating, and running inference with Sparse SSM models.

This script provides a command-line interface for training, evaluating,
and generating text with Sparse SSM models.
"""

import argparse
import os
import json
import yaml
import logging
import torch
import numpy as np
import random
from typing import Dict, Any, Optional

from src.model.mamba import SparseSSM
from src.dataset.dataloader import (
    load_wikitext103,
    load_pile,
    load_synthetic_dataset,
    load_custom_dataset,
    create_dataloaders
)
from src.dataset.tokenizer import SimpleTokenizer, train_tokenizer_from_files
from src.training.trainer import SSMTrainer
from src.dataset.synthetic_data import OllamaClient, SyntheticDataGenerator, get_default_domain_prompts


def setup_logging(output_dir: str, level: int = logging.INFO) -> logging.Logger:
    """
    Set up logging for the script.
    
    Args:
        output_dir: Directory to save log file
        level: Logging level
        
    Returns:
        Logger instance
    """
    os.makedirs(output_dir, exist_ok=True)
    
    logger = logging.getLogger()
    logger.setLevel(level)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    
    # Create file handler
    file_handler = logging.FileHandler(os.path.join(output_dir, 'run.log'))
    file_handler.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger


def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML or JSON file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    if config_path.endswith('.yaml') or config_path.endswith('.yml'):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    elif config_path.endswith('.json'):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
    else:
        raise ValueError(f"Unsupported configuration file format: {config_path}")
    
    return config


def save_config(config: Dict[str, Any], output_path: str) -> None:
    """
    Save configuration to a file.
    
    Args:
        config: Configuration dictionary
        output_path: Path to save the configuration
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if output_path.endswith('.yaml') or output_path.endswith('.yml'):
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False)
    elif output_path.endswith('.json'):
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
    else:
        raise ValueError(f"Unsupported configuration file format: {output_path}")


def create_model(config: Dict[str, Any], device: Optional[torch.device] = None) -> SparseSSM:
    """
    Create a Sparse SSM model based on configuration.
    
    Args:
        config: Model configuration
        device: Device to place the model on
        
    Returns:
        SparseSSM model
    """
    model = SparseSSM(
        vocab_size=config.get('vocab_size', 50000),
        hidden_dim=config.get('hidden_dim', 768),
        state_dim=config.get('state_dim', 128),
        ffn_dim=config.get('ffn_dim', 2048),
        num_layers=config.get('num_layers', 12),
        max_seq_len=config.get('max_seq_len', 1024),
        dropout=config.get('dropout', 0.1),
        activation=config.get('activation', 'gelu'),
        layer_norm_eps=config.get('layer_norm_eps', 1e-5),
        selective_param_class=config.get('selective_param_class', 'sparse'),
        sparsity_level=config.get('sparsity_level', 0.9),
        use_parallel_scan=config.get('use_parallel_scan', True),
        tie_embedding_weights=config.get('tie_embedding_weights', True),
        pad_token_id=config.get('pad_token_id', 0)
    )
    
    if device is not None:
        model = model.to(device)
    
    return model


def load_or_train_tokenizer(
    config: Dict[str, Any],
    output_dir: str,
    logger: logging.Logger
) -> SimpleTokenizer:
    """
    Load an existing tokenizer or train a new one.
    
    Args:
        config: Tokenizer configuration
        output_dir: Directory to save/load the tokenizer
        logger: Logger instance
        
    Returns:
        SimpleTokenizer instance
    """
    tokenizer_dir = os.path.join(output_dir, 'tokenizer')
    
    # Check if tokenizer already exists
    tokenizer_config_path = os.path.join(tokenizer_dir, 'tokenizer_config.json')
    logger.info(f"Checking for tokenizer at {tokenizer_config_path}")
    
    if os.path.exists(tokenizer_config_path):
        logger.info(f"Loading existing tokenizer from {tokenizer_dir}")
        try:
            tokenizer = SimpleTokenizer.from_pretrained(tokenizer_dir)
            logger.info(f"Successfully loaded tokenizer with {len(tokenizer.token_to_id)} tokens")
        except Exception as e:
            logger.error(f"Error loading tokenizer: {str(e)}")
            # If there's an error loading the tokenizer, train a new one
            logger.info("Will train a new tokenizer instead")
            os.makedirs(tokenizer_dir, exist_ok=True)
            # Continue to train a new tokenizer below
    elif 'train' in config and config['train'].get('enabled', False):
        # Train a new tokenizer
        train_config = config['train']
        file_paths = train_config.get('files', [])
        vocab_size = train_config.get('vocab_size', 50000)
        min_freq = train_config.get('min_freq', 5)
        sample_size = train_config.get('sample_size', None)
        
        logger.info(f"Training new tokenizer with vocab_size={vocab_size}, min_freq={min_freq}")
        text_key = train_config.get('text_key', 'text')
        tokenizer = train_tokenizer_from_files(
            file_paths=file_paths,
            vocab_size=vocab_size,
            min_freq=min_freq,
            output_dir=tokenizer_dir,
            sample_size=sample_size,
            text_key=text_key
        )
    else:
        # Use a default tokenizer
        logger.info("Using default tokenizer")
        tokenizer = SimpleTokenizer(
            vocab_size=config.get('vocab_size', 50000),
            pad_token=config.get('pad_token', '[PAD]'),
            unk_token=config.get('unk_token', '[UNK]'),
            bos_token=config.get('bos_token', '[BOS]'),
            eos_token=config.get('eos_token', '[EOS]')
        )
        os.makedirs(tokenizer_dir, exist_ok=True)
        tokenizer.save_pretrained(tokenizer_dir)
    
    return tokenizer


def load_datasets(
    config: Dict[str, Any],
    tokenizer: SimpleTokenizer,
    logger: logging.Logger
) -> Dict[str, torch.utils.data.Dataset]:
    """
    Load datasets based on configuration.
    
    Args:
        config: Dataset configuration
        tokenizer: Tokenizer for text processing
        logger: Logger instance
        
    Returns:
        Dictionary of datasets
    """
    dataset_type = config.get('type', 'wikitext103')
    max_seq_len = config.get('max_seq_len', 1024)
    stride = config.get('stride', None)
    max_examples = config.get('max_examples', None)
    data_dir = config.get('data_dir', 'data')
    
    logger.info(f"Loading {dataset_type} dataset with max_seq_len={max_seq_len}")
    
    if dataset_type == 'wikitext103':
        datasets = load_wikitext103(
            tokenizer=tokenizer,
            data_dir=data_dir,
            max_seq_len=max_seq_len,
            stride=stride,
            max_examples=max_examples
        )
    elif dataset_type == 'pile':
        # Load The Pile datasets for train, val, and test
        datasets = {
            'train': load_pile(
                tokenizer=tokenizer,
                data_dir=data_dir,
                max_seq_len=max_seq_len,
                stride=stride,
                max_examples=max_examples,
                split='train'
            ),
            'validation': load_pile(
                tokenizer=tokenizer,
                data_dir=data_dir,
                max_seq_len=max_seq_len,
                stride=stride,
                max_examples=max_examples,
                split='val'
            ),
            'test': load_pile(
                tokenizer=tokenizer,
                data_dir=data_dir,
                max_seq_len=max_seq_len,
                stride=stride,
                max_examples=max_examples,
                split='test'
            )
        }
    elif dataset_type == 'synthetic':
        # Load synthetic dataset
        dataset = load_synthetic_dataset(
            tokenizer=tokenizer,
            data_path=config.get('data_path', 'data/synthetic/phi4_dataset.json'),
            max_seq_len=max_seq_len,
            stride=stride,
            max_examples=max_examples,
            text_key=config.get('text_key', 'text')
        )
        
        # Create train/val/test splits
        batch_size = config.get('batch_size', 16)
        val_split = config.get('val_split', 0.1)
        test_split = config.get('test_split', 0.1)
        
        dataloaders = create_dataloaders(
            dataset=dataset,
            batch_size=batch_size,
            val_split=val_split,
            test_split=test_split,
            shuffle=True
        )
        
        # Extract datasets from dataloaders
        datasets = {
            split: loader.dataset
            for split, loader in dataloaders.items()
        }
    elif dataset_type == 'custom':
        # Load custom dataset
        dataset = load_custom_dataset(
            tokenizer=tokenizer,
            data_dir=data_dir,
            max_seq_len=max_seq_len,
            stride=stride,
            max_examples=max_examples,
            file_pattern=config.get('file_pattern', '*.txt'),
            is_json=config.get('is_json', False),
            text_key=config.get('text_key', 'text')
        )
        
        # Create train/val/test splits
        batch_size = config.get('batch_size', 16)
        val_split = config.get('val_split', 0.1)
        test_split = config.get('test_split', 0.1)
        
        dataloaders = create_dataloaders(
            dataset=dataset,
            batch_size=batch_size,
            val_split=val_split,
            test_split=test_split,
            shuffle=True
        )
        
        # Extract datasets from dataloaders
        datasets = {
            split: loader.dataset
            for split, loader in dataloaders.items()
        }
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")
    
    return datasets


def create_dataloaders_from_config(
    datasets: Dict[str, torch.utils.data.Dataset],
    config: Dict[str, Any]
) -> Dict[str, torch.utils.data.DataLoader]:
    """
    Create dataloaders from datasets based on configuration.
    
    Args:
        datasets: Dictionary of datasets
        config: Dataloader configuration
        
    Returns:
        Dictionary of dataloaders
    """
    batch_size = config.get('batch_size', 16)
    num_workers = config.get('num_workers', 4)
    shuffle = config.get('shuffle', True)
    
    dataloaders = {}
    
    for split, dataset in datasets.items():
        dataloaders[split] = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle if split == 'train' else False,
            num_workers=num_workers,
            pin_memory=True
        )
    
    return dataloaders


def train_model(args: argparse.Namespace) -> None:
    """
    Train a Sparse SSM model.
    
    Args:
        args: Command-line arguments
    """
    # Load configuration
    config = load_config(args.config)
    output_dir = args.output_dir or config.get('output_dir', 'outputs')
    
    # Set up logging
    logger = setup_logging(output_dir)
    logger.info(f"Training Sparse SSM model with config: {args.config}")
    
    # Set random seed
    seed = args.seed or config.get('seed', 42)
    set_seed(seed)
    logger.info(f"Random seed: {seed}")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    
    # Save configuration
    save_config(config, os.path.join(output_dir, 'config.yaml'))
    
    # Load or train tokenizer
    tokenizer = load_or_train_tokenizer(
        config=config.get('tokenizer', {}),
        output_dir=output_dir,
        logger=logger
    )
    
    # Load datasets
    datasets = load_datasets(
        config=config.get('dataset', {}),
        tokenizer=tokenizer,
        logger=logger
    )
    
    # Create dataloaders
    dataloaders = create_dataloaders_from_config(
        datasets=datasets,
        config=config.get('dataloader', {})
    )
    
    # Create model
    model_config = config.get('model', {})
    model_config['vocab_size'] = len(tokenizer.token_to_id)
    model = create_model(model_config, device=device)
    
    # Create trainer
    trainer = SSMTrainer(
        model=model,
        train_dataloader=dataloaders['train'],
        eval_dataloader=dataloaders.get('validation'),
        optimizer_config=config.get('optimizer', {}),
        scheduler_config=config.get('scheduler', {}),
        training_config=config.get('training', {}),
        device=device,
        output_dir=output_dir
    )
    
    # Train model
    logger.info("Starting training")
    trainer.train()
    
    # Evaluate on test set if available
    if 'test' in dataloaders:
        logger.info("Evaluating on test set")
        model.eval()
        
        test_dataloader = dataloaders['test']
        
        # Set model to use for evaluation
        if args.use_best_model:
            best_model_path = os.path.join(output_dir, 'checkpoints', 'best')
            logger.info(f"Loading best model from {best_model_path}")
            trainer.load_checkpoint(best_model_path)
        
        # Evaluate
        eval_results = trainer.evaluate()
        logger.info(f"Test results: {eval_results}")


def generate_text(args: argparse.Namespace) -> None:
    """
    Generate text using a trained Sparse SSM model.
    
    Args:
        args: Command-line arguments
    """
    # Load configuration
    config = load_config(args.config)
    model_dir = args.model_dir or config.get('model_dir', 'outputs')
    
    # Set up logging
    logger = setup_logging(model_dir)
    logger.info(f"Generating text with model from: {model_dir}")
    
    # Set random seed
    seed = args.seed or config.get('seed', 42)
    set_seed(seed)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load tokenizer
    tokenizer_dir = os.path.join(model_dir, 'tokenizer')
    logger.info(f"Loading tokenizer from {tokenizer_dir}")
    tokenizer = SimpleTokenizer.from_pretrained(tokenizer_dir)
    
    # Load model configuration
    model_config_path = os.path.join(model_dir, 'config.yaml')
    model_config = load_config(model_config_path).get('model', {})
    model_config['vocab_size'] = len(tokenizer.token_to_id)
    
    # Create model
    model = create_model(model_config, device=device)
    
    # Load model weights
    checkpoint_path = os.path.join(model_dir, 'checkpoints', args.checkpoint)
    logger.info(f"Loading model weights from {checkpoint_path}")
    
    if os.path.exists(os.path.join(checkpoint_path, 'pytorch_model.bin')):
        model.load_state_dict(torch.load(
            os.path.join(checkpoint_path, 'pytorch_model.bin'),
            map_location=device
        ))
    else:
        raise ValueError(f"Model weights not found at {checkpoint_path}")
    
    # Set model to evaluation mode
    model.eval()
    
    # Get generation parameters
    max_length = args.max_length or config.get('generation', {}).get('max_length', 100)
    temperature = args.temperature or config.get('generation', {}).get('temperature', 0.7)
    top_k = args.top_k or config.get('generation', {}).get('top_k', 50)
    top_p = args.top_p or config.get('generation', {}).get('top_p', 0.9)
    
    # Process prompt
    logger.info(f"Generating text with prompt: {args.prompt}")
    
    # Tokenize prompt
    input_ids = tokenizer.encode(
        args.prompt,
        add_special_tokens=True,
        truncation=True,
        max_length=model.max_seq_len // 2  # Leave room for generation
    )['input_ids']
    
    input_ids = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)
    
    # Generate text
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=True
        )
    
    # Decode generated text
    generated_text = tokenizer.decode(output_ids[0].tolist(), skip_special_tokens=True)
    
    print("\nGenerated text:")
    print("=" * 40)
    print(generated_text)
    print("=" * 40)
    
    # Save generated text if output file is specified
    if args.output_file:
        with open(args.output_file, 'w', encoding='utf-8') as f:
            f.write(generated_text)
        logger.info(f"Generated text saved to {args.output_file}")


def create_synthetic_dataset(args: argparse.Namespace) -> None:
    """
    Create a synthetic dataset using Ollama phi4-mini model.
    
    Args:
        args: Command-line arguments
    """
    # Load configuration
    config = load_config(args.config)
    output_dir = args.output_dir or config.get('output_dir', 'data/synthetic')
    
    # Set up logging
    logger = setup_logging(output_dir)
    logger.info(f"Creating synthetic dataset with config: {args.config}")
    
    # Set random seed
    seed = args.seed or config.get('seed', 42)
    set_seed(seed)
    
    # Get generation parameters
    num_samples = args.num_samples or config.get('num_samples', 1000)
    temperature = args.temperature or config.get('temperature', 0.7)
    max_tokens = args.max_tokens or config.get('max_tokens', 512)
    batch_size = config.get('batch_size', 1)
    
    # Create Ollama client
    ollama_config = config.get('ollama', {})
    base_url = ollama_config.get('base_url', 'http://10.0.3.1:11434')
    model_name = ollama_config.get('model_name', 'phi4-mini:latest')
    
    logger.info(f"Connecting to Ollama API at {base_url} with model {model_name}")
    ollama_client = OllamaClient(base_url=base_url, model_name=model_name, logger=logger)
    
    # Create synthetic data generator
    generator = SyntheticDataGenerator(
        ollama_client=ollama_client,
        output_dir=output_dir,
        logger=logger
    )
    
    # Load domain prompts
    domain_prompts_path = args.domain_prompts or config.get('domain_prompts_path')
    
    if domain_prompts_path and os.path.exists(domain_prompts_path):
        logger.info(f"Loading domain prompts from {domain_prompts_path}")
        with open(domain_prompts_path, 'r', encoding='utf-8') as f:
            domain_prompts = json.load(f)
    else:
        logger.info("Using default domain prompts")
        domain_prompts = get_default_domain_prompts()
    
    # Generate dataset
    output_path = os.path.join(output_dir, args.output_file or 'phi4_dataset.json')
    
    logger.info(f"Generating {num_samples} samples with {len(domain_prompts)} prompt types")
    
    if args.distillation:
        # Generate knowledge distillation dataset
        generator.generate_knowledge_distillation_dataset(
            num_samples=num_samples,
            domain_prompts=domain_prompts,
            output_path=output_path,
            temperature=temperature,
            max_tokens=max_tokens,
            include_logits=True
        )
    else:
        # Generate standard dataset
        generator.generate_dataset(
            num_samples=num_samples,
            domain_prompts=domain_prompts,
            output_path=output_path,
            temperature=temperature,
            max_tokens=max_tokens,
            batch_size=batch_size
        )
    
    logger.info(f"Synthetic dataset created and saved to {output_path}")


def main():
    """Main function for the script."""
    parser = argparse.ArgumentParser(description="Sparse SSM model training and inference")
    subparsers = parser.add_subparsers(dest='command', help="Command to run")
    
    # Train command
    train_parser = subparsers.add_parser('train', help="Train a Sparse SSM model")
    train_parser.add_argument('--config', type=str, required=True, help="Path to configuration file")
    train_parser.add_argument('--output-dir', type=str, help="Directory to save outputs")
    train_parser.add_argument('--seed', type=int, help="Random seed")
    train_parser.add_argument('--use-best-model', action='store_true', help="Use best model for evaluation")
    
    # Generate command
    generate_parser = subparsers.add_parser('generate', help="Generate text with a trained model")
    generate_parser.add_argument('--config', type=str, required=True, help="Path to configuration file")
    generate_parser.add_argument('--model-dir', type=str, help="Directory containing the model")
    generate_parser.add_argument('--checkpoint', type=str, default='best', help="Checkpoint to use ('best', 'final', or specific step)")
    generate_parser.add_argument('--prompt', type=str, required=True, help="Text prompt for generation")
    generate_parser.add_argument('--max-length', type=int, help="Maximum length to generate")
    generate_parser.add_argument('--temperature', type=float, help="Sampling temperature")
    generate_parser.add_argument('--top-k', type=int, help="Top-k sampling parameter")
    generate_parser.add_argument('--top-p', type=float, help="Top-p sampling parameter")
    generate_parser.add_argument('--seed', type=int, help="Random seed")
    generate_parser.add_argument('--output-file', type=str, help="File to save generated text")
    
    # Create synthetic dataset command
    synthetic_parser = subparsers.add_parser('create-synthetic', help="Create a synthetic dataset")
    synthetic_parser.add_argument('--config', type=str, required=True, help="Path to configuration file")
    synthetic_parser.add_argument('--output-dir', type=str, help="Directory to save dataset")
    synthetic_parser.add_argument('--output-file', type=str, help="Output file name")
    synthetic_parser.add_argument('--num-samples', type=int, help="Number of samples to generate")
    synthetic_parser.add_argument('--temperature', type=float, help="Sampling temperature")
    synthetic_parser.add_argument('--max-tokens', type=int, help="Maximum tokens per sample")
    synthetic_parser.add_argument('--domain-prompts', type=str, help="Path to domain prompts file")
    synthetic_parser.add_argument('--distillation', action='store_true', help="Generate knowledge distillation dataset")
    synthetic_parser.add_argument('--seed', type=int, help="Random seed")
    
    args = parser.parse_args()
    
    if args.command == 'train':
        train_model(args)
    elif args.command == 'generate':
        generate_text(args)
    elif args.command == 'create-synthetic':
        create_synthetic_dataset(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
