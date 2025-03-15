"""
Data loaders for Sparse SSM models.

This module implements data loaders for various language modeling datasets,
including WikiText-103, Long Range Arena, The Pile, and custom datasets.
"""

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from typing import Dict, List, Optional, Union, Tuple, Any
import json
import os
import numpy as np
from tqdm import tqdm
import logging


class TextDataset(Dataset):
    """
    Base text dataset for language modeling.
    
    This dataset handles tokenized text data for next-token prediction tasks.
    """
    
    def __init__(
        self,
        examples: List[Dict[str, torch.Tensor]],
        pad_token_id: int = 0,
    ):
        """
        Initialize the text dataset.
        
        Args:
            examples: List of tokenized examples
            pad_token_id: Token ID for padding
        """
        self.examples = examples
        self.pad_token_id = pad_token_id
    
    def __len__(self) -> int:
        """Get the number of examples in the dataset."""
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get an example from the dataset.
        
        Args:
            idx: Index of the example
            
        Returns:
            Example as a dictionary of tensors
        """
        return self.examples[idx]


class LMDatasetBuilder:
    """
    Builder for language modeling datasets.
    
    This class provides methods for building language modeling datasets
    from tokenized text.
    """
    
    def __init__(
        self,
        tokenizer: Any,
        max_seq_len: int = 1024,
        pad_token_id: int = 0,
        stride: Optional[int] = None
    ):
        """
        Initialize the dataset builder.
        
        Args:
            tokenizer: Tokenizer to use
            max_seq_len: Maximum sequence length
            pad_token_id: Token ID for padding
            stride: Stride for sliding window tokenization (if None, no overlapping)
        """
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id
        self.stride = stride if stride is not None else max_seq_len
        
        self.logger = logging.getLogger(__name__)
    
    def build_from_text_files(
        self,
        file_paths: List[str],
        max_examples: Optional[int] = None,
        skip_incomplete: bool = False
    ) -> TextDataset:
        """
        Build a dataset from text files.
        
        Args:
            file_paths: List of paths to text files
            max_examples: Maximum number of examples to include
            skip_incomplete: Whether to skip incomplete sequences
            
        Returns:
            TextDataset instance
        """
        examples = []
        total_examples = 0
        
        for file_path in tqdm(file_paths, desc="Processing files"):
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
                # Tokenize the text
                try:
                    # Debug the call to tokenizer
                    self.logger.info(f"Calling tokenizer on text of length {len(text)}")
                    encodings = self.tokenizer(
                        text,
                        truncation=False,
                        return_attention_mask=True
                    )
                    
                    # Check what shape we're getting back
                    if isinstance(encodings['input_ids'], list):
                        if isinstance(encodings['input_ids'][0], list):
                            input_ids = encodings['input_ids'][0]  # If it's a batch, take the first item
                            self.logger.info(f"Got input_ids list of lists with shape: {len(encodings['input_ids'])}x{len(input_ids)}")
                        else:
                            input_ids = encodings['input_ids']
                            self.logger.info(f"Got flat input_ids list of length: {len(input_ids)}")
                    else:
                        input_ids = encodings['input_ids']
                        self.logger.info(f"Got input_ids tensor of shape: {input_ids.shape}")
                    
                    # Get attention mask
                    if isinstance(encodings.get('attention_mask'), list):
                        if isinstance(encodings['attention_mask'][0], list):
                            attention_mask = encodings['attention_mask'][0]
                        else:
                            attention_mask = encodings['attention_mask']
                    else:
                        attention_mask = encodings.get('attention_mask')
                    
                    if len(input_ids) <= 1:  # Just checking if we have enough tokens
                        self.logger.warning(f"Text too short after tokenization: {len(input_ids)} tokens")
                        continue
                except Exception as e:
                    self.logger.error(f"Error tokenizing text: {str(e)}")
                    continue
            
            # Create examples using sliding window
            for i in range(0, len(input_ids) - 1, self.stride):
                end_idx = min(i + self.max_seq_len, len(input_ids) - 1)
                
                # Skip incomplete sequences if requested
                if skip_incomplete and end_idx - i < self.max_seq_len:
                    continue
                
                # Extract input sequence
                input_seq = input_ids[i:end_idx]
                
                # Get corresponding attention mask if available
                if attention_mask is not None:
                    mask_seq = attention_mask[i:end_idx]
                else:
                    mask_seq = [1] * len(input_seq)
                
                # Create labels (shifted by 1 for next-token prediction)
                label_seq = input_ids[i+1:end_idx+1]
                
                # Pad sequences if needed
                if len(input_seq) < self.max_seq_len:
                    padding_len = self.max_seq_len - len(input_seq)
                    input_seq = input_seq + [self.pad_token_id] * padding_len
                    mask_seq = mask_seq + [0] * padding_len
                    label_seq = label_seq + [self.pad_token_id] * padding_len
                
                # Create example
                example = {
                    'input_ids': torch.tensor(input_seq, dtype=torch.long),
                    'attention_mask': torch.tensor(mask_seq, dtype=torch.long),
                    'labels': torch.tensor(label_seq, dtype=torch.long)
                }
                
                examples.append(example)
                total_examples += 1
                
                # Break if we have enough examples
                if max_examples is not None and total_examples >= max_examples:
                    break
            
            # Break if we have enough examples
            if max_examples is not None and total_examples >= max_examples:
                break
        
        self.logger.info(f"Created {len(examples)} examples from {len(file_paths)} files")
        
        return TextDataset(examples, pad_token_id=self.pad_token_id)
    
    def build_from_json_files(
        self,
        file_paths: List[str],
        text_key: str = 'text',
        max_examples: Optional[int] = None,
        skip_incomplete: bool = False
    ) -> TextDataset:
        """
        Build a dataset from JSON files.
        
        Args:
            file_paths: List of paths to JSON files
            text_key: Key for extracting text from JSON objects
            max_examples: Maximum number of examples to include
            skip_incomplete: Whether to skip incomplete sequences
            
        Returns:
            TextDataset instance
        """
        examples = []
        total_examples = 0
        
        for file_path in tqdm(file_paths, desc="Processing JSON files"):
            with open(file_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            
            # Extract texts from JSON data
            texts = []
            if isinstance(json_data, list):
                # List of objects
                for item in json_data:
                    if text_key in item:
                        texts.append(item[text_key])
            elif isinstance(json_data, dict):
                # Single object or dictionary of objects
                if text_key in json_data:
                    texts.append(json_data[text_key])
                else:
                    # Try to extract from values if they are dictionaries
                    for value in json_data.values():
                        if isinstance(value, dict) and text_key in value:
                            texts.append(value[text_key])
            
            # Process each text
            for text in texts:
                # Tokenize the text
                try:
                    # Debug the call to tokenizer
                    self.logger.info(f"Calling tokenizer on text of length {len(text)}")
                    encodings = self.tokenizer(
                        text,
                        truncation=False,
                        return_attention_mask=True
                    )
                    
                    # Check what shape we're getting back
                    if isinstance(encodings['input_ids'], list):
                        if isinstance(encodings['input_ids'][0], list):
                            input_ids = encodings['input_ids'][0]  # If it's a batch, take the first item
                            self.logger.info(f"Got input_ids list of lists with shape: {len(encodings['input_ids'])}x{len(input_ids)}")
                        else:
                            input_ids = encodings['input_ids']
                            self.logger.info(f"Got flat input_ids list of length: {len(input_ids)}")
                    else:
                        input_ids = encodings['input_ids']
                        self.logger.info(f"Got input_ids tensor of shape: {input_ids.shape}")
                    
                    # Get attention mask
                    if isinstance(encodings.get('attention_mask'), list):
                        if isinstance(encodings['attention_mask'][0], list):
                            attention_mask = encodings['attention_mask'][0]
                        else:
                            attention_mask = encodings['attention_mask']
                    else:
                        attention_mask = encodings.get('attention_mask')
                    
                    if len(input_ids) <= 1:  # Just checking if we have enough tokens
                        self.logger.warning(f"Text too short after tokenization: {len(input_ids)} tokens")
                        continue
                except Exception as e:
                    self.logger.error(f"Error tokenizing text: {str(e)}")
                    continue
                
                # Create examples using sliding window
                for i in range(0, len(input_ids) - 1, self.stride):
                    end_idx = min(i + self.max_seq_len, len(input_ids) - 1)
                    
                    # Skip incomplete sequences if requested
                    if skip_incomplete and end_idx - i < self.max_seq_len:
                        continue
                    
                    # Extract input sequence
                    input_seq = input_ids[i:end_idx]
                    
                    # Get corresponding attention mask if available
                    if attention_mask is not None:
                        mask_seq = attention_mask[i:end_idx]
                    else:
                        mask_seq = [1] * len(input_seq)
                    
                    # Create labels (shifted by 1 for next-token prediction)
                    label_seq = input_ids[i+1:end_idx+1]
                    
                    # Pad sequences if needed
                    if len(input_seq) < self.max_seq_len:
                        padding_len = self.max_seq_len - len(input_seq)
                        input_seq = input_seq + [self.pad_token_id] * padding_len
                        mask_seq = mask_seq + [0] * padding_len
                        label_seq = label_seq + [self.pad_token_id] * padding_len
                    
                    # Create example
                    example = {
                        'input_ids': torch.tensor(input_seq, dtype=torch.long),
                        'attention_mask': torch.tensor(mask_seq, dtype=torch.long),
                        'labels': torch.tensor(label_seq, dtype=torch.long)
                    }
                    
                    examples.append(example)
                    total_examples += 1
                    
                    # Break if we have enough examples
                    if max_examples is not None and total_examples >= max_examples:
                        break
                
                # Break if we have enough examples
                if max_examples is not None and total_examples >= max_examples:
                    break
            
            # Break if we have enough examples
            if max_examples is not None and total_examples >= max_examples:
                break
        
        self.logger.info(f"Created {len(examples)} examples from {len(file_paths)} JSON files")
        
        return TextDataset(examples, pad_token_id=self.pad_token_id)


def create_dataloaders(
    dataset: TextDataset,
    batch_size: int,
    val_split: float = 0.1,
    test_split: float = 0.0,
    shuffle: bool = True,
    num_workers: int = 4,
    seed: int = 42
) -> Dict[str, DataLoader]:
    """
    Create train, validation, and test DataLoaders from a dataset.
    
    Args:
        dataset: Dataset to split
        batch_size: Batch size for DataLoaders
        val_split: Fraction of data to use for validation
        test_split: Fraction of data to use for testing
        shuffle: Whether to shuffle the training data
        num_workers: Number of workers for DataLoader
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary of DataLoaders for train, validation, and test sets
    """
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Calculate split sizes
    total_size = len(dataset)
    val_size = int(val_split * total_size)
    test_size = int(test_split * total_size)
    train_size = total_size - val_size - test_size
    
    # Split dataset
    if val_size > 0 or test_size > 0:
        if test_size > 0:
            train_dataset, val_dataset, test_dataset = random_split(
                dataset, [train_size, val_size, test_size]
            )
        else:
            train_dataset, val_dataset = random_split(
                dataset, [train_size, val_size]
            )
            test_dataset = None
    else:
        train_dataset = dataset
        val_dataset = None
        test_dataset = None
    
    # Create DataLoaders
    dataloaders = {}
    
    dataloaders['train'] = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    
    if val_dataset is not None:
        dataloaders['validation'] = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
    
    if test_dataset is not None:
        dataloaders['test'] = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
    
    return dataloaders


def load_wikitext103(
    tokenizer: Any,
    data_dir: str,
    max_seq_len: int = 1024,
    stride: Optional[int] = None,
    max_examples: Optional[int] = None
) -> Dict[str, TextDataset]:
    """
    Load the WikiText-103 dataset.
    
    Args:
        tokenizer: Tokenizer to use
        data_dir: Directory containing the WikiText-103 dataset
        max_seq_len: Maximum sequence length
        stride: Stride for sliding window tokenization
        max_examples: Maximum number of examples per split
        
    Returns:
        Dictionary of datasets for train, validation, and test sets
    """
    # Define file paths
    train_path = os.path.join(data_dir, 'wiki.train.tokens')
    valid_path = os.path.join(data_dir, 'wiki.valid.tokens')
    test_path = os.path.join(data_dir, 'wiki.test.tokens')
    
    # Check if files exist
    for path in [train_path, valid_path, test_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"WikiText-103 file not found: {path}")
    
    # Create dataset builder
    builder = LMDatasetBuilder(
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        stride=stride
    )
    
    # Build datasets
    train_dataset = builder.build_from_text_files(
        [train_path],
        max_examples=max_examples
    )
    
    valid_dataset = builder.build_from_text_files(
        [valid_path],
        max_examples=max_examples
    )
    
    test_dataset = builder.build_from_text_files(
        [test_path],
        max_examples=max_examples
    )
    
    return {
        'train': train_dataset,
        'validation': valid_dataset,
        'test': test_dataset
    }


def load_pile(
    tokenizer: Any,
    data_dir: str,
    max_seq_len: int = 1024,
    stride: Optional[int] = None,
    max_examples: Optional[int] = None,
    split: str = 'train'
) -> TextDataset:
    """
    Load The Pile dataset.
    
    Args:
        tokenizer: Tokenizer to use
        data_dir: Directory containing The Pile dataset
        max_seq_len: Maximum sequence length
        stride: Stride for sliding window tokenization
        max_examples: Maximum number of examples
        split: Dataset split to load ('train', 'val', or 'test')
        
    Returns:
        TextDataset instance
    """
    # Define the split folder
    split_dir = {
        'train': 'train',
        'val': 'val',
        'test': 'test'
    }.get(split, 'train')
    
    split_path = os.path.join(data_dir, split_dir)
    
    if not os.path.exists(split_path):
        raise FileNotFoundError(f"The Pile directory not found: {split_path}")
    
    # Get all json files in the directory
    file_paths = [
        os.path.join(split_path, filename)
        for filename in os.listdir(split_path)
        if filename.endswith('.json') or filename.endswith('.jsonl')
    ]
    
    if not file_paths:
        raise FileNotFoundError(f"No JSON files found in {split_path}")
    
    # Create dataset builder
    builder = LMDatasetBuilder(
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        stride=stride
    )
    
    # Build dataset
    dataset = builder.build_from_json_files(
        file_paths,
        text_key='text',
        max_examples=max_examples
    )
    
    return dataset


def load_synthetic_dataset(
    tokenizer: Any,
    data_path: str,
    max_seq_len: int = 1024,
    stride: Optional[int] = None,
    max_examples: Optional[int] = None,
    text_key: str = 'text'
) -> TextDataset:
    """
    Load a synthetic dataset generated from Ollama phi4-mini.
    
    Args:
        tokenizer: Tokenizer to use
        data_path: Path to the synthetic dataset JSON file
        max_seq_len: Maximum sequence length
        stride: Stride for sliding window tokenization
        max_examples: Maximum number of examples
        text_key: Key for extracting text from JSON objects
        
    Returns:
        TextDataset instance
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Synthetic dataset file not found: {data_path}")
    
    # Create dataset builder
    builder = LMDatasetBuilder(
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        stride=stride
    )
    
    # Build dataset
    dataset = builder.build_from_json_files(
        [data_path],
        text_key=text_key,
        max_examples=max_examples
    )
    
    return dataset


def load_custom_dataset(
    tokenizer: Any,
    data_dir: str,
    max_seq_len: int = 1024,
    stride: Optional[int] = None,
    max_examples: Optional[int] = None,
    file_pattern: str = '*.txt',
    is_json: bool = False,
    text_key: str = 'text'
) -> TextDataset:
    """
    Load a custom dataset.
    
    Args:
        tokenizer: Tokenizer to use
        data_dir: Directory containing the dataset
        max_seq_len: Maximum sequence length
        stride: Stride for sliding window tokenization
        max_examples: Maximum number of examples
        file_pattern: Pattern for matching files
        is_json: Whether the files are JSON
        text_key: Key for extracting text from JSON objects
        
    Returns:
        TextDataset instance
    """
    import glob
    
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")
    
    # Get all matching files in the directory
    file_paths = glob.glob(os.path.join(data_dir, file_pattern))
    
    if not file_paths:
        raise FileNotFoundError(f"No files matching {file_pattern} found in {data_dir}")
    
    # Create dataset builder
    builder = LMDatasetBuilder(
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        stride=stride
    )
    
    # Build dataset
    if is_json:
        dataset = builder.build_from_json_files(
            file_paths,
            text_key=text_key,
            max_examples=max_examples
        )
    else:
        dataset = builder.build_from_text_files(
            file_paths,
            max_examples=max_examples
        )
    
    return dataset
