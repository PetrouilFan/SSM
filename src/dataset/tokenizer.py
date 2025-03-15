"""
Tokenizer implementations for Sparse SSM models.

This module provides tokenizer implementations and utilities for
text tokenization in language modeling tasks.
"""

import torch
import os
import json
from typing import Dict, List, Optional, Union, Tuple, Any, Callable
import re
from collections import Counter


class SimpleTokenizer:
    """
    A simple word-level tokenizer with BPE-like subword units.
    
    This tokenizer implements a basic word-level tokenization strategy
    with support for subword tokenization for handling out-of-vocabulary words.
    """
    
    def __init__(
        self,
        vocab_size: int = 50000,
        pad_token: str = "[PAD]",
        unk_token: str = "[UNK]",
        bos_token: str = "[BOS]",
        eos_token: str = "[EOS]"
    ):
        """
        Initialize the tokenizer.
        
        Args:
            vocab_size: Maximum vocabulary size
            pad_token: Token used for padding
            unk_token: Token used for unknown words
            bos_token: Token used for beginning of sequence
            eos_token: Token used for end of sequence
        """
        self.vocab_size = vocab_size
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        
        # Initialize vocabulary with special tokens
        self.special_tokens = [pad_token, unk_token, bos_token, eos_token]
        self.token_to_id = {token: i for i, token in enumerate(self.special_tokens)}
        self.id_to_token = {i: token for i, token in enumerate(self.special_tokens)}
        
        # Initialize subword vocabulary
        self.subword_vocab = {}
        
        # Initialize regex for tokenization
        self.regex = re.compile(r'\w+|[^\w\s]')
    
    def train(self, texts: List[str], min_freq: int = 5) -> None:
        """
        Train the tokenizer on a corpus of texts.
        
        Args:
            texts: List of texts to train on
            min_freq: Minimum frequency for a token to be included in the vocabulary
        """
        # Count word frequencies
        counter = Counter()
        
        for text in texts:
            # Split text into words and punctuation
            words = self.regex.findall(text.lower())
            counter.update(words)
        
        # Create vocabulary
        vocab = [token for token, count in counter.most_common(self.vocab_size - len(self.special_tokens))
                if count >= min_freq]
        
        # Add words to vocabulary
        for i, token in enumerate(vocab):
            idx = i + len(self.special_tokens)
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token
        
        # Create subword vocabulary for handling OOV words
        self._create_subword_vocab(counter)
    
    def _create_subword_vocab(self, counter: Counter) -> None:
        """
        Create a subword vocabulary for handling OOV words.
        
        Args:
            counter: Counter of word frequencies
        """
        # Create subword vocabulary (character-level for simplicity)
        chars = set()
        
        for word in counter.keys():
            chars.update(word)
        
        self.subword_vocab = {char: i for i, char in enumerate(sorted(chars))}
    
    def _tokenize_word(self, word: str) -> List[int]:
        """
        Tokenize a single word, handling OOV words with subword tokens.
        
        Args:
            word: Word to tokenize
            
        Returns:
            List of token IDs
        """
        if word in self.token_to_id:
            return [self.token_to_id[word]]
        else:
            # Fallback to character-level tokenization for OOV words
            subword_ids = []
            
            for char in word:
                if char in self.subword_vocab:
                    # Use the character's ID in the subword vocabulary
                    # We add the main vocabulary size to get a unique ID
                    subword_id = len(self.token_to_id) + self.subword_vocab[char]
                    subword_ids.append(subword_id)
                else:
                    # Unknown character, use UNK token
                    subword_ids.append(self.token_to_id[self.unk_token])
            
            return subword_ids if subword_ids else [self.token_to_id[self.unk_token]]
    
    def tokenize(self, text: str) -> List[int]:
        """
        Tokenize text into token IDs.
        
        Args:
            text: Text to tokenize
            
        Returns:
            List of token IDs
        """
        # Split text into words and punctuation
        words = self.regex.findall(text.lower())
        print(f"Tokenizing text: Found {len(words)} words: {words[:5]}...")
        
        # Tokenize each word
        token_ids = []
        
        for word in words:
            word_tokens = self._tokenize_word(word)
            token_ids.extend(word_tokens)
            if len(word_tokens) == 0:
                print(f"  Warning: Word '{word}' tokenized to 0 tokens")
        
        print(f"  Tokenized to {len(token_ids)} tokens")
        return token_ids
    
    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        truncation: bool = False,
        padding: bool = False
    ) -> Dict[str, List[int]]:
        """
        Encode text into token IDs with optional special tokens and padding.
        
        Args:
            text: Text to encode
            add_special_tokens: Whether to add special tokens (BOS/EOS)
            max_length: Maximum sequence length (if None, no limit)
            truncation: Whether to truncate sequences longer than max_length
            padding: Whether to pad sequences shorter than max_length
            
        Returns:
            Dictionary with input_ids and attention_mask
        """
        # Tokenize text
        token_ids = self.tokenize(text)
        
        # Add special tokens
        if add_special_tokens:
            token_ids = [self.token_to_id[self.bos_token]] + token_ids + [self.token_to_id[self.eos_token]]
        
        # Truncate if needed
        if max_length is not None and truncation and len(token_ids) > max_length:
            token_ids = token_ids[:max_length]
        
        # Create attention mask
        attention_mask = [1] * len(token_ids)
        
        # Pad if needed
        if max_length is not None and padding and len(token_ids) < max_length:
            padding_length = max_length - len(token_ids)
            token_ids = token_ids + [self.token_to_id[self.pad_token]] * padding_length
            attention_mask = attention_mask + [0] * padding_length
        
        return {
            'input_ids': token_ids,
            'attention_mask': attention_mask
        }
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs back into text.
        
        Args:
            token_ids: List of token IDs to decode
            skip_special_tokens: Whether to skip special tokens
            
        Returns:
            Decoded text
        """
        tokens = []
        
        for token_id in token_ids:
            if token_id in self.id_to_token:
                token = self.id_to_token[token_id]
                
                # Skip special tokens if requested
                if skip_special_tokens and token in self.special_tokens:
                    continue
                
                tokens.append(token)
            else:
                # Handle subword tokens
                if token_id >= len(self.token_to_id):
                    subword_idx = token_id - len(self.token_to_id)
                    
                    if subword_idx < len(self.subword_vocab):
                        for char, idx in self.subword_vocab.items():
                            if idx == subword_idx:
                                tokens.append(char)
                                break
                    else:
                        tokens.append(self.unk_token)
                else:
                    tokens.append(self.unk_token)
        
        # Join tokens into text (with spaces between words)
        text = ''.join(tokens).replace('', ' ').strip()
        
        return text
    
    def __call__(
        self,
        text: Union[str, List[str]],
        truncation: bool = False,
        padding: bool = False,
        max_length: Optional[int] = None,
        return_attention_mask: bool = True,
        return_tensors: Optional[str] = None
    ) -> Dict[str, Union[List[int], torch.Tensor]]:
        """
        Tokenize text and convert to model inputs.
        
        Args:
            text: Text or list of texts to tokenize
            truncation: Whether to truncate sequences longer than max_length
            padding: Whether to pad sequences shorter than max_length
            max_length: Maximum sequence length (if None, no limit)
            return_attention_mask: Whether to return attention mask
            return_tensors: Type of tensors to return (None, 'pt' for PyTorch)
            
        Returns:
            Dictionary with input_ids and attention_mask
        """
        if isinstance(text, str):
            text = [text]
        
        # Encode each text
        batch_inputs = []
        
        for t in text:
            inputs = self.encode(
                t,
                add_special_tokens=True,
                max_length=max_length,
                truncation=truncation,
                padding=padding
            )
            batch_inputs.append(inputs)
        
        # Combine batch inputs
        batch_size = len(batch_inputs)
        max_seq_len = max([len(inputs['input_ids']) for inputs in batch_inputs])
        
        # Initialize batch tensors
        input_ids = [[self.token_to_id[self.pad_token]] * max_seq_len for _ in range(batch_size)]
        attention_mask = [[0] * max_seq_len for _ in range(batch_size)]
        
        # Fill in batch tensors
        for i, inputs in enumerate(batch_inputs):
            seq_len = len(inputs['input_ids'])
            input_ids[i][:seq_len] = inputs['input_ids']
            
            if return_attention_mask:
                attention_mask[i][:seq_len] = inputs['attention_mask']
        
        # Convert to tensors if requested
        if return_tensors == 'pt':
            input_ids = torch.tensor(input_ids, dtype=torch.long)
            
            if return_attention_mask:
                attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        
        # Return inputs
        if return_attention_mask:
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask
            }
        else:
            return {
                'input_ids': input_ids
            }
    
    def save_pretrained(self, save_directory: str) -> None:
        """
        Save the tokenizer vocabulary and configuration to files.
        
        Args:
            save_directory: Directory to save the tokenizer files
        """
        os.makedirs(save_directory, exist_ok=True)
        
        # Save vocabulary
        vocab_file = os.path.join(save_directory, 'vocab.json')
        with open(vocab_file, 'w', encoding='utf-8') as f:
            json.dump(self.token_to_id, f, ensure_ascii=False, indent=2)
        print(f"Saved vocabulary with {len(self.token_to_id)} tokens to {vocab_file}")
        
        # Save subword vocabulary
        subword_file = os.path.join(save_directory, 'subword_vocab.json')
        with open(subword_file, 'w', encoding='utf-8') as f:
            json.dump(self.subword_vocab, f, ensure_ascii=False, indent=2)
        print(f"Saved subword vocabulary with {len(self.subword_vocab)} characters to {subword_file}")
        
        # Save configuration
        config_file = os.path.join(save_directory, 'tokenizer_config.json')
        config = {
            'vocab_size': self.vocab_size,
            'pad_token': self.pad_token,
            'unk_token': self.unk_token,
            'bos_token': self.bos_token,
            'eos_token': self.eos_token,
            'special_tokens': self.special_tokens
        }
        
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        print(f"Saved tokenizer configuration to {config_file}")
    
    @classmethod
    def from_pretrained(cls, directory: str) -> 'SimpleTokenizer':
        """
        Load a tokenizer from saved files.
        
        Args:
            directory: Directory containing the tokenizer files
            
        Returns:
            Loaded tokenizer
        """
        # Load configuration
        config_file = os.path.join(directory, 'tokenizer_config.json')
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # Create tokenizer
        tokenizer = cls(
            vocab_size=config['vocab_size'],
            pad_token=config['pad_token'],
            unk_token=config['unk_token'],
            bos_token=config['bos_token'],
            eos_token=config['eos_token']
        )
        
        # Load vocabulary
        vocab_file = os.path.join(directory, 'vocab.json')
        with open(vocab_file, 'r', encoding='utf-8') as f:
            tokenizer.token_to_id = json.load(f)
        
        # Create id_to_token by inverting token_to_id
        tokenizer.id_to_token = {int(v): k for k, v in tokenizer.token_to_id.items()}
        
        # Load subword vocabulary
        subword_file = os.path.join(directory, 'subword_vocab.json')
        with open(subword_file, 'r', encoding='utf-8') as f:
            tokenizer.subword_vocab = json.load(f)
        
        return tokenizer


def train_tokenizer_from_files(
    file_paths: List[str],
    vocab_size: int = 50000,
    min_freq: int = 5,
    output_dir: str = 'tokenizer',
    sample_size: Optional[int] = None,
    text_key: str = 'text'
) -> SimpleTokenizer:
    """
    Train a new tokenizer from a list of text files or JSON files.
    
    Args:
        file_paths: List of paths to text files or JSON files
        vocab_size: Maximum vocabulary size
        min_freq: Minimum frequency for a token to be included in the vocabulary
        output_dir: Directory to save the tokenizer
        sample_size: Maximum number of lines/examples to sample from each file (if None, use all)
        text_key: Key for extracting text from JSON objects (if JSON files)
        
    Returns:
        Trained tokenizer
    """
    # Load texts from files
    texts = []
    
    for file_path in file_paths:
        # Check if file is JSON
        if file_path.endswith('.json') or file_path.endswith('.jsonl'):
            with open(file_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
                
                # Extract texts from JSON data
                json_texts = []
                if isinstance(json_data, list):
                    # List of objects
                    for item in json_data:
                        if text_key in item:
                            json_texts.append(item[text_key])
                elif isinstance(json_data, dict):
                    # Single object or dictionary of objects
                    if text_key in json_data:
                        json_texts.append(json_data[text_key])
                    else:
                        # Try to extract from values if they are dictionaries
                        for value in json_data.values():
                            if isinstance(value, dict) and text_key in value:
                                json_texts.append(value[text_key])
                
                # Sample texts if needed
                if sample_size is not None and len(json_texts) > sample_size:
                    import random
                    json_texts = random.sample(json_texts, sample_size)
                
                texts.extend(json_texts)
        else:
            # Regular text file
            with open(file_path, 'r', encoding='utf-8') as f:
                if sample_size is not None:
                    # Sample lines randomly
                    import random
                    lines = f.readlines()
                    if len(lines) > sample_size:
                        lines = random.sample(lines, sample_size)
                    texts.extend(lines)
                else:
                    texts.extend(f.readlines())
    
    # Create and train tokenizer
    tokenizer = SimpleTokenizer(vocab_size=vocab_size)
    tokenizer.train(texts, min_freq=min_freq)
    
    # Save tokenizer
    os.makedirs(output_dir, exist_ok=True)
    tokenizer.save_pretrained(output_dir)
    
    return tokenizer


def get_hf_compatible_tokenizer(tokenizer: SimpleTokenizer) -> Any:
    """
    Convert a SimpleTokenizer to a HuggingFace-compatible tokenizer.
    
    This is a utility function for compatibility with HuggingFace's transformers library.
    
    Args:
        tokenizer: SimpleTokenizer instance
        
    Returns:
        HuggingFace-compatible tokenizer
    """
    try:
        from transformers import PreTrainedTokenizerFast
        
        # Create a HuggingFace-compatible tokenizer
        hf_tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer,
            unk_token=tokenizer.unk_token,
            pad_token=tokenizer.pad_token,
            bos_token=tokenizer.bos_token,
            eos_token=tokenizer.eos_token,
            # Add tokenizer.encode, tokenizer.decode, etc. methods
            # to match the HuggingFace API
            tokenize=tokenizer.tokenize,
            encode=lambda text, **kwargs: tokenizer.encode(text, **kwargs)['input_ids'],
            decode=tokenizer.decode
        )
        
        return hf_tokenizer
    
    except ImportError:
        # If transformers is not installed, return the original tokenizer
        print("Warning: transformers library not found. Returning the original tokenizer.")
        return tokenizer
