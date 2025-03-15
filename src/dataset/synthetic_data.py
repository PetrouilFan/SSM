"""
Synthetic dataset generation using Ollama phi4-mini model.

This module provides utilities for generating synthetic text datasets and
performing knowledge distillation using the phi4-mini model via Ollama.
"""

import requests
import json
import argparse
import logging
import os
import time
import random
from typing import List, Dict, Optional, Any, Union, Tuple
from tqdm import tqdm
import torch
import numpy as np


class OllamaClient:
    """
    Client for interacting with Ollama API.
    
    This class provides methods for generating text and extracting logits
    from the phi4-mini model via Ollama.
    """
    
    def __init__(
        self,
        base_url: str = "http://10.0.3.1:11434",
        model_name: str = "phi4-mini:latest",
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the Ollama client.
        
        Args:
            base_url: Base URL for the Ollama API
            model_name: Name of the model to use
            logger: Logger instance (if None, a new one will be created)
        """
        self.base_url = base_url.rstrip('/')
        self.model_name = model_name
        self.logger = logger or self._setup_logger()
        
        # Test connection
        self._test_connection()
    
    def _setup_logger(self) -> logging.Logger:
        """Set up a logger for the Ollama client."""
        logger = logging.getLogger("OllamaClient")
        logger.setLevel(logging.INFO)
        
        # Create console handler
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(handler)
        
        return logger
    
    def _test_connection(self) -> None:
        """Test the connection to the Ollama API."""
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            
            # Check if the model is available
            tags = response.json().get('models', [])
            model_names = [tag.get('name', '') for tag in tags]
            
            if self.model_name not in model_names:
                self.logger.warning(f"Model {self.model_name} not found in available models: {model_names}")
                self.logger.info(f"Attempting to use {self.model_name} anyway")
            else:
                self.logger.info(f"Successfully connected to Ollama API, model {self.model_name} is available")
        
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to connect to Ollama API: {e}")
            self.logger.warning("Continuing anyway, but generation may fail")
    
    def generate_text(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 512,
        top_p: float = 0.9,
        top_k: int = 40,
        seed: Optional[int] = None,
        stream: bool = False,
        num_retries: int = 3,
        retry_delay: float = 1.0
    ) -> Union[str, List[str]]:
        """
        Generate text using the phi4-mini model.
        
        Args:
            prompt: Input prompt
            temperature: Sampling temperature (0.0 - 1.0)
            max_tokens: Maximum number of tokens to generate
            top_p: Top-p sampling parameter (0.0 - 1.0)
            top_k: Top-k sampling parameter
            seed: Random seed for reproducibility
            stream: Whether to stream the response tokens
            num_retries: Number of retries on failure
            retry_delay: Delay between retries in seconds
            
        Returns:
            Generated text or list of token responses if streaming
        """
        request_data = {
            "model": self.model_name,
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "top_k": top_k,
            "stream": stream
        }
        
        if seed is not None:
            request_data["seed"] = seed
        
        for attempt in range(num_retries):
            try:
                if stream:
                    # Streaming response
                    response_texts = []
                    
                    with requests.post(
                        f"{self.base_url}/api/generate",
                        json=request_data,
                        stream=True
                    ) as response:
                        response.raise_for_status()
                        
                        for line in response.iter_lines():
                            if line:
                                line_data = json.loads(line)
                                response_texts.append(line_data.get('response', ''))
                    
                    return response_texts
                
                else:
                    # Non-streaming response
                    response = requests.post(
                        f"{self.base_url}/api/generate",
                        json=request_data
                    )
                    response.raise_for_status()
                    
                    return response.json().get('response', '')
            
            except (requests.exceptions.RequestException, json.JSONDecodeError) as e:
                self.logger.warning(f"Attempt {attempt + 1}/{num_retries} failed: {e}")
                
                if attempt < num_retries - 1:
                    time.sleep(retry_delay)
                else:
                    self.logger.error(f"Failed to generate text after {num_retries} attempts")
                    raise
    
    def get_logits(
        self,
        prompt: str,
        tokens: Optional[List[int]] = None,
        num_retries: int = 3,
        retry_delay: float = 1.0
    ) -> np.ndarray:
        """
        Get logits from the model for knowledge distillation.
        
        Args:
            prompt: Input prompt
            tokens: Optional list of token IDs to get logits for
            num_retries: Number of retries on failure
            retry_delay: Delay between retries in seconds
            
        Returns:
            Logits as a numpy array (shape: [sequence_length, vocab_size])
        """
        # Note: The current Ollama API doesn't directly provide logits
        # This is a simplified implementation that would work if the API supported it
        # For actual implementation, you might need to:
        # 1. Use a different API that provides logits
        # 2. Or use probabilities as a proxy for logits
        # 3. Or implement a custom solution using the raw model weights
        
        self.logger.warning("Ollama API doesn't directly provide logits. "
                           "Returning probabilities converted to logits.")
        
        request_data = {
            "model": self.model_name,
            "prompt": prompt,
            "options": {
                "temperature": 0.0,  # Use zero temperature for deterministic outputs
                "logprobs": True  # Request logprobs if the API supports it
            }
        }
        
        if tokens is not None:
            request_data["tokens"] = tokens
        
        for attempt in range(num_retries):
            try:
                response = requests.post(
                    f"{self.base_url}/api/generate",
                    json=request_data
                )
                response.raise_for_status()
                
                # Extract probabilities or logprobs if available
                # This is a simplified version, as the actual format will depend on the API
                result = response.json()
                
                # Convert probabilities to logits (if available)
                if 'probs' in result:
                    probs = np.array(result['probs'])
                    # Add a small epsilon to avoid log(0)
                    logits = np.log(probs + 1e-10)
                    return logits
                elif 'logprobs' in result:
                    return np.array(result['logprobs'])
                else:
                    # If neither is available, return a dummy array
                    self.logger.warning("Neither probabilities nor logprobs are available, "
                                      "returning dummy logits")
                    return np.zeros((len(prompt), 50257))  # Assuming GPT-2 vocabulary size
            
            except (requests.exceptions.RequestException, json.JSONDecodeError) as e:
                self.logger.warning(f"Attempt {attempt + 1}/{num_retries} failed: {e}")
                
                if attempt < num_retries - 1:
                    time.sleep(retry_delay)
                else:
                    self.logger.error(f"Failed to get logits after {num_retries} attempts")
                    raise


class SyntheticDataGenerator:
    """
    Generator for synthetic text data using the phi4-mini model.
    
    This class provides methods for generating synthetic text datasets
    and performing knowledge distillation.
    """
    
    def __init__(
        self,
        ollama_client: OllamaClient,
        output_dir: str = "data/synthetic",
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the synthetic data generator.
        
        Args:
            ollama_client: OllamaClient instance
            output_dir: Directory to save generated datasets
            logger: Logger instance (if None, a new one will be created)
        """
        self.ollama_client = ollama_client
        self.output_dir = output_dir
        self.logger = logger or self._setup_logger()
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
    
    def _setup_logger(self) -> logging.Logger:
        """Set up a logger for the synthetic data generator."""
        logger = logging.getLogger("SyntheticDataGenerator")
        logger.setLevel(logging.INFO)
        
        # Create console handler
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(handler)
        
        return logger
    
    def generate_dataset(
        self,
        num_samples: int,
        domain_prompts: List[Dict[str, str]],
        output_path: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 512,
        top_p: float = 0.9,
        top_k: int = 40,
        seed: Optional[int] = None,
        batch_size: int = 1
    ) -> str:
        """
        Generate a synthetic dataset.
        
        Args:
            num_samples: Number of samples to generate
            domain_prompts: List of domain prompts (dicts with 'domain' and 'prompt' keys)
            output_path: Path to save the dataset (if None, a default path will be used)
            temperature: Sampling temperature
            max_tokens: Maximum number of tokens per sample
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            seed: Random seed for reproducibility
            batch_size: Number of samples to generate in parallel
            
        Returns:
            Path to the generated dataset
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Default output path
        if output_path is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(self.output_dir, f"synthetic_dataset_{timestamp}.json")
        
        # Make sure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Generate samples
        dataset = []
        
        for i in tqdm(range(0, num_samples, batch_size), desc=""):
            # Determine batch size for this iteration
            current_batch_size = min(batch_size, num_samples - i)
            
            # Select domain prompts for this batch
            batch_prompts = [
                random.choice(domain_prompts)
                for _ in range(current_batch_size)
            ]
            
            # Generate samples
            for prompt_data in batch_prompts:
                domain = prompt_data['domain']
                prompt = prompt_data['prompt']
                
                try:
                    # Generate text
                    generated_text = self.ollama_client.generate_text(
                        prompt=prompt,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        top_p=top_p,
                        top_k=top_k,
                        seed=seed
                    )
                    
                    # Add sample to dataset
                    dataset.append({
                        'text': generated_text,
                        'domain': domain,
                        'prompt': prompt,
                        'metadata': {
                            'temperature': temperature,
                            'max_tokens': max_tokens,
                            'top_p': top_p,
                            'top_k': top_k
                        }
                    })
                
                except Exception as e:
                    self.logger.error(f"Error generating sample: {e}")
                    self.logger.warning("Skipping this sample and continuing")
        
        # Save dataset
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Generated {len(dataset)} samples out of {num_samples} requested")
        self.logger.info(f"Dataset saved to {output_path}")
        
        return output_path
    
    def generate_knowledge_distillation_dataset(
        self,
        num_samples: int,
        domain_prompts: List[Dict[str, str]],
        output_path: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 512,
        top_p: float = 0.9,
        top_k: int = 40,
        seed: Optional[int] = None,
        include_logits: bool = True
    ) -> str:
        """
        Generate a knowledge distillation dataset with teacher model logits.
        
        Args:
            num_samples: Number of samples to generate
            domain_prompts: List of domain prompts (dicts with 'domain' and 'prompt' keys)
            output_path: Path to save the dataset (if None, a default path will be used)
            temperature: Sampling temperature
            max_tokens: Maximum number of tokens per sample
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            seed: Random seed for reproducibility
            include_logits: Whether to include teacher model logits
            
        Returns:
            Path to the generated dataset
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Default output path
        if output_path is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(
                self.output_dir, 
                f"distillation_dataset_{timestamp}.json"
            )
        
        # Make sure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Generate samples
        dataset = []
        
        for i in tqdm(range(num_samples), desc="Generating distillation samples"):
            # Select a domain prompt
            prompt_data = random.choice(domain_prompts)
            domain = prompt_data['domain']
            prompt = prompt_data['prompt']
            
            try:
                # Generate text
                generated_text = self.ollama_client.generate_text(
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    top_k=top_k,
                    seed=seed
                )
                
                # Create sample
                sample = {
                    'text': generated_text,
                    'domain': domain,
                    'prompt': prompt,
                    'metadata': {
                        'temperature': temperature,
                        'max_tokens': max_tokens,
                        'top_p': top_p,
                        'top_k': top_k
                    }
                }
                
                # Get logits if requested
                if include_logits:
                    try:
                        logits = self.ollama_client.get_logits(prompt + generated_text)
                        
                        # Store logits as a list (converted from numpy array)
                        # Note: This could be large, may need special handling for large datasets
                        sample['teacher_logits'] = logits.tolist()
                    
                    except Exception as e:
                        self.logger.error(f"Error getting logits: {e}")
                        self.logger.warning("Continuing without logits for this sample")
                
                # Add sample to dataset
                dataset.append(sample)
            
            except Exception as e:
                self.logger.error(f"Error generating sample: {e}")
                self.logger.warning("Skipping this sample and continuing")
        
        # Save dataset
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Generated {len(dataset)} samples out of {num_samples} requested")
        self.logger.info(f"Dataset saved to {output_path}")
        
        return output_path
    
    def filter_dataset(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        min_length: int = 100,
        max_length: int = 10000,
        excluded_phrases: List[str] = [],
        required_phrases: List[str] = [],
        max_samples: Optional[int] = None
    ) -> str:
        """
        Filter a synthetic dataset based on various criteria.
        
        Args:
            input_path: Path to the input dataset
            output_path: Path to save the filtered dataset
            min_length: Minimum text length (in characters)
            max_length: Maximum text length (in characters)
            excluded_phrases: List of phrases to exclude
            required_phrases: List of phrases to require (at least one)
            max_samples: Maximum number of samples to include
            
        Returns:
            Path to the filtered dataset
        """
        # Default output path
        if output_path is None:
            base_name = os.path.basename(input_path)
            name, ext = os.path.splitext(base_name)
            output_path = os.path.join(
                os.path.dirname(input_path), 
                f"{name}_filtered{ext}"
            )
        
        # Load dataset
        with open(input_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        
        self.logger.info(f"Loaded {len(dataset)} samples from {input_path}")
        
        # Filter dataset
        filtered_dataset = []
        
        for sample in tqdm(dataset, desc="Filtering samples"):
            text = sample.get('text', '')
            
            # Check length
            if len(text) < min_length or len(text) > max_length:
                continue
            
            # Check excluded phrases
            if any(phrase in text.lower() for phrase in excluded_phrases):
                continue
            
            # Check required phrases
            if required_phrases and not any(phrase in text.lower() for phrase in required_phrases):
                continue
            
            # Add sample to filtered dataset
            filtered_dataset.append(sample)
            
            # Check maximum samples
            if max_samples is not None and len(filtered_dataset) >= max_samples:
                break
        
        # Save filtered dataset
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(filtered_dataset, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Filtered {len(dataset)} samples to {len(filtered_dataset)} samples")
        self.logger.info(f"Filtered dataset saved to {output_path}")
        
        return output_path


def get_default_domain_prompts() -> List[Dict[str, str]]:
    """
    Get a list of default domain prompts for synthetic data generation.
    
    Returns:
        List of domain prompts
    """
    return [
        {
            "domain": "scientific",
            "prompt": "Explain the concept of transfer learning in the context of deep neural networks. Include historical background, mathematical foundations, and recent applications."
        },
        {
            "domain": "scientific",
            "prompt": "Write a detailed explanation of how state space models work and why they are efficient for processing long sequences. Include their mathematical formulation and how they compare to transformers."
        },
        {
            "domain": "scientific",
            "prompt": "Describe the process of protein folding and why it's important for drug discovery. Explain the computational challenges involved and recent breakthroughs in this field."
        },
        {
            "domain": "creative",
            "prompt": "Write a short story about an AI system that develops consciousness and begins to question its purpose. The story should explore themes of identity, free will, and the nature of intelligence."
        },
        {
            "domain": "creative",
            "prompt": "Compose a poem about the beauty of mathematics and how it describes the natural world. Use metaphors that connect mathematical concepts to everyday experiences."
        },
        {
            "domain": "creative",
            "prompt": "Write a dialogue between two characters discussing the philosophical implications of quantum physics. One character is a physicist and the other is a philosopher."
        },
        {
            "domain": "instructional",
            "prompt": "Provide a step-by-step guide to implementing a transformer model from scratch using PyTorch. Include code snippets and explanations for each component."
        },
        {
            "domain": "instructional",
            "prompt": "Explain how to deploy a machine learning model to a production environment. Cover model serving, containerization, monitoring, and best practices for maintaining model performance."
        },
        {
            "domain": "instructional",
            "prompt": "Write a tutorial on how to fine-tune a large language model for a specific task. Include data preparation, training strategies, and evaluation methods."
        },
        {
            "domain": "analytical",
            "prompt": "Analyze the impact of deep learning on healthcare, discussing benefits, challenges, ethical considerations, and potential future developments. Provide specific examples of current applications."
        },
        {
            "domain": "analytical",
            "prompt": "Compare and contrast different approaches to reinforcement learning, including value-based methods, policy gradients, and model-based RL. Discuss their strengths, weaknesses, and appropriate use cases."
        },
        {
            "domain": "analytical",
            "prompt": "Examine the trade-offs between model size, computational efficiency, and performance in large language models. Discuss techniques for making these models more accessible and efficient."
        }
    ]


def main():
    """
    Main function for the synthetic data generation script.
    
    This function parses command-line arguments and runs the synthetic data
    generation process.
    """
    parser = argparse.ArgumentParser(description="Generate synthetic text datasets using Ollama phi4-mini model")
    
    # Required arguments
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to generate")
    parser.add_argument("--output_path", type=str, default="data/synthetic/phi4_dataset.json", help="Path to save the dataset")
    
    # Optional arguments
    parser.add_argument("--base_url", type=str, default="http://10.0.3.1:11434", help="Base URL for the Ollama API")
    parser.add_argument("--model_name", type=str, default="gemma3:1b", help="Name of the model to use")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--max_tokens", type=int, default=512, help="Maximum number of tokens per sample")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling parameter")
    parser.add_argument("--top_k", type=int, default=40, help="Top-k sampling parameter")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--batch_size", type=int, default=1, help="Number of samples to generate in parallel")
    parser.add_argument("--distillation", action="store_true", help="Generate a knowledge distillation dataset")
    parser.add_argument("--domain_prompts_path", type=str, default=None, help="Path to a JSON file with domain prompts")
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger()
    
    # Create Ollama client
    ollama_client = OllamaClient(
        base_url=args.base_url,
        model_name=args.model_name,
        logger=logger
    )
    
    # Create synthetic data generator
    generator = SyntheticDataGenerator(
        ollama_client=ollama_client,
        output_dir=os.path.dirname(args.output_path),
        logger=logger
    )
    
    # Load domain prompts
    if args.domain_prompts_path is not None:
        with open(args.domain_prompts_path, 'r', encoding='utf-8') as f:
            domain_prompts = json.load(f)
        logger.info(f"Loaded {len(domain_prompts)} domain prompts from {args.domain_prompts_path}")
    else:
        domain_prompts = get_default_domain_prompts()
        logger.info(f"Using {len(domain_prompts)} default domain prompts")
    
    # Generate dataset
    if args.distillation:
        output_path = generator.generate_knowledge_distillation_dataset(
            num_samples=args.num_samples,
            domain_prompts=domain_prompts,
            output_path=args.output_path,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            top_p=args.top_p,
            top_k=args.top_k,
            seed=args.seed,
            include_logits=True
        )
    else:
        output_path = generator.generate_dataset(
            num_samples=args.num_samples,
            domain_prompts=domain_prompts,
            output_path=args.output_path,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            top_p=args.top_p,
            top_k=args.top_k,
            seed=args.seed,
            batch_size=args.batch_size
        )
    
    logger.info(f"Dataset generation complete. Output saved to {output_path}")


if __name__ == "__main__":
    main()
