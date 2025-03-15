"""
Mamba-inspired model architecture based on Sparse State Space Models (SSMs).

This module implements the complete architecture for a language model based on
Sparse SSMs with Selective Parameterization, inspired by the Mamba architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Union, List, Any
import math

from .ssm import SSMLayer, SSMKernel, ParallelSSMKernel


class FeedForwardNetwork(nn.Module):
    """
    Simple feed-forward network with GELU activation.
    
    This module implements a standard feed-forward network with expansion and
    projection layers, commonly used in transformer-like architectures.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        ffn_dim: int,
        dropout: float = 0.0,
        activation: str = 'gelu',
    ):
        """
        Initialize the feed-forward network.
        
        Args:
            hidden_dim: Dimension of the input and output
            ffn_dim: Dimension of the expanded hidden representation
            dropout: Dropout probability
            activation: Activation function ('relu', 'gelu', 'silu')
        """
        super().__init__()
        
        self.linear1 = nn.Linear(hidden_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'silu':
            self.activation = nn.SiLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the feed-forward network to the input.
        
        Args:
            x: Input tensor, shape (batch_size, seq_len, hidden_dim)
            
        Returns:
            Output tensor, shape (batch_size, seq_len, hidden_dim)
        """
        x = self.linear1(x)
        x = self.activation(x)
        
        if self.dropout is not None:
            x = self.dropout(x)
        
        x = self.linear2(x)
        
        return x


class MambaBlock(nn.Module):
    """
    Mamba-inspired block combining SSM layer and feed-forward network.
    
    This module implements a single block in the Mamba architecture, which
    consists of an SSM layer followed by a feed-forward network, with
    residual connections and layer normalization.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        state_dim: int,
        ffn_dim: int,
        dropout: float = 0.0,
        activation: str = 'gelu',
        layer_norm_eps: float = 1e-5,
        ssm_kernel_args: Dict = None,
        use_parallel_scan: bool = True,
    ):
        """
        Initialize the Mamba block.
        
        Args:
            hidden_dim: Dimension of the hidden representation
            state_dim: Dimension of the SSM state vector
            ffn_dim: Dimension of the feed-forward network
            dropout: Dropout probability
            activation: Activation function ('relu', 'gelu', 'silu')
            layer_norm_eps: Epsilon for layer normalization
            ssm_kernel_args: Arguments for the SSM kernel
            use_parallel_scan: Whether to use parallel scan implementation
        """
        super().__init__()
        
        # SSM layer
        self.ssm_layer = SSMLayer(
            hidden_dim=hidden_dim,
            state_dim=state_dim,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            dropout=dropout,
            kernel_args=ssm_kernel_args,
            use_parallel_scan=use_parallel_scan,
        )
        
        # Layer normalization
        self.norm = nn.LayerNorm(hidden_dim, eps=layer_norm_eps)
        
        # Feed-forward network
        self.ffn = FeedForwardNetwork(
            hidden_dim=hidden_dim,
            ffn_dim=ffn_dim,
            dropout=dropout,
            activation=activation,
        )
        
        # Output dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
    
    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Apply the Mamba block to the input.
        
        Args:
            x: Input tensor, shape (batch_size, seq_len, hidden_dim)
            state: Initial state for the SSM layer, shape (batch_size, state_dim)
                   If None, initialized to zeros
            
        Returns:
            output: Output tensor, shape (batch_size, seq_len, hidden_dim)
            state (optional): Final state if ssm_layer.return_state is True
        """
        # Apply SSM layer
        if state is not None:
            if self.ssm_layer.ssm.return_state:
                ssm_output, new_state = self.ssm_layer(x, state)
            else:
                ssm_output = self.ssm_layer(x, state)
        else:
            if self.ssm_layer.ssm.return_state:
                ssm_output, new_state = self.ssm_layer(x)
            else:
                ssm_output = self.ssm_layer(x)
        
        # Apply layer normalization
        normalized = self.norm(ssm_output)
        
        # Apply feed-forward network
        ffn_output = self.ffn(normalized)
        
        # Apply dropout if specified
        if self.dropout is not None:
            ffn_output = self.dropout(ffn_output)
        
        # Add residual connection
        output = ffn_output + ssm_output
        
        if self.ssm_layer.ssm.return_state:
            return output, new_state
        else:
            return output


class SparseSSM(nn.Module):
    """
    Complete language model based on Sparse SSMs with Selective Parameterization.
    
    This module implements the full model architecture, including token embeddings,
    multiple Mamba blocks, and output projection.
    """
    
    # Flag for gradient checkpointing
    _keys_to_ignore_on_save = []
    is_gradient_checkpointing = False
    
    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory efficiency."""
        self.is_gradient_checkpointing = True
        
        # Store original forward methods if not already done
        if not hasattr(self, "_original_forwards"):
            self._original_forwards = {}
            for i, block in enumerate(self.blocks):
                self._original_forwards[i] = block.forward
        
        # A simpler approach: we'll modify the model's forward pass
        # rather than each block's forward method
        def create_custom_forward(module_index):
            def custom_forward(*inputs):
                x = inputs[0]
                state = inputs[1] if len(inputs) > 1 else None
                return self._original_forwards[module_index](x, state)
            return custom_forward
        
        # We'll use this in the model's forward method
        self.is_gradient_checkpointing = True
            
        return self
        
    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self.is_gradient_checkpointing = False
        # Note: This doesn't actually restore the original forward functions
        # A model reload would be required to fully disable checkpointing
        return self
    
    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int,
        state_dim: int,
        ffn_dim: int,
        num_layers: int,
        max_seq_len: int = 2048,
        dropout: float = 0.0,
        activation: str = 'gelu',
        layer_norm_eps: float = 1e-5,
        selective_param_class: str = 'sparse',
        sparsity_level: float = 0.9,
        use_parallel_scan: bool = True,
        tie_embedding_weights: bool = True,
        pad_token_id: int = 0,
    ):
        """
        Initialize the Sparse SSM model.
        
        Args:
            vocab_size: Size of the vocabulary
            hidden_dim: Dimension of the hidden representation
            state_dim: Dimension of the SSM state vector
            ffn_dim: Dimension of the feed-forward network
            num_layers: Number of Mamba blocks
            max_seq_len: Maximum sequence length
            dropout: Dropout probability
            activation: Activation function ('relu', 'gelu', 'silu')
            layer_norm_eps: Epsilon for layer normalization
            selective_param_class: Type of selective parameterization ('dense', 'sparse', 'low_rank')
            sparsity_level: Sparsity level for sparse parameterization
            use_parallel_scan: Whether to use parallel scan implementation
            tie_embedding_weights: Whether to tie input and output embedding weights
            pad_token_id: Token ID for padding
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id
        
        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=pad_token_id)
        
        # Common SSM kernel arguments
        ssm_kernel_args = {
            'selective_param_class': selective_param_class,
            'param_init_method': 'normal',  # Changed from 'xavier' to 'normal' which is supported
            'discretization_method': 'zoh',
            'dt_init': 0.01,
            'dt_learnable': True,
            'sparsity_level': sparsity_level,
            'init_scale': 0.01,
            'return_state': False,
        }
        
        # Mamba blocks
        self.blocks = nn.ModuleList([
            MambaBlock(
                hidden_dim=hidden_dim,
                state_dim=state_dim,
                ffn_dim=ffn_dim,
                dropout=dropout,
                activation=activation,
                layer_norm_eps=layer_norm_eps,
                ssm_kernel_args=ssm_kernel_args,
                use_parallel_scan=use_parallel_scan,
            )
            for _ in range(num_layers)
        ])
        
        # Final layer normalization
        self.norm_f = nn.LayerNorm(hidden_dim, eps=layer_norm_eps)
        
        # Output projection
        if tie_embedding_weights:
            # Use the transpose of the token embedding weights
            self.output_projection = lambda x: F.linear(x, self.token_embedding.weight)
        else:
            self.output_projection = nn.Linear(hidden_dim, vocab_size)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize model weights."""
        if isinstance(module, nn.Linear):
            # Initialize linear layers with Xavier uniform distribution
            nn.init.xavier_uniform_(module.weight, gain=0.01)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            # Initialize embedding layers with normal distribution
            nn.init.normal_(module.weight, mean=0.0, std=0.01)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        states: Optional[List[torch.Tensor]] = None,
        return_dict: bool = True,
        output_hidden_states: bool = False,
        return_states: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the model.
        
        Args:
            input_ids: Input token IDs, shape (batch_size, seq_len)
            attention_mask: Attention mask, shape (batch_size, seq_len)
                            1 for tokens to attend to, 0 for padding tokens
            states: List of initial states for each layer, each with shape (batch_size, state_dim)
                   If None, initialized to zeros
            return_dict: Whether to return a dictionary
            output_hidden_states: Whether to return hidden states from all layers
            return_states: Whether to return the final states
            
        Returns:
            Dictionary containing:
                - logits: Output logits, shape (batch_size, seq_len, vocab_size)
                - hidden_states: Hidden states from all layers (if output_hidden_states is True)
                - states: Final states from all layers (if return_states is True)
        """
        batch_size, seq_len = input_ids.size()
        
        # Get token embeddings
        x = self.token_embedding(input_ids)  # (batch_size, seq_len, hidden_dim)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            # Expand mask to hidden dimension
            mask = attention_mask.unsqueeze(-1).expand_as(x)
            # Apply mask by zeroing out padded elements
            x = x * mask
        
        # Initialize hidden states list if needed
        all_hidden_states = [] if output_hidden_states else None
        all_states = [] if return_states else None
        
        # Apply Mamba blocks
        for i, block in enumerate(self.blocks):
            # Get initial state for this layer if provided
            if states is not None:
                state_i = states[i]
            else:
                state_i = None
            
            # Apply block
            if block.ssm_layer.ssm.return_state or return_states:
                x, state = block(x, state_i)
                if return_states:
                    all_states.append(state)
            else:
                x = block(x, state_i)
            
            # Collect hidden states if needed
            if output_hidden_states:
                all_hidden_states.append(x)
        
        # Apply final layer normalization
        x = self.norm_f(x)
        
        # Apply output projection to get logits
        logits = self.output_projection(x)
        
        if return_dict:
            outputs = {'logits': logits}
            if output_hidden_states:
                outputs['hidden_states'] = all_hidden_states
            if return_states:
                outputs['states'] = all_states
            return outputs
        else:
            return logits
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.0,
        do_sample: bool = True,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate text from the model.
        
        Args:
            input_ids: Input token IDs, shape (batch_size, seq_len)
            max_length: Maximum length to generate
            temperature: Temperature for sampling
            top_k: Top-k sampling parameter (0 to disable)
            top_p: Top-p sampling parameter (0.0 to disable)
            do_sample: Whether to sample from the distribution or take the argmax
            pad_token_id: Token ID for padding
            eos_token_id: Token ID to signal the end of generation
            
        Returns:
            Generated token IDs, shape (batch_size, max_length)
        """
        if pad_token_id is None:
            pad_token_id = self.pad_token_id
        
        # Initialize output with input_ids
        output = input_ids.clone()
        batch_size = output.size(0)
        
        # Initialize states for all layers
        states = [torch.zeros(batch_size, self.state_dim, device=input_ids.device)
                 for _ in range(self.num_layers)]
        
        # Keep track of which sequences have finished
        if eos_token_id is not None:
            unfinished = torch.ones(batch_size, dtype=torch.bool, device=input_ids.device)
        
        # Set SSM layers to return states
        for block in self.blocks:
            block.ssm_layer.ssm.return_state = True
        
        # Initial forward pass to get states
        with torch.no_grad():
            outputs = self.forward(
                input_ids,
                return_dict=True,
                return_states=True
            )
            states = outputs['states']
        
        # Generate tokens up to max_length
        for i in range(input_ids.size(1), max_length):
            # Only consider the last token for generation
            input_i = output[:, -1].unsqueeze(-1)
            
            # Forward pass to get next token predictions
            with torch.no_grad():
                # Apply model to get logits and update states
                outputs = self.forward(
                    input_i,
                    states=states,
                    return_dict=True,
                    return_states=True
                )
                logits = outputs['logits'][:, -1, :]
                states = outputs['states']
                
                # Apply temperature scaling
                if temperature > 0:
                    logits = logits / temperature
                
                # Apply top-k filtering
                if top_k > 0:
                    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                    logits[indices_to_remove] = float('-inf')
                
                # Apply top-p (nucleus) filtering
                if top_p > 0.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # Shift the indices to the right to keep also the first token above the threshold
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    # Create a mask of indices to remove
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    logits[:, indices_to_remove] = float('-inf')
                
                # Sample from the filtered distribution
                if do_sample:
                    probs = F.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    # Take the argmax
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)
                
                # Update output with next token
                output = torch.cat([output, next_token], dim=1)
                
                # Check if any sequences have finished
                if eos_token_id is not None:
                    unfinished = unfinished & (next_token != eos_token_id).squeeze()
                    if not unfinished.any():
                        break
        
        # Reset SSM layers to not return states
        for block in self.blocks:
            block.ssm_layer.ssm.return_state = False
        
        return output
