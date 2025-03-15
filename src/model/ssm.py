"""
Core State Space Model (SSM) implementation.

This module implements the core SSM layer that forms the foundation of our model.
It combines the discretization methods, HiPPO initialization, and selective parameterization
to create a flexible and powerful sequence modeling primitive.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Union, List
import math

from .discretization import DiscretizationLayer
from .hippo import HiPPOInit
from .selective_param import (
    SelectiveParameterization,
    DenseSelectiveParameterization,
    SparseSelectiveParameterization,
    LowRankSelectiveParameterization
)


class SSMKernel(nn.Module):
    """
    State Space Model kernel that implements the core recurrent operation.
    
    This module implements the discrete-time state space model equations:
    x_t = A x_{t-1} + B u_t
    y_t = C x_t + D u_t
    
    With selective parameterization, A, B, C, D are functions of the input.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        state_dim: int,
        selective_param_class: str = 'dense',
        param_init_method: str = 'xavier',
        discretization_method: str = 'zoh',
        dt_init: float = 0.01,
        dt_min: float = 1e-4,
        dt_max: float = 0.1,
        dt_learnable: bool = True,
        sparsity_level: float = 0.9,
        rank: int = 4,
        dropout: float = 0.0,
        use_bias: bool = True,
        init_scale: float = 0.01,
        return_state: bool = False
    ):
        """
        Initialize the SSM kernel.
        
        Args:
            hidden_dim: Dimension of the hidden representation
            state_dim: Dimension of the state vector
            selective_param_class: Type of selective parameterization ('dense', 'sparse', 'low_rank')
            param_init_method: Method for parameter initialization
            discretization_method: Method for discretization ('zoh', 'bilinear')
            dt_init: Initial time step for discretization
            dt_min: Minimum time step
            dt_max: Maximum time step
            dt_learnable: Whether the time step is learnable
            sparsity_level: Sparsity level for sparse parameterization
            rank: Rank for low-rank parameterization
            dropout: Dropout probability for selective parameterization
            use_bias: Whether to use bias in selective parameterization
            init_scale: Scale for parameter initialization
            return_state: Whether to return the final state
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim
        self.return_state = return_state
        
        # Initialize selective parameterization module
        if selective_param_class == 'dense':
            self.param_generator = DenseSelectiveParameterization(
                input_dim=hidden_dim,
                state_dim=state_dim,
                hidden_dim=hidden_dim,
                dropout=dropout,
                init_scale=init_scale,
                param_init_method=param_init_method,
                use_bias=use_bias
            )
        elif selective_param_class == 'sparse':
            self.param_generator = SparseSelectiveParameterization(
                input_dim=hidden_dim,
                state_dim=state_dim,
                hidden_dim=hidden_dim,
                sparsity_level=sparsity_level,
                dropout=dropout,
                init_scale=init_scale,
                sparse_init_method=param_init_method,
                use_bias=use_bias
            )
        elif selective_param_class == 'low_rank':
            self.param_generator = LowRankSelectiveParameterization(
                input_dim=hidden_dim,
                state_dim=state_dim,
                hidden_dim=hidden_dim,
                rank=rank,
                dropout=dropout,
                init_scale=init_scale,
                use_bias=use_bias
            )
        else:
            raise ValueError(f"Unsupported selective parameterization class: {selective_param_class}")
        
        # Initialize discretization layer
        self.discretizer = DiscretizationLayer(
            method=discretization_method,
            dt_init=dt_init,
            dt_min=dt_min,
            dt_max=dt_max,
            dt_learnable=dt_learnable
        )
    
    def forward(
        self,
        u: torch.Tensor,
        state: Optional[torch.Tensor] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Apply the SSM kernel to the input sequence.
        
        Args:
            u: Input tensor, shape (batch_size, seq_len, hidden_dim)
            state: Initial state, shape (batch_size, state_dim)
                   If None, initialized to zeros
            
        Returns:
            y: Output tensor, shape (batch_size, seq_len, hidden_dim)
            state (optional): Final state, shape (batch_size, state_dim)
        """
        batch_size, seq_len, _ = u.shape
        
        # Initialize state if not provided
        if state is None:
            state = torch.zeros(batch_size, self.state_dim, device=u.device)
        
        # Generate SSM parameters based on input
        params = self.param_generator(u)
        A, B, C, D = params['A'], params['B'], params['C'], params['D']
        
        # Discretize A and B matrices for each time step
        outputs = []
        
        # Loop through sequence for recurrent computation
        for t in range(seq_len):
            # Get parameters for this time step
            A_t = A[:, t]  # (batch_size, state_dim, state_dim)
            B_t = B[:, t]  # (batch_size, state_dim, 1)
            C_t = C[:, t]  # (batch_size, 1, state_dim)
            D_t = D[:, t]  # (batch_size, 1, 1)
            
            # Discretize A_t and B_t
            A_d, B_d = self.discretizer(A_t, B_t)
            
            # Reshape state for batch matrix multiplication
            state = state.unsqueeze(2)  # (batch_size, state_dim, 1)
            
            # Apply SSM equation: x_t = A_d x_{t-1} + B_d u_t
            u_t = u[:, t].unsqueeze(2)  # (batch_size, hidden_dim, 1)
            state = torch.bmm(A_d, state) + torch.bmm(B_d, u_t)
            
            # Apply output equation: y_t = C_t x_t + D_t u_t
            y_t = torch.bmm(C_t, state) + torch.bmm(D_t, u_t)
            outputs.append(y_t.squeeze(2))  # Remove singleton dimension
            
            # Update state for next step
            state = state.squeeze(2)  # (batch_size, state_dim)
        
        # Stack outputs along sequence dimension
        y = torch.stack(outputs, dim=1)  # (batch_size, seq_len, hidden_dim)
        
        if self.return_state:
            return y, state
        else:
            return y


class ParallelSSMKernel(SSMKernel):
    """
    Parallel implementation of the SSM kernel for faster training.
    
    This implementation uses parallel scan algorithms to compute the
    state space model operation more efficiently on GPUs.
    """
    
    def forward(
        self,
        u: torch.Tensor,
        state: Optional[torch.Tensor] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Apply the SSM kernel to the input sequence using parallel scan.
        
        Args:
            u: Input tensor, shape (batch_size, seq_len, hidden_dim)
            state: Initial state, shape (batch_size, state_dim)
                   If None, initialized to zeros
            
        Returns:
            y: Output tensor, shape (batch_size, seq_len, hidden_dim)
            state (optional): Final state, shape (batch_size, state_dim)
        """
        batch_size, seq_len, _ = u.shape
        
        # Initialize state if not provided
        if state is None:
            state = torch.zeros(batch_size, self.state_dim, device=u.device)
        
        # Generate SSM parameters based on input
        params = self.param_generator(u)
        A, B, C, D = params['A'], params['B'], params['C'], params['D']
        
        # Discretize A and B matrices for each time step
        A_d_list = []
        B_d_list = []
        
        for t in range(seq_len):
            A_t = A[:, t]
            B_t = B[:, t]
            A_d, B_d = self.discretizer(A_t, B_t)
            A_d_list.append(A_d)
            B_d_list.append(B_d)
        
        # Stack discretized matrices
        A_d = torch.stack(A_d_list, dim=1)  # (batch_size, seq_len, state_dim, state_dim)
        B_d = torch.stack(B_d_list, dim=1)  # (batch_size, seq_len, state_dim, 1)
        
        # Reshape u for batch matrix multiplication
        u_reshaped = u.unsqueeze(3)  # (batch_size, seq_len, hidden_dim, 1)
        
        # Compute Bu term for each step: B_d_t * u_t
        Bu = torch.matmul(B_d, u_reshaped).squeeze(3)  # (batch_size, seq_len, state_dim)
        
        # Initialize states for all time steps
        states = torch.zeros(batch_size, seq_len + 1, self.state_dim, device=u.device)
        states[:, 0] = state  # Set initial state
        
        # Parallel scan implementation (simplified for readability)
        # For each position, we need to compute: x_t = A_t * x_{t-1} + B_t * u_t
        # But this depends on all previous time steps through the recurrence
        
        # For efficient parallel implementation, we use a modified scan algorithm
        # This is a simplified version - a production implementation would use
        # more optimized parallel scan algorithms
        
        # Forward pass to compute all states
        for t in range(seq_len):
            # states[:, t+1] = torch.bmm(A_d[:, t], states[:, t].unsqueeze(2)).squeeze(2) + Bu[:, t]
            states[:, t+1] = torch.matmul(
                A_d[:, t], 
                states[:, t].unsqueeze(2)
            ).squeeze(2) + Bu[:, t]
        
        # Extract all states except the initial one
        x = states[:, 1:]  # (batch_size, seq_len, state_dim)
        
        # Compute output: y_t = C_t * x_t + D_t * u_t for all t
        # Reshape x for batch matrix multiplication
        x_reshaped = x.unsqueeze(3)  # (batch_size, seq_len, state_dim, 1)
        
        # Compute Cx term
        Cx = torch.matmul(C, x_reshaped).squeeze(3)  # (batch_size, seq_len, 1)
        
        # Compute Du term
        Du = torch.matmul(D, u_reshaped).squeeze(3)  # (batch_size, seq_len, 1)
        
        # Compute output: y = Cx + Du
        y = Cx + Du  # (batch_size, seq_len, 1)
        
        # Final output
        if self.return_state:
            return y, states[:, -1]  # Return output and final state
        else:
            return y  # Return only output


class SSMLayer(nn.Module):
    """
    Complete SSM layer with input/output projections and normalization.
    
    This layer adds input/output projections, normalization, and residual connections
    around the core SSM kernel to create a complete building block for sequence models.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        state_dim: int,
        activation: str = 'gelu',
        layer_norm_eps: float = 1e-5,
        dropout: float = 0.0,
        kernel_args: Dict = None,
        use_parallel_scan: bool = True,
    ):
        """
        Initialize the SSM layer.
        
        Args:
            hidden_dim: Dimension of the hidden representation
            state_dim: Dimension of the state vector
            activation: Activation function ('relu', 'gelu', 'silu')
            layer_norm_eps: Epsilon for layer normalization
            dropout: Dropout probability for the output
            kernel_args: Arguments for the SSM kernel
            use_parallel_scan: Whether to use parallel scan implementation
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim
        
        # Default kernel arguments if not provided
        if kernel_args is None:
            kernel_args = {
                'selective_param_class': 'dense',
                'param_init_method': 'xavier',
                'discretization_method': 'zoh',
                'dt_init': 0.01,
                'dt_learnable': True,
                'init_scale': 0.01,
                'return_state': False
            }
        
        # Input normalization
        self.norm = nn.LayerNorm(hidden_dim, eps=layer_norm_eps)
        
        # Input projection
        self.in_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # SSM kernel
        kernel_class = ParallelSSMKernel if use_parallel_scan else SSMKernel
        self.ssm = kernel_class(
            hidden_dim=hidden_dim,
            state_dim=state_dim,
            dropout=dropout,
            **kernel_args
        )
        
        # Output projection
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'silu':
            self.activation = nn.SiLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
    
    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Apply the SSM layer to the input sequence.
        
        Args:
            x: Input tensor, shape (batch_size, seq_len, hidden_dim)
            state: Initial state, shape (batch_size, state_dim)
                   If None, initialized to zeros
            
        Returns:
            output: Output tensor, shape (batch_size, seq_len, hidden_dim)
            state (optional): Final state if ssm.return_state is True
        """
        # Apply layer normalization
        normalized = self.norm(x)
        
        # Input projection
        u = self.in_proj(normalized)
        
        # Apply activation
        u = self.activation(u)
        
        # Apply SSM
        if state is not None:
            if self.ssm.return_state:
                y, new_state = self.ssm(u, state)
            else:
                y = self.ssm(u, state)
        else:
            if self.ssm.return_state:
                y, new_state = self.ssm(u)
            else:
                y = self.ssm(u)
        
        # Output projection
        output = self.out_proj(y)
        
        # Apply dropout if specified
        if self.dropout is not None:
            output = self.dropout(output)
        
        # Add residual connection
        output = output + x
        
        if self.ssm.return_state:
            return output, new_state
        else:
            return output
