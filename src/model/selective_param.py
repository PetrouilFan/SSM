"""
Selective Parameterization for State Space Models.

This module implements selective parameterization techniques for SSMs, where model
parameters (A, B, C, D matrices) are functions of the input. This allows the model
to dynamically adjust its internal dynamics based on the current input.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Union, Dict, Any, List
import math


class SelectiveParameterization(nn.Module):
    """
    Base class for selective parameterization modules.
    
    This class defines the interface for selective parameterization modules
    that generate SSM parameters based on input.
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Generate parameters based on input.
        
        Args:
            x: Input tensor, shape (batch_size, seq_len, input_dim)
            
        Returns:
            Dictionary containing generated parameters (A, B, C, D)
        """
        raise NotImplementedError("Subclasses must implement forward method")


class InputProjection(nn.Module):
    """
    Module for projecting input to parameter space.
    
    This module projects the input to a hidden representation that will be 
    used to generate the SSM parameters.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout: float = 0.0,
        activation: str = 'gelu',
        layer_norm: bool = True
    ):
        """
        Initialize input projection module.
        
        Args:
            input_dim: Dimension of the input
            hidden_dim: Dimension of the hidden representation
            dropout: Dropout probability
            activation: Activation function ('relu', 'gelu', 'silu')
            layer_norm: Whether to use layer normalization
        """
        super().__init__()
        
        self.layer_norm = nn.LayerNorm(input_dim) if layer_norm else None
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        
        self.linear = nn.Linear(input_dim, hidden_dim)
        
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
        Project input to hidden representation.
        
        Args:
            x: Input tensor, shape (batch_size, seq_len, input_dim)
            
        Returns:
            Hidden representation, shape (batch_size, seq_len, hidden_dim)
        """
        if self.layer_norm is not None:
            x = self.layer_norm(x)
        
        if self.dropout is not None:
            x = self.dropout(x)
        
        x = self.linear(x)
        x = self.activation(x)
        
        return x


class DenseSelectiveParameterization(SelectiveParameterization):
    """
    Dense selective parameterization module.
    
    This module generates dense SSM parameters based on the input using
    fully connected layers.
    """
    
    def __init__(
        self,
        input_dim: int,
        state_dim: int,
        hidden_dim: int,
        param_proj_dim: Optional[int] = None,
        dropout: float = 0.0,
        init_scale: float = 0.01,
        param_init_method: str = 'xavier',
        use_bias: bool = True
    ):
        """
        Initialize dense selective parameterization module.
        
        Args:
            input_dim: Dimension of the input
            state_dim: Dimension of the state
            hidden_dim: Dimension of the hidden representation
            param_proj_dim: Dimension of the parameter projection (if None, use hidden_dim)
            dropout: Dropout probability
            init_scale: Scale for parameter initialization
            param_init_method: Method for parameter initialization ('xavier', 'normal', 'zeros')
            use_bias: Whether to use bias in parameter projections
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.param_proj_dim = param_proj_dim if param_proj_dim is not None else hidden_dim
        self.init_scale = init_scale
        
        # Input projection
        self.input_proj = InputProjection(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            activation='gelu',
            layer_norm=True
        )
        
        # Parameter projections
        # A matrix: state_dim x state_dim
        self.A_proj = nn.Linear(self.param_proj_dim, state_dim * state_dim, bias=use_bias)
        
        # B matrix: state_dim x 1
        self.B_proj = nn.Linear(self.param_proj_dim, state_dim, bias=use_bias)
        
        # C matrix: hidden_dim x state_dim
        self.C_proj = nn.Linear(self.param_proj_dim, hidden_dim * state_dim, bias=use_bias)
        
        # D matrix: hidden_dim x hidden_dim
        self.D_proj = nn.Linear(self.param_proj_dim, hidden_dim * hidden_dim, bias=use_bias)
        
        # Initialize parameter projections
        self._init_parameters(param_init_method)
    
    def _init_parameters(self, method: str):
        """
        Initialize parameter projection weights.
        
        Args:
            method: Initialization method ('xavier', 'normal', 'zeros')
        """
        if method == 'xavier':
            # Xavier/Glorot initialization
            nn.init.xavier_uniform_(self.A_proj.weight, gain=self.init_scale)
            nn.init.xavier_uniform_(self.B_proj.weight, gain=self.init_scale)
            nn.init.xavier_uniform_(self.C_proj.weight, gain=self.init_scale)
            nn.init.xavier_uniform_(self.D_proj.weight, gain=self.init_scale)
        elif method == 'normal':
            # Normal initialization
            nn.init.normal_(self.A_proj.weight, std=self.init_scale)
            nn.init.normal_(self.B_proj.weight, std=self.init_scale)
            nn.init.normal_(self.C_proj.weight, std=self.init_scale)
            nn.init.normal_(self.D_proj.weight, std=self.init_scale)
        elif method == 'zeros':
            # Zero initialization with a small amount of noise
            nn.init.normal_(self.A_proj.weight, std=1e-4)
            nn.init.normal_(self.B_proj.weight, std=1e-4)
            nn.init.normal_(self.C_proj.weight, std=1e-4)
            nn.init.normal_(self.D_proj.weight, std=1e-4)
        else:
            raise ValueError(f"Unsupported initialization method: {method}")
        
        # Initialize biases to zero if they exist
        if self.A_proj.bias is not None:
            nn.init.zeros_(self.A_proj.bias)
        if self.B_proj.bias is not None:
            nn.init.zeros_(self.B_proj.bias)
        if self.C_proj.bias is not None:
            nn.init.zeros_(self.C_proj.bias)
        if self.D_proj.bias is not None:
            nn.init.zeros_(self.D_proj.bias)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Generate SSM parameters based on input.
        
        Args:
            x: Input tensor, shape (batch_size, seq_len, input_dim)
            
        Returns:
            Dictionary containing generated parameters:
                - A: shape (batch_size, seq_len, state_dim, state_dim)
                - B: shape (batch_size, seq_len, state_dim, hidden_dim)
                - C: shape (batch_size, seq_len, hidden_dim, state_dim)
                - D: shape (batch_size, seq_len, hidden_dim, hidden_dim)
        """
        batch_size, seq_len, _ = x.shape
        
        # Project input to hidden representation
        h = self.input_proj(x)  # (batch_size, seq_len, hidden_dim)
        
        # Generate SSM parameters
        A = self.A_proj(h).view(batch_size, seq_len, self.state_dim, self.state_dim)
        B = self.B_proj(h).view(batch_size, seq_len, self.state_dim, self.hidden_dim)
        C = self.C_proj(h).view(batch_size, seq_len, self.hidden_dim, self.state_dim)
        D = self.D_proj(h).view(batch_size, seq_len, self.hidden_dim, self.hidden_dim)
        
        return {'A': A, 'B': B, 'C': C, 'D': D}


class SparseSelectiveParameterization(SelectiveParameterization):
    """
    Sparse selective parameterization module.
    
    This module generates sparse SSM parameters based on the input,
    with structured sparsity in the state transition matrix A.
    """
    
    def __init__(
        self,
        input_dim: int,
        state_dim: int,
        hidden_dim: int,
        sparsity_level: float = 0.9,
        param_proj_dim: Optional[int] = None,
        dropout: float = 0.0,
        init_scale: float = 0.01,
        sparse_init_method: str = 'normal',
        use_structured_sparsity: bool = True,
        use_bias: bool = True
    ):
        """
        Initialize sparse selective parameterization module.
        
        Args:
            input_dim: Dimension of the input
            state_dim: Dimension of the state
            hidden_dim: Dimension of the hidden representation
            sparsity_level: Target sparsity level (0.0-1.0, higher means more sparse)
            param_proj_dim: Dimension of the parameter projection (if None, use hidden_dim)
            dropout: Dropout probability
            init_scale: Scale for parameter initialization
            sparse_init_method: Method for sparse parameter initialization ('normal', 'uniform')
            use_structured_sparsity: Whether to use structured sparsity patterns
            use_bias: Whether to use bias in parameter projections
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.param_proj_dim = param_proj_dim if param_proj_dim is not None else hidden_dim
        self.sparsity_level = sparsity_level
        self.init_scale = init_scale
        self.use_structured_sparsity = use_structured_sparsity
        
        # Input projection
        self.input_proj = InputProjection(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            activation='gelu',
            layer_norm=True
        )
        
        # Create structured sparsity mask for A
        if use_structured_sparsity:
            self.A_mask = self._create_structured_sparsity_mask(state_dim, sparsity_level)
        else:
            self.A_mask = self._create_random_sparsity_mask(state_dim, sparsity_level)
        
        # Register buffer for mask
        self.register_buffer('_A_mask', self.A_mask)
        
        # Parameter projections with reduced output dimension due to sparsity
        A_nnz = int(self.A_mask.sum().item())  # Number of non-zero elements in A
        self.A_proj = nn.Linear(self.param_proj_dim, A_nnz, bias=use_bias)
        
        # B matrix: state_dim x 1
        self.B_proj = nn.Linear(self.param_proj_dim, state_dim * hidden_dim, bias=use_bias)
        
        # C matrix: hidden_dim x state_dim
        self.C_proj = nn.Linear(self.param_proj_dim, hidden_dim * state_dim, bias=use_bias)
        
        # D matrix: hidden_dim x hidden_dim
        self.D_proj = nn.Linear(self.param_proj_dim, hidden_dim * hidden_dim, bias=use_bias)
        
        # Initialize parameters
        self._init_parameters(sparse_init_method)
    
    def _create_structured_sparsity_mask(self, dim: int, sparsity_level: float) -> torch.Tensor:
        """
        Create a structured sparsity mask for the A matrix.
        
        This method creates structured sparsity patterns that work well on GPU
        architectures, like block-diagonal or band-diagonal patterns.
        
        Args:
            dim: Dimension of the state (A is dim x dim)
            sparsity_level: Target sparsity level (0.0-1.0)
            
        Returns:
            Boolean mask for A matrix
        """
        mask = torch.zeros(dim, dim, dtype=torch.bool)
        
        # Determine structure type based on sparsity level
        if sparsity_level > 0.9:
            # Ultra-sparse: Only diagonal elements
            for i in range(dim):
                mask[i, i] = True
        elif sparsity_level > 0.7:
            # Very sparse: Tri-diagonal structure
            for i in range(dim):
                for j in range(max(0, i-1), min(dim, i+2)):
                    mask[i, j] = True
        elif sparsity_level > 0.5:
            # Moderately sparse: Band diagonal with occasional connections
            band_width = max(1, int(dim * (1 - sparsity_level) / 2))
            for i in range(dim):
                for j in range(max(0, i-band_width), min(dim, i+band_width+1)):
                    mask[i, j] = True
                    
            # Add some structured long-range connections for expressivity
            for i in range(0, dim, max(1, int(dim / 5))):
                for j in range(0, dim, max(1, int(dim / 5))):
                    mask[i, j] = True
        else:
            # Low sparsity: Block diagonal with blocks of size block_size
            block_size = max(2, int(dim * (1 - sparsity_level / 2)))
            num_blocks = math.ceil(dim / block_size)
            
            for b in range(num_blocks):
                start_idx = b * block_size
                end_idx = min((b + 1) * block_size, dim)
                
                for i in range(start_idx, end_idx):
                    for j in range(start_idx, end_idx):
                        mask[i, j] = True
        
        # Ensure target sparsity level
        target_nnz = int(dim * dim * (1 - sparsity_level))
        current_nnz = mask.sum().item()
        
        if current_nnz < target_nnz:
            # Need to add more non-zero elements
            add_count = target_nnz - current_nnz
            indices = torch.nonzero(~mask)
            perm = torch.randperm(indices.shape[0])
            for idx in range(min(add_count, indices.shape[0])):
                i, j = indices[perm[idx]]
                mask[i, j] = True
        elif current_nnz > target_nnz:
            # Need to remove some non-zero elements
            remove_count = current_nnz - target_nnz
            indices = torch.nonzero(mask)
            perm = torch.randperm(indices.shape[0])
            for idx in range(min(remove_count, indices.shape[0])):
                i, j = indices[perm[idx]]
                # Don't remove diagonal elements to maintain stability
                if i != j:
                    mask[i, j] = False
        
        return mask
    
    def _create_random_sparsity_mask(self, dim: int, sparsity_level: float) -> torch.Tensor:
        """
        Create a random sparsity mask for the A matrix.
        
        Args:
            dim: Dimension of the state (A is dim x dim)
            sparsity_level: Target sparsity level (0.0-1.0)
            
        Returns:
            Boolean mask for A matrix
        """
        # Create random mask with target density
        mask = torch.zeros(dim, dim, dtype=torch.bool)
        nnz = int(dim * dim * (1 - sparsity_level))
        
        # Set diagonal elements to ensure stability
        for i in range(dim):
            mask[i, i] = True
        
        # Fill rest of mask randomly
        remaining = nnz - dim
        if remaining > 0:
            # Create off-diagonal indices
            indices = []
            for i in range(dim):
                for j in range(dim):
                    if i != j:
                        indices.append((i, j))
            
            # Randomly select indices to set to True
            perm = torch.randperm(len(indices))
            for idx in range(min(remaining, len(indices))):
                i, j = indices[perm[idx]]
                mask[i, j] = True
        
        return mask
    
    def _init_parameters(self, method: str):
        """
        Initialize parameter projection weights.
        
        Args:
            method: Initialization method ('normal', 'uniform')
        """
        if method == 'normal':
            # Normal initialization
            nn.init.normal_(self.A_proj.weight, std=self.init_scale)
            nn.init.normal_(self.B_proj.weight, std=self.init_scale)
            nn.init.normal_(self.C_proj.weight, std=self.init_scale)
            nn.init.normal_(self.D_proj.weight, std=self.init_scale)
        elif method == 'uniform':
            # Uniform initialization
            nn.init.uniform_(self.A_proj.weight, -self.init_scale, self.init_scale)
            nn.init.uniform_(self.B_proj.weight, -self.init_scale, self.init_scale)
            nn.init.uniform_(self.C_proj.weight, -self.init_scale, self.init_scale)
            nn.init.uniform_(self.D_proj.weight, -self.init_scale, self.init_scale)
        else:
            raise ValueError(f"Unsupported initialization method: {method}")
        
        # Initialize biases to zero if they exist
        if self.A_proj.bias is not None:
            nn.init.zeros_(self.A_proj.bias)
        if self.B_proj.bias is not None:
            nn.init.zeros_(self.B_proj.bias)
        if self.C_proj.bias is not None:
            nn.init.zeros_(self.C_proj.bias)
        if self.D_proj.bias is not None:
            nn.init.zeros_(self.D_proj.bias)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Generate sparse SSM parameters based on input (optimized version).

        Args:
            x: Input tensor, shape (batch_size, seq_len, input_dim)

        Returns:
            Dictionary containing generated parameters:
                - A: shape (batch_size, seq_len, state_dim, state_dim)
                - B: shape (batch_size, seq_len, state_dim, hidden_dim)
                - C: shape (batch_size, seq_len, hidden_dim, state_dim)
                - D: shape (batch_size, seq_len, hidden_dim, hidden_dim)
        """
        batch_size, seq_len, _ = x.shape

        # Project input to hidden representation
        h = self.input_proj(x)  # (batch_size, seq_len, hidden_dim)

        # Generate non-zero elements for A
        A_nonzero = self.A_proj(h)  # (batch_size, seq_len, A_nnz)

        # Convert to sparse A matrix using the mask
        A = torch.zeros(batch_size, seq_len, self.state_dim, self.state_dim,
                       device=x.device, dtype=x.dtype)

        # Get indices of non-zero elements in mask
        nonzero_idx = torch.nonzero(self._A_mask)
        A_nnz = nonzero_idx.size(0)

        # Create batch and sequence index tensors
        batch_idx = torch.arange(batch_size, device=x.device).view(-1, 1, 1).expand(-1, seq_len, A_nnz).flatten()
        seq_idx = torch.arange(seq_len, device=x.device).view(1, -1, 1).expand(batch_size, -1, A_nnz).flatten()
        row_idx = nonzero_idx[:, 0].view(1, 1, -1).expand(batch_size, seq_len, -1).flatten()
        col_idx = nonzero_idx[:, 1].view(1, 1, -1).expand(batch_size, seq_len, -1).flatten()

        # Use index_put_ for efficient sparse assignment
        indices = torch.stack([batch_idx, seq_idx, row_idx, col_idx], dim=0)
        A.index_put_((indices[0], indices[1], indices[2], indices[3]), A_nonzero.flatten())


        # Generate B, C, and D normally
        B = self.B_proj(h).view(batch_size, seq_len, self.state_dim, self.hidden_dim)
        C = self.C_proj(h).view(batch_size, seq_len, self.hidden_dim, self.state_dim)
        D = self.D_proj(h).view(batch_size, seq_len, self.hidden_dim, self.hidden_dim)

        return {'A': A, 'B': B, 'C': C, 'D': D}


class LowRankSelectiveParameterization(SelectiveParameterization):
    """
    Low-rank selective parameterization module.
    
    This module generates SSM parameters using low-rank approximations, which
    can significantly reduce the parameter count while maintaining expressivity.
    """
    
    def __init__(
        self,
        input_dim: int,
        state_dim: int,
        hidden_dim: int,
        rank: int = 4,
        param_proj_dim: Optional[int] = None,
        dropout: float = 0.0,
        init_scale: float = 0.01,
        use_bias: bool = True
    ):
        """
        Initialize low-rank selective parameterization module.
        
        Args:
            input_dim: Dimension of the input
            state_dim: Dimension of the state
            hidden_dim: Dimension of the hidden representation
            rank: Rank of the low-rank approximation
            param_proj_dim: Dimension of the parameter projection (if None, use hidden_dim)
            dropout: Dropout probability
            init_scale: Scale for parameter initialization
            use_bias: Whether to use bias in parameter projections
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.param_proj_dim = param_proj_dim if param_proj_dim is not None else hidden_dim
        self.rank = rank
        self.init_scale = init_scale
        
        # Input projection
        self.input_proj = InputProjection(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            activation='gelu',
            layer_norm=True
        )
        
        # For A, we use a low-rank approximation A = U * V + diagonal
        # Where U is (state_dim x rank) and V is (rank x state_dim)
        self.A_U_proj = nn.Linear(self.param_proj_dim, state_dim * rank, bias=use_bias)
        self.A_V_proj = nn.Linear(self.param_proj_dim, rank * state_dim, bias=use_bias)
        self.A_diag_proj = nn.Linear(self.param_proj_dim, state_dim, bias=use_bias)
        
        # B matrix: state_dim x 1
        self.B_proj = nn.Linear(self.param_proj_dim, state_dim * hidden_dim, bias=use_bias)
        
        # C matrix: hidden_dim x state_dim
        self.C_proj = nn.Linear(self.param_proj_dim, hidden_dim * state_dim, bias=use_bias)
        
        # D matrix: hidden_dim x hidden_dim
        self.D_proj = nn.Linear(self.param_proj_dim, hidden_dim * hidden_dim, bias=use_bias)
        
        # Initialize parameters
        self._init_parameters()
    
    def _init_parameters(self):
        """Initialize parameter projection weights."""
        # Xavier initialization with appropriate scaling
        for proj in [self.A_U_proj, self.A_V_proj, self.A_diag_proj, 
                     self.B_proj, self.C_proj, self.D_proj]:
            nn.init.xavier_uniform_(proj.weight, gain=self.init_scale)
            if proj.bias is not None:
                nn.init.zeros_(proj.bias)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Generate low-rank SSM parameters based on input.
        
        Args:
            x: Input tensor, shape (batch_size, seq_len, input_dim)
            
        Returns:
            Dictionary containing generated parameters:
                - A: shape (batch_size, seq_len, state_dim, state_dim)
                - B: shape (batch_size, seq_len, state_dim, hidden_dim)
                - C: shape (batch_size, seq_len, hidden_dim, state_dim)
                - D: shape (batch_size, seq_len, hidden_dim, hidden_dim)
        """
        batch_size, seq_len, _ = x.shape
        
        # Project input to hidden representation
        h = self.input_proj(x)  # (batch_size, seq_len, hidden_dim)
        
        # Generate low-rank components for A
        A_U = self.A_U_proj(h).view(batch_size, seq_len, self.state_dim, self.rank)
        A_V = self.A_V_proj(h).view(batch_size, seq_len, self.rank, self.state_dim)
        A_diag = self.A_diag_proj(h).view(batch_size, seq_len, self.state_dim)
        
        # Compute A = U * V + diagonal
        A = torch.matmul(A_U, A_V)  # Low-rank term
        
        # Add diagonal term for stability
        batch_indices = torch.arange(batch_size).view(-1, 1, 1).expand(-1, seq_len, self.state_dim)
        seq_indices = torch.arange(seq_len).view(1, -1, 1).expand(batch_size, -1, self.state_dim)
        diag_indices = torch.arange(self.state_dim).view(1, 1, -1).expand(batch_size, seq_len, -1)
        
        A[batch_indices, seq_indices, diag_indices, diag_indices] += A_diag
        
        # Generate B, C, and D normally
        B = self.B_proj(h).view(batch_size, seq_len, self.state_dim, self.hidden_dim)
        C = self.C_proj(h).view(batch_size, seq_len, self.hidden_dim, self.state_dim)
        D = self.D_proj(h).view(batch_size, seq_len, self.hidden_dim, self.hidden_dim)
        
        return {'A': A, 'B': B, 'C': C, 'D': D}
