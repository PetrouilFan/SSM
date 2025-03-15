"""
HiPPO (Higher Order Polynomial Projection Operator) initialization methods.

This module implements various HiPPO initialization methods for state space models.
HiPPO provides mathematically principled initialization for capturing long-range
dependencies in sequence data.

References:
- HiPPO: Recurrent Memory with Optimal Polynomial Projections (NeurIPS 2020)
- Efficiently Modeling Long Sequences with Structured State Spaces (ICLR 2022)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, Union, Callable


def make_hippo_legs(N: int, normalize: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Construct HiPPO-LegS (Legendre) matrices.
    
    This method constructs matrices A and B based on Legendre polynomials,
    which provide a principled foundation for modeling continuous-time functions.
    
    Args:
        N: State dimension
        normalize: Whether to normalize the matrices
        
    Returns:
        A: HiPPO state transition matrix (N x N)
        B: HiPPO input projection matrix (N x 1)
    """
    # Construct the HiPPO-LegS matrices based on the mathematical formulation
    # A_{n,k} = (2n+1)^{1/2} * (2k+1)^{1/2} * int_0^1 P_n(x) d/dx P_k(x) dx
    # where P_n is the Legendre polynomial of degree n
    
    # Pre-compute some constants for numerical stability
    Q = np.arange(N, dtype=np.float64)
    R = (2 * Q + 1) ** 0.5
    
    # Construct the component matrices for A
    j, i = np.meshgrid(Q, Q)
    A = np.zeros((N, N), dtype=np.float64)
    
    # Only need to compute upper triangular part
    mask = i < j
    A[mask] = -1.0 * R[i[mask]] * R[j[mask]]
    
    # Using the symmetry of the matrix to fill in the lower triangular part
    mask = i > j
    A[mask] = 1.0 * R[i[mask]] * R[j[mask]]
    
    # Diagonal is zero for HiPPO-LegS
    
    # B vector for the input projection
    B = np.zeros(N, dtype=np.float64)
    B[0::2] = (2 * np.arange(N // 2 + N % 2) + 1) ** 0.5 * 2
    
    if normalize:
        # Optional normalization for better stability
        norm = np.sqrt(N / 2)
        A = A / norm
        B = B / norm
    
    # Convert to PyTorch tensors
    A = torch.from_numpy(A).float()
    B = torch.from_numpy(B).float().unsqueeze(1)  # Shape (N, 1)
    
    return A, B


def make_hippo_legt(N: int, normalize: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Construct HiPPO-LegT (Translated Legendre) matrices.
    
    The translated Legendre basis maps the time interval [0, T] to [-1, 1]
    for the recurrent updates, which can be more stable.
    
    Args:
        N: State dimension
        normalize: Whether to normalize the matrices
        
    Returns:
        A: HiPPO state transition matrix (N x N)
        B: HiPPO input projection matrix (N x 1)
    """
    # Pre-compute coefficients
    Q = np.arange(N, dtype=np.float64)
    R = (2 * Q + 1)[:, None] ** 0.5
    S = (2 * Q + 1)[None, :] ** 0.5
    
    # Construct A matrix
    i, j = np.meshgrid(Q, Q)
    A = np.zeros((N, N), dtype=np.float64)
    
    # Upper triangular part
    mask = i < j
    A[mask] = R[i[mask]] * S[mask] * (-1) ** (i[mask] - j[mask])
    
    # Lower triangular part
    mask = i > j
    A[mask] = R[i[mask]] * S[mask] * (-1) ** (i[mask] - j[mask]) * -1
    
    # Diagonal part
    A += np.diag(-Q - 0.5)
    
    # B vector
    B = np.zeros(N, dtype=np.float64)
    B[0::2] = (2 * np.arange(N // 2 + N % 2) + 1) ** 0.5 * 2
    
    if normalize:
        # Optional normalization for better stability
        norm = np.sqrt(N)
        A = A / norm
        B = B / norm
    
    # Convert to PyTorch tensors
    A = torch.from_numpy(A).float()
    B = torch.from_numpy(B).float().unsqueeze(1)  # Shape (N, 1)
    
    return A, B


def make_hippo_fourier(N: int, normalize: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Construct HiPPO-Fourier matrices using frequency-based initialization.
    
    This basis uses Fourier modes for capturing periodic patterns in the data.
    
    Args:
        N: State dimension
        normalize: Whether to normalize the matrices
        
    Returns:
        A: HiPPO state transition matrix (N x N)
        B: HiPPO input projection matrix (N x 1)
    """
    # For even N, we use N/2 complex-valued oscillators with different frequencies
    # We represent them using real matrices of size N x N
    
    # Ensure N is even for Fourier basis
    if N % 2 == 1:
        N = N - 1
    
    # Create frequencies for the oscillators
    freqs = np.arange(1, N // 2 + 1, dtype=np.float64)
    
    # Create block diagonal matrix with 2x2 blocks representing oscillators
    A = np.zeros((N, N), dtype=np.float64)
    for i in range(N // 2):
        freq = freqs[i]
        block = np.array([
            [0, freq],
            [-freq, 0]
        ])
        A[2*i:2*i+2, 2*i:2*i+2] = block
    
    # Create B vector which projects the input to each oscillator
    B = np.zeros(N, dtype=np.float64)
    B[0::2] = 1.0
    
    if normalize:
        # Optional normalization for better stability
        norm = np.sqrt(N / 2)
        A = A / norm
        B = B / norm
    
    # Convert to PyTorch tensors
    A = torch.from_numpy(A).float()
    B = torch.from_numpy(B).float().unsqueeze(1)  # Shape (N, 1)
    
    return A, B


def make_hippo_scaled_rotation(
    N: int,
    theta_interval: Tuple[float, float] = (0.001, 0.1),
    gamma: float = 0.99,
    normalize: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Construct scaled rotation matrices for initialization.
    
    This version uses a scaled rotation matrix approach which can be more
    numerically stable and provides control over the frequency range.
    
    Args:
        N: State dimension
        theta_interval: Interval for frequency initialization (min, max)
        gamma: Decay factor for stability (< 1)
        normalize: Whether to normalize the matrices
        
    Returns:
        A: Scaled rotation state transition matrix (N x N)
        B: Input projection matrix (N x 1)
    """
    # Ensure N is even
    if N % 2 == 1:
        N = N - 1
    
    # Generate log-spaced frequencies in the given interval
    theta_min, theta_max = theta_interval
    log_min, log_max = np.log(theta_min), np.log(theta_max)
    freqs = np.exp(np.linspace(log_min, log_max, N // 2))
    
    # Create block diagonal matrix with 2x2 rotation blocks
    A = np.zeros((N, N), dtype=np.float64)
    for i in range(N // 2):
        theta = freqs[i]
        # Scaled rotation block with decay factor gamma
        block = gamma * np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])
        A[2*i:2*i+2, 2*i:2*i+2] = block
    
    # B vector for input projection
    B = np.zeros(N, dtype=np.float64)
    B[0::2] = 1.0  # Project input only to the first component of each oscillator
    
    if normalize:
        # Optional normalization for better stability
        norm = np.sqrt(N / 2)
        A = A / norm
        B = B / norm * (1 - gamma**2)**0.5  # Normalize to preserve energy
    
    # Convert to PyTorch tensors
    A = torch.from_numpy(A).float()
    B = torch.from_numpy(B).float().unsqueeze(1)  # Shape (N, 1)
    
    return A, B


class HiPPOInit(nn.Module):
    """
    Module for initializing SSM parameters using HiPPO methods.
    
    This module provides various initialization methods based on the HiPPO framework
    for state space models.
    """
    
    METHODS = {
        'legs': make_hippo_legs,
        'legt': make_hippo_legt,
        'fourier': make_hippo_fourier,
        'scaled_rotation': make_hippo_scaled_rotation
    }
    
    def __init__(
        self,
        state_dim: int,
        method: str = 'legs',
        normalize: bool = True,
        trainable: bool = True,
        **kwargs
    ):
        """
        Initialize the HiPPOInit module.
        
        Args:
            state_dim: Dimension of the state vector
            method: Initialization method ('legs', 'legt', 'fourier', 'scaled_rotation')
            normalize: Whether to normalize the matrices
            trainable: Whether the matrices are trainable
            **kwargs: Additional arguments for specific initialization methods
        """
        super().__init__()
        
        if method not in self.METHODS:
            raise ValueError(f"Unsupported HiPPO method: {method}. "
                            f"Choose from {list(self.METHODS.keys())}")
        
        # Initialize A and B matrices using the selected method
        init_fn = self.METHODS[method]
        if method == 'scaled_rotation':
            A, B = init_fn(state_dim, normalize=normalize, **kwargs)
        else:
            A, B = init_fn(state_dim, normalize=normalize)
        
        # Register parameters or buffers based on trainability
        if trainable:
            self.A = nn.Parameter(A)
            self.B = nn.Parameter(B)
        else:
            self.register_buffer('A', A)
            self.register_buffer('B', B)
    
    def forward(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return the initialized A and B matrices.
        
        Returns:
            A: State transition matrix
            B: Input projection matrix
        """
        return self.A, self.B
