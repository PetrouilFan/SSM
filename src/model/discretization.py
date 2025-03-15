"""
Discretization methods for continuous-time state space models.

This module implements various methods to discretize continuous-time state space models
defined by the equations:
    dx(t)/dt = Ax(t) + Bu(t)
    y(t) = Cx(t) + Du(t)

into discrete-time state space models:
    x_k = Â x_{k-1} + B̂ u_k
    y_k = C x_k + D u_k
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Union, Callable


def discretize_zoh(
    A: torch.Tensor, 
    B: torch.Tensor, 
    dt: Union[float, torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Zero-Order Hold (ZOH) discretization method.
    
    This method assumes the input u(t) is constant over the sampling interval.
    
    Args:
        A: Continuous-time state transition matrix, shape (..., n, n)
        B: Continuous-time input matrix, shape (..., n, m)
        dt: Time step for discretization, scalar or tensor
        
    Returns:
        A_discrete: Discretized state transition matrix
        B_discrete: Discretized input matrix
    """
    if isinstance(dt, torch.Tensor):
        dt = dt.unsqueeze(-1).unsqueeze(-1)  # For broadcasting
    
    n = A.shape[-1]
    
    # Identity matrix for matrix exponential calculation
    I = torch.eye(n, device=A.device).expand_as(A)
    
    # Compute matrix exponential exp(A*dt)
    # For numerical stability, we use the Taylor series approximation
    # exp(A*dt) ≈ I + A*dt + (A*dt)^2/2! + (A*dt)^3/3! + ...
    At = A * dt
    At_power = I
    A_discrete = I.clone()
    
    # Number of terms for approximation
    num_terms = 10
    factorial = 1
    
    for i in range(1, num_terms + 1):
        factorial *= i
        At_power = torch.matmul(At_power, At)
        A_discrete += At_power / factorial
    
    # For B, we need to compute ∫exp(Aτ)dτ * B from 0 to dt
    # We can approximate this as (∫exp(Aτ)dτ) * B
    # The integral is approximately (exp(A*dt) - I) * A^(-1)
    # To avoid computing A^(-1), we use the approximation:
    # B_discrete ≈ (dt * I + dt^2/2 * A + dt^3/6 * A^2 + ...) * B
    
    B_discrete = dt * B
    At_integral = I.clone()
    dt_power = dt
    
    for i in range(1, num_terms):
        dt_power *= dt
        At_integral = torch.matmul(At_integral, At)
        B_discrete += (dt_power / factorial) * torch.matmul(At_integral, B)
    
    return A_discrete, B_discrete


def discretize_bilinear(
    A: torch.Tensor, 
    B: torch.Tensor, 
    dt: Union[float, torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Bilinear (Tustin) discretization method.
    
    This method maps the s-plane to the z-plane using the transform:
    s = 2/dt * (z-1)/(z+1)
    
    Args:
        A: Continuous-time state transition matrix, shape (..., n, n)
        B: Continuous-time input matrix, shape (..., n, m)
        dt: Time step for discretization, scalar or tensor
        
    Returns:
        A_discrete: Discretized state transition matrix
        B_discrete: Discretized input matrix
    """
    if isinstance(dt, torch.Tensor):
        dt = dt.unsqueeze(-1).unsqueeze(-1)  # For broadcasting
    
    n = A.shape[-1]
    
    # Identity matrix
    I = torch.eye(n, device=A.device).expand_as(A)
    
    # Compute (I - dt/2 * A)^(-1)
    inv_term = torch.inverse(I - (dt/2) * A)
    
    # Compute A_discrete = (I - dt/2 * A)^(-1) * (I + dt/2 * A)
    A_discrete = torch.matmul(inv_term, I + (dt/2) * A)
    
    # Compute B_discrete = (I - dt/2 * A)^(-1) * dt * B
    B_discrete = torch.matmul(inv_term, dt * B)
    
    return A_discrete, B_discrete


def discretize_generalized_bilinear(
    A: torch.Tensor, 
    B: torch.Tensor, 
    dt: Union[float, torch.Tensor],
    alpha: float = 0.5
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generalized bilinear discretization method.
    
    This method is a generalization of the bilinear transform with a parameter alpha:
    - alpha = 0: Forward Euler (explicit)
    - alpha = 0.5: Tustin's method (bilinear)
    - alpha = 1: Backward Euler (implicit)
    
    Args:
        A: Continuous-time state transition matrix, shape (..., n, n)
        B: Continuous-time input matrix, shape (..., n, m)
        dt: Time step for discretization, scalar or tensor
        alpha: Interpolation parameter (0 to 1)
        
    Returns:
        A_discrete: Discretized state transition matrix
        B_discrete: Discretized input matrix
    """
    if isinstance(dt, torch.Tensor):
        dt = dt.unsqueeze(-1).unsqueeze(-1)  # For broadcasting
    
    n = A.shape[-1]
    
    # Identity matrix
    I = torch.eye(n, device=A.device).expand_as(A)
    
    # Compute (I - alpha*dt*A)^(-1)
    inv_term = torch.inverse(I - alpha * dt * A)
    
    # Compute A_discrete = (I - alpha*dt*A)^(-1) * (I + (1-alpha)*dt*A)
    A_discrete = torch.matmul(inv_term, I + (1 - alpha) * dt * A)
    
    # Compute B_discrete = (I - alpha*dt*A)^(-1) * dt * B
    B_discrete = torch.matmul(inv_term, dt * B)
    
    return A_discrete, B_discrete


class DiscretizationLayer(nn.Module):
    """
    Layer for discretizing continuous-time state space models.
    
    This layer implements different discretization methods and can handle
    learnable time steps (dt).
    """
    
    METHODS = {
        'zoh': discretize_zoh,
        'bilinear': discretize_bilinear,
        'generalized_bilinear': discretize_generalized_bilinear
    }
    
    def __init__(
        self,
        method: str = 'zoh',
        dt_init: float = 0.01,
        dt_min: float = 1e-4,
        dt_max: float = 0.1,
        dt_learnable: bool = True,
        alpha: float = 0.5,
    ):
        """
        Initialize discretization layer.
        
        Args:
            method: Discretization method ('zoh', 'bilinear', 'generalized_bilinear')
            dt_init: Initial time step
            dt_min: Minimum allowed time step (for stability)
            dt_max: Maximum allowed time step
            dt_learnable: Whether dt is a learnable parameter
            alpha: Parameter for generalized bilinear method
        """
        super().__init__()
        
        if method not in self.METHODS:
            raise ValueError(f"Unsupported discretization method: {method}. "
                            f"Choose from {list(self.METHODS.keys())}")
        
        self.method = method
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.alpha = alpha
        
        # Initialize dt as a learnable parameter or buffer
        dt_init = torch.tensor(dt_init)
        if dt_learnable:
            # For learnable dt, we use a parameterization that ensures dt stays within bounds
            # We use sigmoid(p) * (dt_max - dt_min) + dt_min
            p_init = torch.logit((dt_init - dt_min) / (dt_max - dt_min))
            self.register_parameter('p', nn.Parameter(p_init))
        else:
            self.register_buffer('dt', dt_init)
    
    @property
    def dt(self) -> torch.Tensor:
        """Get the current time step value."""
        if hasattr(self, 'p'):
            # For learnable dt, apply sigmoid and scale to [dt_min, dt_max]
            return torch.sigmoid(self.p) * (self.dt_max - self.dt_min) + self.dt_min
        else:
            return self.dt
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Discretize continuous-time matrices A and B.
        
        Args:
            A: Continuous-time state transition matrix, shape (..., n, n)
            B: Continuous-time input matrix, shape (..., n, m)
            
        Returns:
            A_discrete: Discretized state transition matrix
            B_discrete: Discretized input matrix
        """
        if self.method == 'generalized_bilinear':
            return discretize_generalized_bilinear(A, B, self.dt, self.alpha)
        else:
            return self.METHODS[self.method](A, B, self.dt)
