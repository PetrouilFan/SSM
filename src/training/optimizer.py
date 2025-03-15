"""
Optimizer configurations for training Sparse SSM models.

This module provides optimizer configurations and learning rate schedulers
optimized for training SSM models efficiently.
"""

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import (
    LambdaLR,
    CosineAnnealingLR,
    LinearLR,
    SequentialLR,
    CosineAnnealingWarmRestarts
)
from typing import Dict, Optional, List, Union, Callable, Any, Tuple
import math


def get_linear_warmup_cosine_decay_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    max_steps: int,
    min_lr_ratio: float = 0.1,
    warmup_init_lr_ratio: float = 0.0
) -> SequentialLR:
    """
    Get a learning rate scheduler with linear warmup and cosine decay.
    
    This scheduler applies a linear warmup followed by a cosine decay,
    which is a common strategy for training large language models.
    
    Args:
        optimizer: PyTorch optimizer
        warmup_steps: Number of warmup steps
        max_steps: Total number of training steps
        min_lr_ratio: Minimum learning rate ratio at the end of training
        warmup_init_lr_ratio: Initial learning rate ratio during warmup
        
    Returns:
        SequentialLR scheduler combining linear warmup and cosine decay
    """
    # Ensure valid warmup_init_lr_ratio (must be between 0 and 1)
    start_factor = max(0.0, min(1.0, warmup_init_lr_ratio))
    
    # Create linear warmup scheduler
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=start_factor,
        end_factor=1.0,
        total_iters=warmup_steps
    )
    
    # Create cosine decay scheduler
    cosine_steps = max_steps - warmup_steps
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=cosine_steps,
        eta_min=min_lr_ratio
    )
    
    # Combine the schedulers
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps]
    )
    
    return scheduler


def get_cosine_with_restarts_scheduler(
    optimizer: torch.optim.Optimizer,
    first_cycle_steps: int,
    cycle_mult: float = 1.0,
    max_steps: Optional[int] = None,
    min_lr_ratio: float = 0.1,
    warmup_steps: int = 0,
    warmup_init_lr_ratio: float = 0.0
) -> Union[CosineAnnealingWarmRestarts, SequentialLR]:
    """
    Get a cosine annealing scheduler with warm restarts and optional warmup.
    
    This scheduler applies cosine annealing with warm restarts, which can
    help the model escape local minima during training.
    
    Args:
        optimizer: PyTorch optimizer
        first_cycle_steps: Length of the first cycle in steps
        cycle_mult: Multiplier for cycle length after each restart
        max_steps: Maximum number of training steps
        min_lr_ratio: Minimum learning rate ratio at the end of each cycle
        warmup_steps: Number of warmup steps
        warmup_init_lr_ratio: Initial learning rate ratio during warmup
        
    Returns:
        Scheduler with cosine annealing and warm restarts
    """
    # Get base learning rates for each parameter group
    base_lrs = [group['lr'] for group in optimizer.param_groups]
    
    # Create cosine annealing scheduler with warm restarts
    cosine_scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=first_cycle_steps,
        T_mult=cycle_mult,
        eta_min=min(min_lr_ratio * lr for lr in base_lrs)
    )
    
    # If warmup steps > 0, add a warmup phase
    if warmup_steps > 0:
        # Ensure valid warmup_init_lr_ratio (must be between 0 and 1)
        start_factor = max(0.0, min(1.0, warmup_init_lr_ratio))
        
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=start_factor,
            end_factor=1.0,
            total_iters=warmup_steps
        )
        
        # Combine the schedulers
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps]
        )
        
        return scheduler
    else:
        return cosine_scheduler


def create_optimizer_and_scheduler(
    model: torch.nn.Module,
    optimizer_config: Dict[str, Any],
    scheduler_config: Dict[str, Any],
    num_training_steps: int
) -> Tuple[torch.optim.Optimizer, Any]:
    """
    Create optimizer and learning rate scheduler for model training.
    
    This function sets up the optimizer and scheduler based on the provided
    configurations.
    
    Args:
        model: Model to optimize
        optimizer_config: Dictionary with optimizer configuration
        scheduler_config: Dictionary with scheduler configuration
        num_training_steps: Number of training steps
        
    Returns:
        Tuple of (optimizer, scheduler)
    """
    # Get optimizer parameters
    optimizer_name = optimizer_config.get('name', 'adamw').lower()
    lr = optimizer_config.get('learning_rate', 5e-4)
    weight_decay = optimizer_config.get('weight_decay', 0.01)
    beta1 = optimizer_config.get('beta1', 0.9)
    beta2 = optimizer_config.get('beta2', 0.999)
    eps = optimizer_config.get('eps', 1e-8)
    
    # Check for parameter-specific settings
    param_groups = []
    use_param_groups = optimizer_config.get('use_param_groups', False)
    
    if use_param_groups:
        # Example of parameter grouping strategy
        # 1. No weight decay for bias and layer normalization parameters
        # 2. Different learning rates for different components
        
        # Default group
        default_group = {
            'params': [],
            'lr': lr,
            'weight_decay': weight_decay,
            'betas': (beta1, beta2),
            'eps': eps
        }
        
        # No weight decay group
        no_wd_group = {
            'params': [],
            'lr': lr,
            'weight_decay': 0.0,
            'betas': (beta1, beta2),
            'eps': eps
        }
        
        # Group for selective parameterization parameters with potentially higher learning rate
        selective_param_group = {
            'params': [],
            'lr': lr * optimizer_config.get('selective_param_lr_multiplier', 1.0),
            'weight_decay': weight_decay,
            'betas': (beta1, beta2),
            'eps': eps
        }
        
        # Assign parameters to groups
        for name, param in model.named_parameters():
            if 'bias' in name or 'norm' in name or 'ln' in name or 'layer_norm' in name:
                no_wd_group['params'].append(param)
            elif 'param_generator' in name:
                selective_param_group['params'].append(param)
            else:
                default_group['params'].append(param)
        
        # Add non-empty groups to param_groups
        param_groups = [
            group for group in [default_group, no_wd_group, selective_param_group]
            if len(group['params']) > 0
        ]
    
    # Create optimizer
    if optimizer_name == 'adamw':
        if use_param_groups:
            optimizer = optim.AdamW(param_groups)
        else:
            optimizer = optim.AdamW(
                model.parameters(),
                lr=lr,
                betas=(beta1, beta2),
                eps=eps,
                weight_decay=weight_decay
            )
    elif optimizer_name == 'adam':
        if use_param_groups:
            optimizer = optim.Adam(param_groups)
        else:
            optimizer = optim.Adam(
                model.parameters(),
                lr=lr,
                betas=(beta1, beta2),
                eps=eps,
                weight_decay=weight_decay
            )
    elif optimizer_name == 'sgd':
        momentum = optimizer_config.get('momentum', 0.9)
        if use_param_groups:
            for group in param_groups:
                group['momentum'] = momentum
            optimizer = optim.SGD(param_groups)
        else:
            optimizer = optim.SGD(
                model.parameters(),
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay
            )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    
    # Get scheduler parameters
    scheduler_name = scheduler_config.get('name', 'cosine').lower()
    warmup_steps = scheduler_config.get('warmup_steps', int(0.1 * num_training_steps))
    min_lr_ratio = scheduler_config.get('min_lr_ratio', 0.1)
    
    # Create scheduler
    if scheduler_name == 'cosine':
        scheduler = get_linear_warmup_cosine_decay_scheduler(
            optimizer,
            warmup_steps=warmup_steps,
            max_steps=num_training_steps,
            min_lr_ratio=min_lr_ratio,
            warmup_init_lr_ratio=scheduler_config.get('warmup_init_lr_ratio', 0.0)
        )
    elif scheduler_name == 'cosine_restarts':
        first_cycle_steps = scheduler_config.get('first_cycle_steps', num_training_steps // 4)
        cycle_mult = scheduler_config.get('cycle_mult', 1.0)
        scheduler = get_cosine_with_restarts_scheduler(
            optimizer,
            first_cycle_steps=first_cycle_steps,
            cycle_mult=cycle_mult,
            max_steps=num_training_steps,
            min_lr_ratio=min_lr_ratio,
            warmup_steps=warmup_steps,
            warmup_init_lr_ratio=scheduler_config.get('warmup_init_lr_ratio', 0.0)
        )
    elif scheduler_name == 'linear':
        scheduler = LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=min_lr_ratio,
            total_iters=num_training_steps
        )
    elif scheduler_name == 'constant':
        # Constant learning rate (no scheduler)
        scheduler = LambdaLR(optimizer, lambda _: 1.0)
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")
    
    return optimizer, scheduler


def apply_gradient_clipping(
    optimizer: torch.optim.Optimizer,
    max_norm: float = 1.0,
    norm_type: float = 2.0
) -> None:
    """
    Apply gradient clipping to optimizer parameter groups.
    
    This function clips the gradients to prevent exploding gradients
    during training.
    
    Args:
        optimizer: PyTorch optimizer
        max_norm: Maximum allowed norm for the gradients
        norm_type: Type of norm to use (e.g., 2.0 for L2 norm)
    """
    for param_group in optimizer.param_groups:
        torch.nn.utils.clip_grad_norm_(
            param_group['params'],
            max_norm=max_norm,
            norm_type=norm_type
        )
