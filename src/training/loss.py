"""
Loss functions for training Sparse SSM models.

This module implements various loss functions for language modeling tasks,
including standard cross-entropy loss and knowledge distillation losses.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Union, Tuple


class LanguageModelingLoss(nn.Module):
    """
    Standard language modeling loss using cross-entropy.
    
    This loss function implements the standard next-token prediction
    loss for autoregressive language models.
    """
    
    def __init__(
        self,
        ignore_index: int = -100,
        reduction: str = 'mean',
        label_smoothing: float = 0.0
    ):
        """
        Initialize the language modeling loss.
        
        Args:
            ignore_index: Target value to ignore in loss computation
            reduction: Reduction method ('none', 'mean', 'sum')
            label_smoothing: Label smoothing factor (0.0 to 1.0)
        """
        super().__init__()
        
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.label_smoothing = label_smoothing
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute the language modeling loss.
        
        Args:
            logits: Model output logits, shape (batch_size, seq_len, vocab_size)
            targets: Target token IDs, shape (batch_size, seq_len)
            attention_mask: Attention mask, shape (batch_size, seq_len)
                            1 for tokens to compute loss on, 0 for padding tokens
            
        Returns:
            Loss value
        """
        batch_size, seq_len, vocab_size = logits.size()
        
        # Reshape logits to (batch_size * seq_len, vocab_size)
        logits = logits.view(-1, vocab_size)
        
        # Reshape targets to (batch_size * seq_len)
        targets = targets.view(-1)
        
        # Compute cross-entropy loss with optional label smoothing
        loss = F.cross_entropy(
            logits,
            targets,
            ignore_index=self.ignore_index,
            reduction='none',
            label_smoothing=self.label_smoothing
        )
        
        # Apply attention mask if provided
        if attention_mask is not None:
            # Reshape attention mask to (batch_size * seq_len)
            mask = attention_mask.view(-1).float()
            loss = loss * mask
            
            # Apply reduction
            if self.reduction == 'mean':
                return loss.sum() / mask.sum() if mask.sum() > 0 else 0.0
            elif self.reduction == 'sum':
                return loss.sum()
            else:  # 'none'
                return loss.view(batch_size, seq_len)
        else:
            # Apply reduction
            if self.reduction == 'mean':
                return loss.mean()
            elif self.reduction == 'sum':
                return loss.sum()
            else:  # 'none'
                return loss.view(batch_size, seq_len)


class KnowledgeDistillationLoss(nn.Module):
    """
    Knowledge distillation loss for training with a teacher model.
    
    This loss function combines the standard cross-entropy loss with a
    distillation loss that encourages the model to match the output
    distribution of a larger teacher model.
    """
    
    def __init__(
        self,
        temperature: float = 1.0,
        alpha: float = 0.5,
        ignore_index: int = -100,
        reduction: str = 'mean'
    ):
        """
        Initialize the knowledge distillation loss.
        
        Args:
            temperature: Temperature for softmax in distillation loss
            alpha: Weight for distillation loss (0.0 to 1.0)
                   0.0 = only hard targets, 1.0 = only soft targets
            ignore_index: Target value to ignore in hard-target loss computation
            reduction: Reduction method ('none', 'mean', 'sum')
        """
        super().__init__()
        
        self.temperature = temperature
        self.alpha = alpha
        self.ignore_index = ignore_index
        self.reduction = reduction
        
        # Hard-target loss
        self.ce_loss = LanguageModelingLoss(
            ignore_index=ignore_index,
            reduction=reduction
        )
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        teacher_logits: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute the knowledge distillation loss.
        
        Args:
            logits: Model output logits, shape (batch_size, seq_len, vocab_size)
            targets: Hard target token IDs, shape (batch_size, seq_len)
            teacher_logits: Teacher model logits, shape (batch_size, seq_len, vocab_size)
            attention_mask: Attention mask, shape (batch_size, seq_len)
                            1 for tokens to compute loss on, 0 for padding tokens
            
        Returns:
            Total loss value or dictionary of loss components
        """
        batch_size, seq_len, vocab_size = logits.size()
        
        # Compute hard-target loss
        hard_loss = self.ce_loss(logits, targets, attention_mask)
        
        # Compute soft-target loss (distillation loss)
        if self.alpha > 0.0:
            # Apply temperature scaling to logits and teacher_logits
            scaled_logits = logits / self.temperature
            scaled_teacher_logits = teacher_logits / self.temperature
            
            # Compute KL divergence loss
            log_probs = F.log_softmax(scaled_logits, dim=-1)
            probs = F.softmax(scaled_teacher_logits, dim=-1)
            
            # Reshape to (batch_size * seq_len, vocab_size)
            log_probs = log_probs.view(-1, vocab_size)
            probs = probs.view(-1, vocab_size)
            
            # Compute KL divergence: KL(p||q) = p * (log(p) - log(q))
            kl_loss = F.kl_div(log_probs, probs, reduction='none').sum(-1)
            
            # Reshape back to (batch_size, seq_len)
            kl_loss = kl_loss.view(batch_size, seq_len)
            
            # Apply temperature scaling factor
            soft_loss = (self.temperature ** 2) * kl_loss
            
            # Apply attention mask if provided
            if attention_mask is not None:
                mask = attention_mask.float()
                soft_loss = soft_loss * mask
                
                # Apply reduction
                if self.reduction == 'mean':
                    soft_loss = soft_loss.sum() / mask.sum() if mask.sum() > 0 else 0.0
                elif self.reduction == 'sum':
                    soft_loss = soft_loss.sum()
                # 'none' reduction already has the correct shape
                
            else:
                # Apply reduction
                if self.reduction == 'mean':
                    soft_loss = soft_loss.mean()
                elif self.reduction == 'sum':
                    soft_loss = soft_loss.sum()
                # 'none' reduction already has the correct shape
            
            # Combine hard and soft losses
            total_loss = (1 - self.alpha) * hard_loss + self.alpha * soft_loss
            
            # Return both total loss and individual components
            return {
                'loss': total_loss,
                'hard_loss': hard_loss,
                'soft_loss': soft_loss
            }
        
        else:
            # Return only hard-target loss
            return hard_loss
