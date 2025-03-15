"""
Trainer for Sparse SSM models.

This module implements the training loop, evaluation, and model checkpointing
for Sparse SSM models with selective parameterization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Optional, List, Union, Callable, Any, Tuple
import os
import time
import math
import json
import logging
from tqdm import tqdm

from .loss import LanguageModelingLoss, KnowledgeDistillationLoss
from .optimizer import create_optimizer_and_scheduler, apply_gradient_clipping


class SSMTrainer:
    """
    Trainer for Sparse SSM models.
    
    This class implements the training loop, evaluation, and checkpointing
    for Sparse SSM models with selective parameterization.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
        optimizer_config: Dict[str, Any] = None,
        scheduler_config: Dict[str, Any] = None,
        training_config: Dict[str, Any] = None,
        distillation_config: Dict[str, Any] = None,
        teacher_model: Optional[nn.Module] = None,
        device: Optional[torch.device] = None,
        output_dir: str = './outputs'
    ):
        """
        Initialize the trainer.
        
        Args:
            model: Model to train
            train_dataloader: DataLoader for training data
            eval_dataloader: DataLoader for evaluation data
            optimizer_config: Dictionary with optimizer configuration
            scheduler_config: Dictionary with scheduler configuration
            training_config: Dictionary with training configuration
            distillation_config: Dictionary with knowledge distillation configuration
            teacher_model: Teacher model for knowledge distillation
            device: Device to use for training
            output_dir: Directory to save checkpoints and logs
        """
        self.model = model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.teacher_model = teacher_model
        self.output_dir = output_dir
        
        # Setup device
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        if self.teacher_model is not None:
            self.teacher_model = self.teacher_model.to(self.device)
            self.teacher_model.eval()  # Set teacher model to evaluation mode
        
        # Default configurations
        if optimizer_config is None:
            optimizer_config = {'name': 'adamw', 'learning_rate': 5e-4, 'weight_decay': 0.01}
        
        if scheduler_config is None:
            scheduler_config = {'name': 'cosine', 'warmup_steps': 1000, 'min_lr_ratio': 0.1}
        
        if training_config is None:
            training_config = {
                'num_epochs': 3,
                'gradient_accumulation_steps': 1,
                'max_grad_norm': 1.0,
                'logging_steps': 100,
                'save_steps': 1000,
                'eval_steps': 1000,
                'mixed_precision': 'no',  # 'no', 'fp16', or 'bf16'
                'gradient_checkpointing': False
            }
        
        self.training_config = training_config
        
        # Calculate number of training steps
        num_epochs = training_config.get('num_epochs', 3)
        gradient_accumulation_steps = training_config.get('gradient_accumulation_steps', 1)
        steps_per_epoch = len(train_dataloader) // gradient_accumulation_steps
        num_training_steps = num_epochs * steps_per_epoch
        
        # Create optimizer and scheduler
        self.optimizer, self.scheduler = create_optimizer_and_scheduler(
            model=model,
            optimizer_config=optimizer_config,
            scheduler_config=scheduler_config,
            num_training_steps=num_training_steps
        )
        
        # Create loss function
        if teacher_model is not None and distillation_config is not None:
            self.loss_fn = KnowledgeDistillationLoss(
                temperature=distillation_config.get('temperature', 1.0),
                alpha=distillation_config.get('alpha', 0.5),
                ignore_index=distillation_config.get('ignore_index', -100),
                reduction=distillation_config.get('reduction', 'mean')
            )
        else:
            self.loss_fn = LanguageModelingLoss(
                ignore_index=training_config.get('ignore_index', -100),
                reduction='mean',
                label_smoothing=training_config.get('label_smoothing', 0.0)
            )
        
        # Setup gradient checkpointing if enabled
        if training_config.get('gradient_checkpointing', False):
            model.gradient_checkpointing_enable()
        
        # Setup mixed precision training
        self.mixed_precision = training_config.get('mixed_precision', 'no')
        self.scaler = torch.cuda.amp.GradScaler() if self.mixed_precision != 'no' else None
        
        # Setup logging and checkpointing
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'checkpoints'), exist_ok=True)
        self.writer = SummaryWriter(log_dir=os.path.join(output_dir, 'logs'))
        
        # Setup logging
        logging.basicConfig(
            format='%(asctime)s - %(levelname)s - %(message)s',
            level=logging.INFO,
            handlers=[
                logging.FileHandler(os.path.join(output_dir, 'train.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_eval_loss = float('inf')
    
    def train(self):
        """
        Train the model for the specified number of epochs.
        
        This method implements the main training loop.
        """
        num_epochs = self.training_config.get('num_epochs', 3)
        logging_steps = self.training_config.get('logging_steps', 100)
        save_steps = self.training_config.get('save_steps', 1000)
        eval_steps = self.training_config.get('eval_steps', 1000)
        gradient_accumulation_steps = self.training_config.get('gradient_accumulation_steps', 1)
        max_grad_norm = self.training_config.get('max_grad_norm', 1.0)
        
        # Log training configuration
        self.logger.info(f"Starting training on device: {self.device}")
        self.logger.info(f"Number of epochs: {num_epochs}")
        self.logger.info(f"Number of training steps: {num_epochs * len(self.train_dataloader)}")
        self.logger.info(f"Number of optimization steps: {num_epochs * len(self.train_dataloader) // gradient_accumulation_steps}")
        self.logger.info(f"Gradient accumulation steps: {gradient_accumulation_steps}")
        self.logger.info(f"Mixed precision mode: {self.mixed_precision}")
        
        # Training loop
        self.model.train()
        start_time = time.time()
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            epoch_start_time = time.time()
            
            for step, batch in enumerate(tqdm(self.train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")):
                # Process batch
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Get teacher predictions if using knowledge distillation
                if self.teacher_model is not None:
                    with torch.no_grad():
                        teacher_outputs = self.teacher_model(**batch)
                        teacher_logits = teacher_outputs['logits'] if isinstance(teacher_outputs, dict) else teacher_outputs
                else:
                    teacher_logits = None
                
                # Forward pass with optional mixed precision
                if self.mixed_precision != 'no':
                    with torch.cuda.amp.autocast(dtype=torch.float16 if self.mixed_precision == 'fp16' else torch.bfloat16):
                        outputs = self.model(**batch)
                        logits = outputs['logits'] if isinstance(outputs, dict) else outputs
                        
                        # Compute loss
                        if teacher_logits is not None:
                            loss_dict = self.loss_fn(
                                logits=logits,
                                targets=batch['labels'],
                                teacher_logits=teacher_logits,
                                attention_mask=batch.get('attention_mask')
                            )
                            loss = loss_dict['loss'] if isinstance(loss_dict, dict) else loss_dict
                        else:
                            loss = self.loss_fn(
                                logits=logits,
                                targets=batch['labels'],
                                attention_mask=batch.get('attention_mask')
                            )
                        
                        # Scale loss for gradient accumulation
                        loss = loss / gradient_accumulation_steps
                    
                    # Backward pass with gradient scaling
                    self.scaler.scale(loss).backward()
                else:
                    # Forward pass without mixed precision
                    outputs = self.model(**batch)
                    logits = outputs['logits'] if isinstance(outputs, dict) else outputs
                    
                    # Compute loss
                    if teacher_logits is not None:
                        loss_dict = self.loss_fn(
                            logits=logits,
                            targets=batch['labels'],
                            teacher_logits=teacher_logits,
                            attention_mask=batch.get('attention_mask')
                        )
                        loss = loss_dict['loss'] if isinstance(loss_dict, dict) else loss_dict
                    else:
                        loss = self.loss_fn(
                            logits=logits,
                            targets=batch['labels'],
                            attention_mask=batch.get('attention_mask')
                        )
                    
                    # Scale loss for gradient accumulation
                    loss = loss / gradient_accumulation_steps
                    
                    # Backward pass
                    loss.backward()
                
                # Optimize every gradient_accumulation_steps or at the end of the epoch
                if (step + 1) % gradient_accumulation_steps == 0 or step == len(self.train_dataloader) - 1:
                    # Update global step
                    self.global_step += 1
                    
                    # Clip gradients
                    if max_grad_norm > 0:
                        if self.mixed_precision != 'no':
                            self.scaler.unscale_(self.optimizer)
                        
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                    
                    # Update parameters
                    if self.mixed_precision != 'no':
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    
                    # Update learning rate
                    self.scheduler.step()
                    
                    # Reset gradients
                    self.optimizer.zero_grad()
                    
                    # Log training progress
                    if self.global_step % logging_steps == 0:
                        lr = self.scheduler.get_last_lr()[0]
                        self.writer.add_scalar('lr', lr, self.global_step)
                        self.writer.add_scalar('loss', loss.item() * gradient_accumulation_steps, self.global_step)
                        
                        if isinstance(loss_dict, dict) and 'hard_loss' in loss_dict and 'soft_loss' in loss_dict:
                            self.writer.add_scalar('hard_loss', loss_dict['hard_loss'].item(), self.global_step)
                            self.writer.add_scalar('soft_loss', loss_dict['soft_loss'].item(), self.global_step)
                        
                        self.logger.info(
                            f"Step: {self.global_step}, Loss: {loss.item() * gradient_accumulation_steps:.4f}, "
                            f"LR: {lr:.8f}, Time: {(time.time() - start_time) / 60:.2f}m"
                        )
                    
                    # Save checkpoint
                    if self.global_step % save_steps == 0:
                        self.save_checkpoint(os.path.join(self.output_dir, 'checkpoints', f'step_{self.global_step}'))
                    
                    # Evaluate
                    if self.eval_dataloader is not None and self.global_step % eval_steps == 0:
                        eval_results = self.evaluate()
                        self.model.train()  # Set model back to training mode
                        
                        # Log evaluation results
                        self.writer.add_scalar('eval_loss', eval_results['loss'], self.global_step)
                        self.writer.add_scalar('eval_perplexity', eval_results['perplexity'], self.global_step)
                        
                        # Save best model
                        if eval_results['loss'] < self.best_eval_loss:
                            self.best_eval_loss = eval_results['loss']
                            self.save_checkpoint(os.path.join(self.output_dir, 'checkpoints', 'best'))
                            self.logger.info(f"New best model saved with eval loss: {self.best_eval_loss:.4f}")
            
            # End of epoch
            epoch_time = time.time() - epoch_start_time
            self.logger.info(f"Epoch {epoch + 1}/{num_epochs} completed in {epoch_time / 60:.2f}m")
            
            # Save checkpoint at the end of each epoch
            self.save_checkpoint(os.path.join(self.output_dir, 'checkpoints', f'epoch_{epoch + 1}'))
        
        # End of training
        total_time = time.time() - start_time
        self.logger.info(f"Training completed in {total_time / 60:.2f}m")
        
        # Final evaluation
        if self.eval_dataloader is not None:
            eval_results = self.evaluate()
            self.logger.info(f"Final evaluation results: {eval_results}")
        
        # Save final model
        self.save_checkpoint(os.path.join(self.output_dir, 'checkpoints', 'final'))
        
        return self.model
    
    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate the model on the evaluation dataset.
        
        Returns:
            Dictionary with evaluation metrics
        """
        if self.eval_dataloader is None:
            raise ValueError("Evaluation dataloader is required for evaluation")
        
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        
        with torch.no_grad():
            for batch in tqdm(self.eval_dataloader, desc="Evaluating"):
                # Process batch
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(**batch)
                logits = outputs['logits'] if isinstance(outputs, dict) else outputs
                
                # Compute loss
                loss = self.loss_fn(
                    logits=logits,
                    targets=batch['labels'],
                    attention_mask=batch.get('attention_mask')
                )
                
                # Accumulate loss
                if batch.get('attention_mask') is not None:
                    total_tokens += batch['attention_mask'].sum().item()
                    total_loss += loss.item() * batch['attention_mask'].sum().item()
                else:
                    # Assume all tokens are valid if no attention mask is provided
                    total_tokens += batch['labels'].numel()
                    total_loss += loss.item() * batch['labels'].numel()
        
        # Calculate metrics
        avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
        perplexity = math.exp(avg_loss)
        
        # Log results
        self.logger.info(f"Evaluation results at step {self.global_step}:")
        self.logger.info(f"  Loss: {avg_loss:.4f}")
        self.logger.info(f"  Perplexity: {perplexity:.4f}")
        
        # Return metrics
        return {
            'loss': avg_loss,
            'perplexity': perplexity,
            'step': self.global_step
        }
    
    def save_checkpoint(self, path: str) -> None:
        """
        Save a model checkpoint.
        
        Args:
            path: Path to save the checkpoint
        """
        os.makedirs(path, exist_ok=True)
        
        # Save model
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        model_to_save.save_pretrained(path) if hasattr(model_to_save, 'save_pretrained') else torch.save(model_to_save.state_dict(), os.path.join(path, 'pytorch_model.bin'))
        
        # Save optimizer and scheduler states
        torch.save(self.optimizer.state_dict(), os.path.join(path, 'optimizer.pt'))
        torch.save(self.scheduler.state_dict(), os.path.join(path, 'scheduler.pt'))
        
        # Save training state
        training_state = {
            'global_step': self.global_step,
            'epoch': self.epoch,
            'best_eval_loss': self.best_eval_loss
        }
        with open(os.path.join(path, 'training_state.json'), 'w') as f:
            json.dump(training_state, f)
        
        self.logger.info(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str) -> None:
        """
        Load a model checkpoint.
        
        Args:
            path: Path to load the checkpoint from
        """
        # Load model
        if hasattr(self.model, 'from_pretrained'):
            self.model = type(self.model).from_pretrained(path)
        else:
            self.model.load_state_dict(torch.load(os.path.join(path, 'pytorch_model.bin')))
        
        self.model = self.model.to(self.device)
        
        # Load optimizer and scheduler states
        self.optimizer.load_state_dict(torch.load(os.path.join(path, 'optimizer.pt')))
        self.scheduler.load_state_dict(torch.load(os.path.join(path, 'scheduler.pt')))
        
        # Load training state
        with open(os.path.join(path, 'training_state.json'), 'r') as f:
            training_state = json.load(f)
        
        self.global_step = training_state['global_step']
        self.epoch = training_state['epoch']
        self.best_eval_loss = training_state['best_eval_loss']
        
        self.logger.info(f"Checkpoint loaded from {path}")
        self.logger.info(f"Resuming from global step {self.global_step}, epoch {self.epoch}")
