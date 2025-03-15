# Sparse State Space Model (SSM) with Selective Parameterization

This documentation provides detailed information on the implementation, usage, and theoretical foundations of our Sparse State Space Model (SSM) with Selective Parameterization for language modeling.

## Table of Contents

1. [Introduction](#introduction)
2. [Theoretical Background](#theoretical-background)
3. [Architecture Overview](#architecture-overview)
4. [Installation](#installation)
5. [Usage Guide](#usage-guide)
   - [Synthetic Data Generation](#synthetic-data-generation)
   - [Training](#training)
   - [Inference](#inference)
6. [Configuration](#configuration)
7. [GPU Optimization](#gpu-optimization)
8. [Advanced Topics](#advanced-topics)
9. [Troubleshooting](#troubleshooting)

## Introduction

State Space Models (SSMs) represent a promising alternative to attention-based mechanisms for sequence modeling. By leveraging continuous-time dynamical systems theory, SSMs offer linear time and memory complexity with respect to sequence length, making them well-suited for processing long sequences efficiently.

This implementation focuses on:

1. **Sparse SSMs**: Using structured sparsity patterns in parameter matrices to reduce computational and memory requirements.
2. **Selective Parameterization**: Dynamically adjusting the model's parameters (A, B, C, D matrices) based on the input content.
3. **HiPPO Initialization**: Leveraging the Higher-order Polynomial Projection Operator framework for principled initialization of state matrices.
4. **GPU Optimizations**: Techniques like parallel scan algorithms, kernel fusion, and memory-efficient backpropagation.

## Theoretical Background

### State Space Equations

The core of our model is based on the continuous-time linear state space equations:

$$\dot{x}(t) = Ax(t) + Bu(t)$$
$$y(t) = Cx(t) + Du(t)$$

Where:
- $x(t)$ is the hidden state vector
- $u(t)$ is the input vector
- $y(t)$ is the output vector
- $A$, $B$, $C$, and $D$ are parameter matrices

For computational purposes, we discretize these equations to obtain:

$$x_t = \overline{A}x_{t-1} + \overline{B}u_t$$
$$y_t = Cx_t + Du_t$$

Where $\overline{A}$ and $\overline{B}$ are discretized versions of the continuous-time parameters.

### Selective Parameterization

In selective parameterization, the model parameters A, B, C, and D become functions of the input:

$$A = f_A(u_t), B = f_B(u_t), C = f_C(u_t), D = f_D(u_t)$$

This allows the model to adapt its dynamics based on the input content, significantly enhancing its expressivity while maintaining the computational advantages of SSMs.

### HiPPO Initialization

The HiPPO framework provides a principled way to initialize the state matrix A to capture information at different timescales. We implement several variants:

1. **HiPPO-LegS**: Based on Legendre polynomials, optimized for memory retention.
2. **HiPPO-LegT**: Translated Legendre polynomials for better stability.
3. **HiPPO-Fourier**: Based on Fourier modes for capturing periodic patterns.

### Sparsity

We implement structured sparsity patterns for the parameter matrices, particularly focusing on the state transition matrix A. These patterns are designed to:

1. Reduce memory and computational requirements
2. Maintain model expressivity
3. Be compatible with GPU execution patterns for efficient computation

## Architecture Overview

The overall architecture consists of:

### Model Components

1. **Discretization Layer**: Implements different methods to convert continuous-time parameters to discrete-time equivalents.
2. **HiPPO Initialization**: Provides principled initialization for capturing long-range dependencies.
3. **Selective Parameterization Modules**: Generate model parameters dynamically based on input.
4. **SSM Layer**: Core layer that implements the state space equation operations.
5. **Mamba-inspired Architecture**: Combines SSM operations with other elements like feed-forward networks in a block structure.

### Infrastructure Components

1. **Dataset Handling**: Modules for loading, preprocessing, and creating synthetic datasets.
2. **Training Pipeline**: Includes optimizers, schedulers, and the training loop.
3. **Inference Procedures**: Efficient text generation with various decoding strategies.
4. **Configuration System**: YAML-based configuration for model parameters and training settings.

## Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)

### Installation Steps

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd SSM
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage Guide

### Synthetic Data Generation

To generate a synthetic dataset using Ollama's phi4-mini model:

```bash
python main.py create-synthetic --config config/default_config.yaml
```

This will generate a synthetic dataset based on the configuration in the YAML file. You can customize parameters like number of samples, temperature, and output location:

```bash
python main.py create-synthetic \
    --config config/default_config.yaml \
    --num-samples 1000 \
    --temperature 0.7 \
    --output-dir data/synthetic \
    --output-file my_dataset.json
```

For knowledge distillation datasets that include teacher model logits:

```bash
python main.py create-synthetic \
    --config config/default_config.yaml \
    --distillation
```

### Training

To train the model with the default configuration:

```bash
python main.py train --config config/default_config.yaml
```

You can specify additional parameters like output directory and random seed:

```bash
python main.py train \
    --config config/default_config.yaml \
    --output-dir outputs/my_model \
    --seed 42
```

For testing, you can use the best model from training:

```bash
python main.py train \
    --config config/default_config.yaml \
    --use-best-model
```

### Inference

To generate text using a trained model:

```bash
python main.py generate \
    --config config/default_config.yaml \
    --model-dir outputs/sparse_ssm \
    --prompt "Once upon a time"
```

You can customize the generation parameters:

```bash
python main.py generate \
    --config config/default_config.yaml \
    --model-dir outputs/sparse_ssm \
    --checkpoint best \
    --prompt "The Sparse State Space Model" \
    --max-length 200 \
    --temperature 0.8 \
    --top-k 50 \
    --top-p 0.9 \
    --output-file generated_text.txt
```

## Configuration

The model and training process are configured using YAML files. Here's an overview of the main configuration sections:

### Model Configuration

```yaml
model:
  vocab_size: 50000
  hidden_dim: 768
  state_dim: 128
  ffn_dim: 2048
  num_layers: 12
  max_seq_len: 1024
  dropout: 0.1
  activation: "gelu"
  selective_param_class: "sparse"  # "dense", "sparse", or "low_rank"
  sparsity_level: 0.9
  use_parallel_scan: true
```

### Training Configuration

```yaml
training:
  num_epochs: 3
  gradient_accumulation_steps: 2
  max_grad_norm: 1.0
  logging_steps: 100
  save_steps: 1000
  eval_steps: 1000
  mixed_precision: "fp16"  # "no", "fp16", or "bf16"
  gradient_checkpointing: true
```

### Optimizer and Scheduler

```yaml
optimizer:
  name: "adamw"
  learning_rate: 5.0e-4
  weight_decay: 0.01
  use_param_groups: true
  selective_param_lr_multiplier: 1.5

scheduler:
  name: "cosine"
  warmup_steps: 1000
  min_lr_ratio: 0.1
```

### Dataset Configuration

```yaml
dataset:
  type: "synthetic"  # "wikitext103", "pile", "synthetic", or "custom"
  max_seq_len: 1024
  stride: 512
  data_path: "data/synthetic/phi4_dataset.json"
  val_split: 0.1
  test_split: 0.1
```

## GPU Optimization

This implementation includes several optimizations for efficient GPU execution, particularly for memory-constrained environments:

### Memory Efficiency

1. **Gradient Checkpointing**: Trades computation for memory by recomputing certain activations during backpropagation.
2. **Mixed Precision Training**: Uses lower precision (FP16 or BF16) to reduce memory requirements and speed up computation.
3. **Efficient Parameter Sharing**: Tied embedding weights to reduce parameter count.

### Computational Efficiency

1. **Parallel Scan Algorithms**: Efficient implementation of recurrent computations with better GPU utilization.
2. **Structured Sparsity**: Designed for compatibility with GPU execution patterns.
3. **Selective Computation**: Dynamic parameter generation focused on computation where it matters most.

### Optimization Techniques

1. **Parameter Groups**: Different learning rates for different components (especially beneficial for selective parameterization).
2. **Gradient Accumulation**: Allows for effectively larger batch sizes on limited hardware.
3. **Learning Rate Scheduling**: Cosine decay with warmup for better convergence.

## Advanced Topics

### Sparse Parameter Implementation

The implementation supports three types of selective parameterization:

1. **Dense**: Full parameter matrices generated based on input.
2. **Sparse**: Structured sparse parameter matrices with different patterns based on sparsity level.
3. **Low-Rank**: Parameter matrices represented as low-rank approximations (A = U*V + diagonal).

### Discretization Methods

We implement multiple discretization techniques:

1. **Zero-Order Hold (ZOH)**: Assumes constant input over the sampling interval.
2. **Bilinear Transform (Tustin's method)**: Maps the s-plane to the z-plane, preserving stability.
3. **Generalized Bilinear**: Parameterized method that can represent multiple discretization approaches.

### Knowledge Distillation

For training with a teacher model:

1. Use the `create-synthetic` command with `--distillation` to generate a dataset with teacher logits.
2. Configure the training to use knowledge distillation with temperature scaling and mixing ratio.

## Troubleshooting

### Common Issues

1. **Out of Memory Errors**:
   - Reduce batch size
   - Enable gradient checkpointing
   - Use mixed precision training
   - Reduce model size (hidden_dim, state_dim, num_layers)

2. **Slow Training**:
   - Ensure parallel scan is enabled
   - Check that structured sparsity is being used
   - Verify GPU utilization is high

3. **Poor Generation Quality**:
   - Adjust temperature, top-k, and top-p parameters
   - Train for more epochs
   - Ensure dataset quality is high
   - Try different HiPPO initialization methods

### Debugging Tools

- Enable TensorBoard logging by setting `training.logging_steps` to a lower value
- Use the `--use-best-model` flag to evaluate with the best checkpoint
- Check model gradients and parameter norms during training
