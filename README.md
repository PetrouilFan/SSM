# Sparse State Space Model (SSM) with Selective Parameterization

This project implements a language model based on Sparse State Space Models (SSMs) with Selective Parameterization. The model is designed to efficiently process long sequences with linear complexity while being suitable for GPU deployment, even on resource-constrained hardware like GTX 1080 Ti.

## Project Overview

State Space Models (SSMs) are a class of sequence models that represent sequences using linear state space equations. This implementation focuses on:

1. **Sparse SSMs**: Utilizing sparsity in parameter matrices to reduce computational cost and memory footprint
2. **Selective Parameterization**: Dynamically adjusting the model's parameters based on the input
3. **HiPPO Initialization**: Leveraging the Higher-order Polynomial Projection Operator for initializing state matrices
4. **GPU Optimization**: Employing techniques for efficient execution on resource-constrained GPUs

## Project Structure

```
.
├── config/               # Configuration files for model and training
├── data/                 # Data storage and data generation scripts
├── docs/                 # Documentation files
├── src/                  # Source code
│   ├── model/            # Model implementation
│   │   ├── ssm.py        # Core SSM implementation
│   │   ├── hippo.py      # HiPPO initialization
│   │   ├── mamba.py      # Mamba-inspired architecture
│   │   ├── discretization.py # Discretization methods
│   │   └── selective_param.py # Selective parameterization
│   ├── training/         # Training routines
│   │   ├── trainer.py    # Training loop
│   │   ├── optimizer.py  # Optimizer configurations
│   │   └── loss.py       # Loss functions
│   ├── inference/        # Inference procedures
│   │   ├── inference.py  # Inference routines
│   │   └── gpu_optimizations.py # GPU-specific optimizations
│   ├── dataset/          # Dataset handling
│   │   ├── dataloader.py # Data loading utilities
│   │   ├── tokenizer.py  # Tokenization utilities
│   │   └── synthetic_data.py # Synthetic dataset generation
│   └── utils/            # Utility functions
│       ├── logging.py    # Logging utilities
│       ├── checkpointing.py # Checkpointing utilities
│       └── metrics.py    # Evaluation metrics
└── tests/                # Unit and integration tests
```

## Mathematical Foundations

The model is based on the continuous-time linear state space equations:

$$\dot{x}(t) = Ax(t) + Bu(t)$$
$$y(t) = Cx(t) + Du(t)$$

These equations are discretized to obtain:

$$x_t = \overline{A}x_{t-1} + \overline{B}u_t$$
$$y_t = Cx_t + Du_t$$

Where selective parameterization allows $A$, $B$, $C$, and $D$ to be functions of the input $u_t$.

## Installation

To install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Training

To train the model:

```bash
python -m src.main train --config config/train_config.yaml
```

### Inference

To run inference:

```bash
python -m src.main generate --config config/inference_config.yaml --prompt "Your prompt here"
```

### Synthetic Dataset Generation

To generate a synthetic dataset using phi4-mini model via Ollama:

```bash
python -m src.dataset.synthetic_data --num_samples 1000 --output_path data/synthetic/phi4_dataset.json
```

## License

[MIT License](LICENSE)
