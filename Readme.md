# GRAPE: Group Robust Multi-target Adaptive Data Mixture for Pretraining

**[NeurIPS 2025]** Official implementation of **GRAPE**: Optimize Data Mixture for Group Robust Multi-target Adaptive Pretraining

## Overview

GRAPE is a research framework for optimizing data mixtures in multi-target adaptive pretraining scenarios. The framework addresses the challenge of selecting optimal training data distributions when adapting language models to multiple downstream tasks simultaneously, with a focus on group robustness.

### Key Features

- **Multi-target Adaptive Pretraining**: Simultaneously optimize for multiple downstream tasks
- **Group Robust Optimization**: Ensure robust performance across different task groups
- **Dynamic Data Reweighting**: Adaptive weighting strategies (GRAPE, DoGE, CRISP, RegMix)
- **Flexible Scheduler Support**: Custom learning rate schedulers with warmup and decay
- **Distributed Training**: Multi-GPU support via PyTorch DDP and Accelerate
- **Benchmark Integration**: Built-in support for multiple reasoning benchmarks

## Installation

### Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA (for GPU training)

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/GRAPE_data_mixture_with_multi_target.git
cd GRAPE_data_mixture_with_multi_target

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Basic Training

Run multi-target adaptive pretraining with GRAPE reweighting:

```bash
python -m src.run \
  --config_json config/climb/grape_mu5.json \
  --wandb_proj your-project-name \
  --wandb_run your-run-name
```

### Distributed Training

For multi-GPU training with DDP:

```bash
torchrun --nproc_per_node=4 -m src.run \
  --config_json config/climb/grape_mu5.json \
  --wandb_proj your-project-name
```

## Project Structure

```
GRAPE_data_mixture_with_multi_target/
├── src/
│   ├── run.py                  # Main training entry point
│   ├── trainer.py              # Trainer implementation with adaptive data reweighting algorithms
│   ├── schedulers.py           # Custom LR schedulers
│   ├── ema.py                  # Exponential moving average utilities
│   ├── eval_acc.py             # Evaluation utilities
│   ├── data/
│   │   ├── benchmarks.py       # Benchmark dataset loaders
│   │   ├── dataloader.py       # Base dataloader
│   │   ├── weighted_dataloader.py  # Weighted sampling
│   │   ├── climb.py            # CLIMB dataset integration
│   │   ├── wiki40b.py          # Wiki40B dataset
│   │   ├── slim_redpajama.py   # SlimPajama dataset
│   │   └── utils.py            # Data utilities
│   └── models/
│       ├── gpt2.py             # GPT-2 model implementations
│       └── utils.py            # Model utilities
├── config/
│   ├── climb/                  # CLIMB benchmark configs
│   │   ├── grape_mu5.json      # GRAPE with μ=5
│   │   ├── grape_mu10.json     # GRAPE with μ=10
│   │   ├── doge.json           # DoGE baseline
│   │   ├── crisp.json          # CRISP baseline
│   │   └── regmix.json         # RegMix baseline
│   └── wiki40b/                # Wiki40B configs
│       └── ...
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Configuration

### Configuration Files

Training configurations are defined in JSON files under `config/`. Key parameters include:

```json
{
  "dataset": "climb-logiqa-piqa-arc_easy-arc_challenge-hellaswag-sciq",
  "train_domains": "cluster_1,cluster_2,...",
  "tgt_domains": "logiqa,piqa,arc_easy,...",
  "reweight_train": "doge",
  "reweight_tgt": "grape",
  "mu_train": 1.5,
  "mu_tgt": 5.0,
  "learning_rate": 1.5e-4,
  "max_steps": 50000,
  "per_device_train_batch_size": 16,
  "gradient_accumulation_steps": 2
}
```

### Reweighting Methods

- **GRAPE**: Group Robust Adaptive Pretraining (proposed method)
- **DoGE**: Domain-oriented Gradient-based Exploration
- **CRISP**: Contrastive Representation for Improved Sample Prioritization
- **RegMix**: Regularized Mixture Learning

### Learning Rate Schedulers

Available schedulers in `src/schedulers.py`:

- `linear_warmup_exponential`: Exponential decay with linear warmup
- `linear_warmup_cosine`: Cosine annealing with linear warmup
- `linear_warmup_decay`: Constant lr with linear decay and warmup

## Supported Datasets

### Training Data Sources

- **[CLIMB](https://arxiv.org/abs/2504.13161)**: Clustered language model benchmark data
- **Wiki40B**: Multilingual Wikipedia corpus
- **SlimPajama**: Deduplicated subset of RedPajama

### Available Benchmarks

- **Reasoning**: LogiQA, ARC-Easy, ARC-Challenge, PIQA, HellaSwag, SciQ
- **Coding**: HumanEval, KodCode
- **Math**: GSM8K, MathQA
- **Medical**: MedQA

## Experiments

### Example Configurations

**CLIMB Benchmark with GRAPE (μ=5)**

```bash
python -m src.run \
  --config_json config/climb/grape_mu5.json \
  --wandb_proj climb-experiments
```

**Wiki40B Multi-task T1**

```bash
python -m src.run \
  --config_json config/wiki40b/T1-wiki40b.json \
  --wandb_proj wiki40b-experiments
```

**Wiki40B all jobs**

```bash
bash scripts/wiki40b/base.sh
bash scripts/wiki40b/doge.sh
bash scripts/wiki40b/grape.sh
```

**CLIMB all jobs**

```bash
bash scripts/climb/climb.sh
```

### Checkpointing

- Checkpoints are saved to `<output_dir>/<run_name>/`
- Automatic resume from last checkpoint
- Use `--overwrite_output_dir` to train from scratch

## Monitoring

### Weights & Biases Integration

The framework integrates with W&B for experiment tracking:

```bash
# Set W&B project name
export WANDB_PROJECT=your-project-name

# Run training
python -m src.run --config_json config/your_config.json
```

Logged metrics include:

- Training loss
- Learning rate
- Domain weights (for reweighting methods)
- Evaluation metrics per task

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{grape2025,
  title={GRAPE: Optimize Data Mixture for Group Robust Multi-target Adaptive Pretraining},
  author={Your Name et al.},
  booktitle={Advances in Neural Information Processing Systems},
  year={2025}
}
```
