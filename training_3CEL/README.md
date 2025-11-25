# Three-Curriculum Context-Enhanced Learning (3CEL)

This module implements the Three-Curriculum Context-Enhanced Learning architecture for training language models with multi-channel curriculum conditioning.

## Architecture Overview

Each training sample consists of three curriculum channels and a task input:

```
(cS, cC, cU, x, y)
```

Where:
- **cS** (Static Curriculum): Expert rules, protocols, or domain knowledge
- **cC** (Case Curriculum): Situation-specific context
- **cU** (User Curriculum): Practitioner-level preferences, expertise, or workload factors
- **x**: Task input
- **y**: Target output

The curricula are concatenated into a single conditioning sequence:
```
c = concat(cS, cC, cU)
```

## Training Procedure (Algorithm 1)

```
Algorithm 1: Three-Curriculum Context-Enhanced Learning (Training)

Require: Dataset D = {(cS, cC, cU, x, y)}; model fθ;
         dropout rates pS_drop, pC_drop, pU_drop;
         optimizer; number of training steps N_steps

1: for s = 1 to N_steps do
2:   Sample mini-batch B ⊂ D
3:   L_batch ← 0
4:   for all (cS, cC, cU, x, y) ∈ B do
5:     c̃S ← Dropout(cS, pS_drop)
6:     c̃C ← Dropout(cC, pC_drop)
7:     c̃U ← Dropout(cU, pU_drop)
8:     c̃ ← concat(c̃S, c̃C, c̃U)
9:     s_in ← concat(c̃, x, y1, ..., yT-1)
10:    s_tgt ← concat(c̃, x, y1, ..., yT)
11:    Construct mask m where mt = 1 for target positions, else 0
12:    ℓ ← CrossEntropy(fθ(s_in), s_tgt, m)
13:    L_batch ← L_batch + ℓ
14:  end for
15:  Update θ using gradients of L_batch
16: end for
```

## Training Objective

The loss is computed only on target tokens:

```
L(θ) = -Σ_{t∈T} log p_θ(y_t | c̃, x, y_{<t})
```

Where T is the set of token positions corresponding to the target sequence y.

## Key Features

1. **Independent Curriculum Dropout**: Each curriculum channel has its own dropout probability, forcing the model to internalize reasoning patterns rather than memorize specific content.

2. **Selective Loss Masking**: Curriculum tokens and task input tokens are excluded from the loss computation.

3. **Flexible Dropout Scheduling**: Support for constant, linear, warmup, and cosine dropout schedules.

4. **Multiple Dropout Modes**:
   - `full`: Drop entire curriculum channel
   - `token`: Drop individual tokens within a channel

## Installation

```bash
# From the repository root
pip install -r requirements.txt
```

## Usage

### Basic Training

```bash
cd scripts
./train_3CEL.sh meta-llama/Llama-3.2-1B-Instruct ./outputs/my_run
```

### Distributed Training

```bash
cd scripts
./train_3CEL_distributed.sh 4 meta-llama/Llama-3.2-3B-Instruct ./outputs/my_distributed_run
```

### Python API

```python
from training_3CEL.components import (
    ThreeCurriculumSample,
    ThreeCurriculumDataset,
    ThreeCurriculumArguments,
    ThreeCurriculumCollator,
    ThreeCurriculumTrainer
)

# Create a sample
sample = ThreeCurriculumSample(
    c_static=tokenizer.encode("Protocol: Follow standard procedure."),
    c_case=tokenizer.encode("Patient presents with symptom X."),
    c_user=tokenizer.encode("Senior practitioner, 10 years experience."),
    x=tokenizer.encode("What is the recommended action?"),
    y=tokenizer.encode("Based on protocol, recommend action A.")
)

# Configure curriculum dropout
curriculum_args = ThreeCurriculumArguments(
    dropout_static=0.5,
    dropout_case=0.5,
    dropout_user=0.5,
    dropout_schedule='constant',
    mask_curriculum_in_loss=True,
    mask_input_in_loss=True
)

# Create dataset with curriculum dropout
dataset = ThreeCurriculumDataset(
    samples=[sample],
    tokenizer=tokenizer,
    curriculum_args=curriculum_args,
    training=True
)
```

## Data Format

Training data should be provided as a list of dictionaries with the following keys:

```python
{
    'c_static': str or List[int],  # Static curriculum (text or token IDs)
    'c_case': str or List[int],    # Case curriculum
    'c_user': str or List[int],    # User curriculum
    'x': str or List[int],         # Task input
    'y': str or List[int]          # Target output
}
```

Save as pickle (`.pkl`) or JSON (`.json`) format.

## Configuration Options

### Curriculum Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `dropout_static` | 0.5 | Dropout probability for static curriculum |
| `dropout_case` | 0.5 | Dropout probability for case curriculum |
| `dropout_user` | 0.5 | Dropout probability for user curriculum |
| `dropout_schedule` | 'constant' | Schedule: constant, linear_increase, warmup, cosine |
| `channel_dropout_mode` | 'full' | Drop full channel or individual tokens |
| `dropout_replacement` | 'empty' | Replace with: empty, mask token, or pad |
| `mask_curriculum_in_loss` | True | Exclude curriculum tokens from loss |
| `mask_input_in_loss` | True | Exclude input tokens from loss |

## File Structure

```
training_3CEL/
├── train_3CEL.py              # Main training script
├── README.md                  # This file
├── components/
│   ├── __init__.py           # Package exports
│   ├── arguments.py          # Configuration dataclasses
│   ├── data_utils.py         # Dataset and dropout classes
│   ├── data_collator.py      # Batch collation
│   ├── trainer.py            # Custom trainer
│   └── model_utils.py        # Model utilities
└── scripts/
    ├── train_3CEL.sh         # Single-GPU training
    └── train_3CEL_distributed.sh  # Multi-GPU training
```

## Citation

If you use this code, please cite:

```bibtex
@article{three_curriculum_cel,
  title={Three-Curriculum Context-Enhanced Learning},
  author={...},
  year={2024}
}
```
