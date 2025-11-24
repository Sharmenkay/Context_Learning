"""
Argument dataclasses for Three-Curriculum Context-Enhanced Learning (3CEL).

This module defines configuration for the three-curriculum training architecture:
- cS: Static curriculum (expert rules/protocols)
- cC: Case curriculum (situation-specific context)
- cU: User curriculum (practitioner-level preferences/expertise)
"""

from dataclasses import dataclass, field
from typing import Optional
from transformers import TrainingArguments as TA


@dataclass
class ModelArguments:
    """
    Arguments pertaining to model configuration.
    """
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Cache directory for model files"}
    )
    use_lora: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to use LoRA for parameter-efficient fine-tuning"}
    )
    lora_r: Optional[int] = field(
        default=8,
        metadata={"help": "LoRA attention dimension"}
    )
    lora_alpha: Optional[int] = field(
        default=16,
        metadata={"help": "LoRA alpha scaling factor"}
    )
    lora_dropout: Optional[float] = field(
        default=0.1,
        metadata={"help": "LoRA dropout probability"}
    )


@dataclass
class TrainingArguments(TA):
    """
    Extended training arguments for 3CEL.
    """
    seed: Optional[int] = field(
        default=42,
        metadata={"help": "Random seed for reproducibility"}
    )

    project_name: Optional[str] = field(
        default='three_curriculum_cel',
        metadata={"help": "Name of the W&B project"}
    )

    report_to: Optional[str] = field(
        default='wandb',
        metadata={"help": "Logging backend (wandb, tensorboard, none)"}
    )

    eval_strategy: Optional[str] = field(
        default='steps',
        metadata={"help": "Evaluation strategy (no, steps, epoch)"}
    )

    eval_steps: Optional[int] = field(
        default=100,
        metadata={"help": "Evaluation frequency in steps"}
    )

    eval_on_start: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether to evaluate before training starts"}
    )

    disable_tqdm: Optional[bool] = field(
        default=False,
        metadata={"help": "Disable TQDM progress bars"}
    )

    logging_first_step: Optional[bool] = field(
        default=True,
        metadata={"help": "Log metrics at the first step"}
    )

    logging_steps: Optional[int] = field(
        default=10,
        metadata={"help": "Logging frequency in steps"}
    )

    include_inputs_for_metrics: Optional[bool] = field(
        default=True,
        metadata={"help": "Include inputs when computing metrics"}
    )

    shuffle_train: Optional[bool] = field(
        default=True,
        metadata={"help": "Shuffle training data"}
    )

    save_only_model: Optional[bool] = field(
        default=True,
        metadata={"help": "Save only model weights (not optimizer state)"}
    )

    weight_decay_pretrained: Optional[float] = field(
        default=0.0,
        metadata={"help": "Weight decay towards pretrained weights"}
    )


@dataclass
class DataArguments:
    """
    Arguments for data loading and processing.
    """
    train_path: Optional[str] = field(
        default='',
        metadata={"help": "Path to training dataset"}
    )

    eval_path: Optional[str] = field(
        default='',
        metadata={"help": "Path to evaluation dataset"}
    )

    max_sequence_length: Optional[int] = field(
        default=2048,
        metadata={"help": "Maximum sequence length for tokenization"}
    )

    train_samples: Optional[int] = field(
        default=10000,
        metadata={"help": "Number of training samples"}
    )

    eval_samples: Optional[int] = field(
        default=1000,
        metadata={"help": "Number of evaluation samples"}
    )

    preprocessing_num_workers: Optional[int] = field(
        default=4,
        metadata={"help": "Number of workers for data preprocessing"}
    )


@dataclass
class ThreeCurriculumArguments:
    """
    Arguments specific to the three-curriculum architecture.

    The three curriculum channels are:
    - cS (static): Expert rules, protocols, domain knowledge
    - cC (case): Case-specific situational context
    - cU (user): Practitioner-level preferences and expertise

    Each channel has independent dropout probability during training.
    """

    # Dropout probabilities for each curriculum channel
    dropout_static: Optional[float] = field(
        default=0.5,
        metadata={"help": "Dropout probability for static curriculum (cS). "
                  "Higher values force more internalization of static rules."}
    )

    dropout_case: Optional[float] = field(
        default=0.5,
        metadata={"help": "Dropout probability for case curriculum (cC). "
                  "Higher values force more generalization across cases."}
    )

    dropout_user: Optional[float] = field(
        default=0.5,
        metadata={"help": "Dropout probability for user curriculum (cU). "
                  "Higher values force adaptation to diverse practitioner profiles."}
    )

    # Curriculum scheduling
    dropout_schedule: Optional[str] = field(
        default='constant',
        metadata={"help": "Schedule for dropout rates during training. "
                  "Options: constant, linear_increase, warmup"}
    )

    dropout_warmup_steps: Optional[int] = field(
        default=0,
        metadata={"help": "Number of warmup steps before applying full dropout"}
    )

    # Special tokens for curriculum boundaries
    static_start_token: Optional[str] = field(
        default='<|static_start|>',
        metadata={"help": "Token marking start of static curriculum"}
    )

    static_end_token: Optional[str] = field(
        default='<|static_end|>',
        metadata={"help": "Token marking end of static curriculum"}
    )

    case_start_token: Optional[str] = field(
        default='<|case_start|>',
        metadata={"help": "Token marking start of case curriculum"}
    )

    case_end_token: Optional[str] = field(
        default='<|case_end|>',
        metadata={"help": "Token marking end of case curriculum"}
    )

    user_start_token: Optional[str] = field(
        default='<|user_start|>',
        metadata={"help": "Token marking start of user curriculum"}
    )

    user_end_token: Optional[str] = field(
        default='<|user_end|>',
        metadata={"help": "Token marking end of user curriculum"}
    )

    # Masking behavior
    mask_curriculum_in_loss: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether to mask curriculum tokens from loss computation"}
    )

    mask_input_in_loss: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether to mask task input tokens from loss computation"}
    )

    # Dropout behavior
    channel_dropout_mode: Optional[str] = field(
        default='full',
        metadata={"help": "How to apply dropout. "
                  "Options: full (drop entire channel), token (drop individual tokens)"}
    )

    dropout_replacement: Optional[str] = field(
        default='empty',
        metadata={"help": "What to replace dropped content with. "
                  "Options: empty (remove), mask (replace with mask token), pad"}
    )

    mask_token: Optional[str] = field(
        default='<|curriculum_mask|>',
        metadata={"help": "Token used when dropout_replacement='mask'"}
    )
