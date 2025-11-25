"""
Data utilities for Three-Curriculum Context-Enhanced Learning (3CEL).

This module provides:
- ThreeCurriculumSample: Data structure for training samples
- ThreeCurriculumDataset: PyTorch Dataset with curriculum dropout
- Utility functions for data loading and processing
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple, Union
import pickle
import os
import copy


@dataclass
class ThreeCurriculumSample:
    """
    Represents a single training sample with three curriculum channels.

    Attributes:
        c_static: Static curriculum tokens (expert rules/protocols)
        c_case: Case curriculum tokens (situation-specific context)
        c_user: User curriculum tokens (practitioner preferences)
        x: Task input tokens
        y: Target output tokens
    """
    c_static: List[int]
    c_case: List[int]
    c_user: List[int]
    x: List[int]
    y: List[int]

    def to_dict(self) -> Dict[str, List[int]]:
        return {
            'c_static': self.c_static,
            'c_case': self.c_case,
            'c_user': self.c_user,
            'x': self.x,
            'y': self.y
        }


class CurriculumDropout:
    """
    Implements independent dropout for each curriculum channel.

    Following Algorithm 1 from the paper, each curriculum channel
    undergoes independent dropout with its own probability.
    """

    def __init__(
        self,
        p_static: float = 0.5,
        p_case: float = 0.5,
        p_user: float = 0.5,
        mode: str = 'full',
        replacement: str = 'empty',
        mask_token_id: Optional[int] = None
    ):
        """
        Args:
            p_static: Dropout probability for static curriculum
            p_case: Dropout probability for case curriculum
            p_user: Dropout probability for user curriculum
            mode: 'full' (drop entire channel) or 'token' (drop individual tokens)
            replacement: 'empty' (remove), 'mask' (replace with mask token), 'pad'
            mask_token_id: Token ID for mask replacement
        """
        self.p_static = p_static
        self.p_case = p_case
        self.p_user = p_user
        self.mode = mode
        self.replacement = replacement
        self.mask_token_id = mask_token_id

    def _apply_dropout(
        self,
        tokens: List[int],
        dropout_prob: float
    ) -> Tuple[List[int], bool]:
        """
        Apply dropout to a curriculum channel.

        Args:
            tokens: Token IDs for the curriculum
            dropout_prob: Probability of dropping

        Returns:
            Tuple of (processed tokens, was_dropped flag)
        """
        if len(tokens) == 0:
            return tokens, False

        if self.mode == 'full':
            # Drop entire channel with probability p
            if np.random.random() < dropout_prob:
                if self.replacement == 'empty':
                    return [], True
                elif self.replacement == 'mask' and self.mask_token_id is not None:
                    return [self.mask_token_id], True
                else:
                    return [], True
            return tokens, False

        elif self.mode == 'token':
            # Drop individual tokens with probability p
            result = []
            any_dropped = False
            for token in tokens:
                if np.random.random() < dropout_prob:
                    any_dropped = True
                    if self.replacement == 'mask' and self.mask_token_id is not None:
                        result.append(self.mask_token_id)
                    # For 'empty', we simply don't append
                else:
                    result.append(token)
            return result, any_dropped

        return tokens, False

    def __call__(
        self,
        sample: ThreeCurriculumSample
    ) -> Tuple[ThreeCurriculumSample, Dict[str, bool]]:
        """
        Apply dropout to all three curriculum channels.

        Args:
            sample: The training sample

        Returns:
            Tuple of (processed sample, dropout info dict)
        """
        c_static_dropped, static_was_dropped = self._apply_dropout(
            sample.c_static, self.p_static
        )
        c_case_dropped, case_was_dropped = self._apply_dropout(
            sample.c_case, self.p_case
        )
        c_user_dropped, user_was_dropped = self._apply_dropout(
            sample.c_user, self.p_user
        )

        dropped_sample = ThreeCurriculumSample(
            c_static=c_static_dropped,
            c_case=c_case_dropped,
            c_user=c_user_dropped,
            x=sample.x,  # Task input is never dropped
            y=sample.y   # Target is never dropped
        )

        dropout_info = {
            'static_dropped': static_was_dropped,
            'case_dropped': case_was_dropped,
            'user_dropped': user_was_dropped
        }

        return dropped_sample, dropout_info


def get_dropout_schedule(
    step: int,
    total_steps: int,
    base_dropout: float,
    schedule: str = 'constant',
    warmup_steps: int = 0
) -> float:
    """
    Compute dropout rate based on training schedule.

    Args:
        step: Current training step
        total_steps: Total number of training steps
        base_dropout: Base dropout probability
        schedule: Schedule type (constant, linear_increase, warmup)
        warmup_steps: Number of warmup steps

    Returns:
        Dropout probability for current step
    """
    if schedule == 'constant':
        return base_dropout

    elif schedule == 'linear_increase':
        # Linearly increase dropout from 0 to base_dropout
        progress = min(step / max(total_steps, 1), 1.0)
        return base_dropout * progress

    elif schedule == 'warmup':
        # No dropout during warmup, then constant
        if step < warmup_steps:
            return 0.0
        return base_dropout

    elif schedule == 'cosine':
        # Cosine annealing from 0 to base_dropout
        progress = min(step / max(total_steps, 1), 1.0)
        return base_dropout * (1 - np.cos(np.pi * progress)) / 2

    return base_dropout


class ThreeCurriculumDataset(Dataset):
    """
    PyTorch Dataset for three-curriculum training.

    Each sample consists of:
    - c_S: Static curriculum (expert rules/protocols)
    - c_C: Case curriculum (situation-specific context)
    - c_U: User curriculum (practitioner preferences)
    - x: Task input
    - y: Target output

    The dataset applies curriculum dropout during __getitem__.
    """

    def __init__(
        self,
        samples: List[ThreeCurriculumSample],
        tokenizer,
        curriculum_args,
        training: bool = True,
        total_steps: Optional[int] = None
    ):
        """
        Args:
            samples: List of ThreeCurriculumSample objects
            tokenizer: HuggingFace tokenizer
            curriculum_args: ThreeCurriculumArguments instance
            training: Whether this is for training (applies dropout)
            total_steps: Total training steps (for scheduled dropout)
        """
        self.samples = samples
        self.tokenizer = tokenizer
        self.curriculum_args = curriculum_args
        self.training = training
        self.total_steps = total_steps
        self.current_step = 0

        # Initialize dropout
        self.dropout = CurriculumDropout(
            p_static=curriculum_args.dropout_static,
            p_case=curriculum_args.dropout_case,
            p_user=curriculum_args.dropout_user,
            mode=curriculum_args.channel_dropout_mode,
            replacement=curriculum_args.dropout_replacement,
            mask_token_id=self._get_mask_token_id()
        )

        # Get special token IDs
        self._init_special_tokens()

    def _get_mask_token_id(self) -> Optional[int]:
        """Get the mask token ID if using mask replacement."""
        if self.curriculum_args.dropout_replacement == 'mask':
            mask_token = self.curriculum_args.mask_token
            if mask_token in self.tokenizer.get_vocab():
                return self.tokenizer.convert_tokens_to_ids(mask_token)
        return None

    def _init_special_tokens(self):
        """Initialize special token IDs for curriculum boundaries."""
        vocab = self.tokenizer.get_vocab()

        self.static_start_id = vocab.get(
            self.curriculum_args.static_start_token, None
        )
        self.static_end_id = vocab.get(
            self.curriculum_args.static_end_token, None
        )
        self.case_start_id = vocab.get(
            self.curriculum_args.case_start_token, None
        )
        self.case_end_id = vocab.get(
            self.curriculum_args.case_end_token, None
        )
        self.user_start_id = vocab.get(
            self.curriculum_args.user_start_token, None
        )
        self.user_end_id = vocab.get(
            self.curriculum_args.user_end_token, None
        )

    def set_step(self, step: int):
        """Update current training step for scheduled dropout."""
        self.current_step = step

    def _get_current_dropout_rates(self) -> Tuple[float, float, float]:
        """Get dropout rates based on current training step and schedule."""
        if not self.training:
            return 0.0, 0.0, 0.0

        schedule = self.curriculum_args.dropout_schedule
        warmup = self.curriculum_args.dropout_warmup_steps
        total = self.total_steps or 1

        p_static = get_dropout_schedule(
            self.current_step, total,
            self.curriculum_args.dropout_static,
            schedule, warmup
        )
        p_case = get_dropout_schedule(
            self.current_step, total,
            self.curriculum_args.dropout_case,
            schedule, warmup
        )
        p_user = get_dropout_schedule(
            self.current_step, total,
            self.curriculum_args.dropout_user,
            schedule, warmup
        )

        return p_static, p_case, p_user

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a training sample with curriculum dropout applied.

        Returns a dict with:
        - input_ids: Concatenated sequence [c_S, c_C, c_U, x, y[:-1]]
        - labels: Target sequence with curriculum/input positions masked
        - attention_mask: Attention mask for the sequence
        - curriculum_mask: Binary mask indicating curriculum token positions
        """
        sample = self.samples[idx]

        # Apply curriculum dropout during training
        if self.training:
            p_static, p_case, p_user = self._get_current_dropout_rates()
            self.dropout.p_static = p_static
            self.dropout.p_case = p_case
            self.dropout.p_user = p_user
            sample, dropout_info = self.dropout(sample)

        # Build the concatenated sequence: concat(c_S, c_C, c_U, x, y)
        # Following the paper: s_in = concat(c~, x, y_1, ..., y_{T-1})
        curriculum_tokens = []
        curriculum_length = 0

        # Add static curriculum with boundary tokens
        if sample.c_static:
            if self.static_start_id is not None:
                curriculum_tokens.append(self.static_start_id)
            curriculum_tokens.extend(sample.c_static)
            if self.static_end_id is not None:
                curriculum_tokens.append(self.static_end_id)

        static_end_pos = len(curriculum_tokens)

        # Add case curriculum with boundary tokens
        if sample.c_case:
            if self.case_start_id is not None:
                curriculum_tokens.append(self.case_start_id)
            curriculum_tokens.extend(sample.c_case)
            if self.case_end_id is not None:
                curriculum_tokens.append(self.case_end_id)

        case_end_pos = len(curriculum_tokens)

        # Add user curriculum with boundary tokens
        if sample.c_user:
            if self.user_start_id is not None:
                curriculum_tokens.append(self.user_start_id)
            curriculum_tokens.extend(sample.c_user)
            if self.user_end_id is not None:
                curriculum_tokens.append(self.user_end_id)

        curriculum_length = len(curriculum_tokens)

        # Build input sequence: [curriculum, x, y[:-1]]
        input_tokens = curriculum_tokens + sample.x + sample.y[:-1]

        # Build target sequence: [curriculum, x, y]
        # Note: Labels will have -100 for positions we don't want to compute loss on
        target_tokens = curriculum_tokens + sample.x + sample.y

        # Create labels with masking
        # Following the paper: only target positions (y) contribute to loss
        labels = [-100] * len(target_tokens)
        target_start = curriculum_length + len(sample.x)

        # Only compute loss on target sequence positions
        for i in range(target_start, len(target_tokens)):
            labels[i] = target_tokens[i]

        # Create curriculum mask (1 for curriculum tokens, 0 otherwise)
        curriculum_mask = [1] * curriculum_length + [0] * (len(input_tokens) - curriculum_length)

        return {
            'input_ids': input_tokens,
            'labels': labels[1:] + [-100],  # Shift for autoregressive
            'curriculum_mask': curriculum_mask,
            'curriculum_length': curriculum_length,
            'input_length': len(sample.x),
            'target_length': len(sample.y)
        }


def load_three_curriculum_dataset(
    data_path: str,
    tokenizer,
    curriculum_args,
    split: str = 'train',
    max_samples: Optional[int] = None
) -> ThreeCurriculumDataset:
    """
    Load a three-curriculum dataset from disk.

    Expected data format: List of dicts with keys:
    - 'c_static': Static curriculum text or tokens
    - 'c_case': Case curriculum text or tokens
    - 'c_user': User curriculum text or tokens
    - 'x': Task input text or tokens
    - 'y': Target output text or tokens

    Args:
        data_path: Path to the dataset file (pickle or json)
        tokenizer: HuggingFace tokenizer
        curriculum_args: ThreeCurriculumArguments instance
        split: 'train' or 'eval'
        max_samples: Maximum number of samples to load

    Returns:
        ThreeCurriculumDataset instance
    """
    # Load raw data
    if data_path.endswith('.pkl'):
        with open(data_path, 'rb') as f:
            raw_data = pickle.load(f)
    elif data_path.endswith('.json'):
        import json
        with open(data_path, 'r') as f:
            raw_data = json.load(f)
    else:
        raise ValueError(f"Unsupported data format: {data_path}")

    if max_samples is not None:
        raw_data = raw_data[:max_samples]

    # Convert to ThreeCurriculumSample objects
    samples = []
    for item in raw_data:
        # Handle both text and pre-tokenized inputs
        if isinstance(item.get('c_static', ''), str):
            c_static = tokenizer.encode(item.get('c_static', ''), add_special_tokens=False)
        else:
            c_static = item.get('c_static', [])

        if isinstance(item.get('c_case', ''), str):
            c_case = tokenizer.encode(item.get('c_case', ''), add_special_tokens=False)
        else:
            c_case = item.get('c_case', [])

        if isinstance(item.get('c_user', ''), str):
            c_user = tokenizer.encode(item.get('c_user', ''), add_special_tokens=False)
        else:
            c_user = item.get('c_user', [])

        if isinstance(item.get('x', ''), str):
            x = tokenizer.encode(item.get('x', ''), add_special_tokens=False)
        else:
            x = item.get('x', [])

        if isinstance(item.get('y', ''), str):
            y = tokenizer.encode(item.get('y', ''), add_special_tokens=False)
        else:
            y = item.get('y', [])

        samples.append(ThreeCurriculumSample(
            c_static=c_static,
            c_case=c_case,
            c_user=c_user,
            x=x,
            y=y
        ))

    return ThreeCurriculumDataset(
        samples=samples,
        tokenizer=tokenizer,
        curriculum_args=curriculum_args,
        training=(split == 'train')
    )


def create_synthetic_sample(
    tokenizer,
    c_static_text: str,
    c_case_text: str,
    c_user_text: str,
    x_text: str,
    y_text: str
) -> ThreeCurriculumSample:
    """
    Create a synthetic ThreeCurriculumSample from text inputs.

    Utility function for testing and demo purposes.
    """
    return ThreeCurriculumSample(
        c_static=tokenizer.encode(c_static_text, add_special_tokens=False),
        c_case=tokenizer.encode(c_case_text, add_special_tokens=False),
        c_user=tokenizer.encode(c_user_text, add_special_tokens=False),
        x=tokenizer.encode(x_text, add_special_tokens=False),
        y=tokenizer.encode(y_text, add_special_tokens=False)
    )
