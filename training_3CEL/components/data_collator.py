"""
Data Collators for Three-Curriculum Context-Enhanced Learning (3CEL).

This module provides collators that handle:
- Batching of variable-length sequences
- Selective loss masking (curriculum and input tokens excluded from loss)
- Proper padding and attention mask generation
"""

import numpy as np
import torch
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


class ThreeCurriculumCollator:
    """
    Data collator for three-curriculum training.

    Implements the masking scheme from Algorithm 1 where:
    - Curriculum tokens (c_S, c_C, c_U) are masked from loss
    - Task input tokens (x) are masked from loss
    - Only target tokens (y) contribute to the gradient

    The training objective is:
    L(theta) = -sum_{t in T} log p_theta(y_t | c~, x, y_{<t})

    where T is the set of target token positions.
    """

    def __init__(
        self,
        tokenizer,
        mask_curriculum: bool = True,
        mask_input: bool = True,
        max_length: Optional[int] = None
    ):
        """
        Args:
            tokenizer: HuggingFace tokenizer
            mask_curriculum: Whether to mask curriculum tokens from loss
            mask_input: Whether to mask input tokens from loss
            max_length: Maximum sequence length (for truncation)
        """
        self.tokenizer = tokenizer
        self.mask_curriculum = mask_curriculum
        self.mask_input = mask_input
        self.max_length = max_length
        self.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of features.

        Args:
            features: List of dicts from ThreeCurriculumDataset.__getitem__

        Returns:
            Batched tensors for training:
            - input_ids: [batch_size, seq_len]
            - labels: [batch_size, seq_len] with -100 for masked positions
            - attention_mask: [batch_size, seq_len]
        """
        batch_size = len(features)

        # Extract sequences
        input_ids_list = [f['input_ids'] for f in features]
        labels_list = [f['labels'] for f in features]
        curriculum_lengths = [f.get('curriculum_length', 0) for f in features]
        input_lengths = [f.get('input_length', 0) for f in features]

        # Find max length
        max_seq_len = max(len(ids) for ids in input_ids_list)
        if self.max_length is not None:
            max_seq_len = min(max_seq_len, self.max_length)

        # Initialize padded arrays
        padded_input_ids = np.full(
            (batch_size, max_seq_len),
            fill_value=self.pad_token_id,
            dtype=np.int64
        )
        padded_labels = np.full(
            (batch_size, max_seq_len),
            fill_value=-100,
            dtype=np.int64
        )
        attention_mask = np.zeros((batch_size, max_seq_len), dtype=np.int64)

        for i, (input_ids, labels) in enumerate(zip(input_ids_list, labels_list)):
            # Truncate if necessary
            seq_len = min(len(input_ids), max_seq_len)
            truncated_input = input_ids[:seq_len]
            truncated_labels = labels[:seq_len]

            # Fill padded arrays
            padded_input_ids[i, :seq_len] = truncated_input
            attention_mask[i, :seq_len] = 1

            # Apply selective masking for loss computation
            for j in range(seq_len):
                label = truncated_labels[j] if j < len(truncated_labels) else -100

                # Mask curriculum positions
                if self.mask_curriculum and j < curriculum_lengths[i]:
                    label = -100

                # Mask input positions (curriculum_length to curriculum_length + input_length)
                elif self.mask_input:
                    input_start = curriculum_lengths[i]
                    input_end = input_start + input_lengths[i]
                    if input_start <= j < input_end:
                        label = -100

                padded_labels[i, j] = label

        return {
            'input_ids': torch.tensor(padded_input_ids, dtype=torch.long),
            'labels': torch.tensor(padded_labels, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long)
        }


class ThreeCurriculumEvalCollator:
    """
    Collator for evaluation that always masks curriculum and input.

    During evaluation, we want consistent behavior regardless of training settings.
    """

    def __init__(self, tokenizer, max_length: Optional[int] = None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch_size = len(features)

        input_ids_list = [f['input_ids'] for f in features]
        labels_list = [f['labels'] for f in features]
        curriculum_lengths = [f.get('curriculum_length', 0) for f in features]
        input_lengths = [f.get('input_length', 0) for f in features]

        max_seq_len = max(len(ids) for ids in input_ids_list)
        if self.max_length is not None:
            max_seq_len = min(max_seq_len, self.max_length)

        padded_input_ids = np.full(
            (batch_size, max_seq_len),
            fill_value=self.pad_token_id,
            dtype=np.int64
        )
        padded_labels = np.full(
            (batch_size, max_seq_len),
            fill_value=-100,
            dtype=np.int64
        )
        attention_mask = np.zeros((batch_size, max_seq_len), dtype=np.int64)

        for i, (input_ids, labels) in enumerate(zip(input_ids_list, labels_list)):
            seq_len = min(len(input_ids), max_seq_len)
            padded_input_ids[i, :seq_len] = input_ids[:seq_len]
            attention_mask[i, :seq_len] = 1

            # For eval, mask everything except targets
            target_start = curriculum_lengths[i] + input_lengths[i]
            for j in range(seq_len):
                if j >= target_start and j < len(labels):
                    padded_labels[i, j] = labels[j]

        return {
            'input_ids': torch.tensor(padded_input_ids, dtype=torch.long),
            'labels': torch.tensor(padded_labels, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long)
        }


class ThreeCurriculumInferenceCollator:
    """
    Collator for inference/generation.

    Prepares sequences for generation by providing only the curriculum and input,
    without the target sequence.
    """

    def __init__(self, tokenizer, max_length: Optional[int] = None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch_size = len(features)

        # Build prompt sequences (curriculum + input, no target)
        prompt_list = []
        for f in features:
            curriculum_len = f.get('curriculum_length', 0)
            input_len = f.get('input_length', 0)
            prompt_len = curriculum_len + input_len
            prompt = f['input_ids'][:prompt_len]
            prompt_list.append(prompt)

        max_prompt_len = max(len(p) for p in prompt_list)
        if self.max_length is not None:
            max_prompt_len = min(max_prompt_len, self.max_length)

        # Left-pad for generation (common for decoder-only models)
        padded_input_ids = np.full(
            (batch_size, max_prompt_len),
            fill_value=self.pad_token_id,
            dtype=np.int64
        )
        attention_mask = np.zeros((batch_size, max_prompt_len), dtype=np.int64)

        for i, prompt in enumerate(prompt_list):
            seq_len = min(len(prompt), max_prompt_len)
            # Left padding
            start_idx = max_prompt_len - seq_len
            padded_input_ids[i, start_idx:] = prompt[:seq_len]
            attention_mask[i, start_idx:] = 1

        return {
            'input_ids': torch.tensor(padded_input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long)
        }
