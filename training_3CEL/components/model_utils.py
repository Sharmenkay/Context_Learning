"""
Model utilities for Three-Curriculum Context-Enhanced Learning (3CEL).

This module provides:
- Special token registration for curriculum boundaries
- Model initialization helpers
- Dataset archiving for reproducibility
"""

import os
import json
import torch
from typing import List, Optional, Dict, Any


def add_curriculum_tokens(tokenizer, curriculum_args):
    """
    Add special tokens for curriculum boundaries to the tokenizer.

    Tokens added:
    - Static curriculum: start/end markers
    - Case curriculum: start/end markers
    - User curriculum: start/end markers
    - Mask token (for dropout replacement)

    Args:
        tokenizer: HuggingFace tokenizer
        curriculum_args: ThreeCurriculumArguments instance

    Returns:
        List of added token strings
    """
    special_tokens = []

    # Curriculum boundary tokens
    special_tokens.extend([
        curriculum_args.static_start_token,
        curriculum_args.static_end_token,
        curriculum_args.case_start_token,
        curriculum_args.case_end_token,
        curriculum_args.user_start_token,
        curriculum_args.user_end_token,
    ])

    # Mask token for dropout replacement
    if curriculum_args.dropout_replacement == 'mask':
        special_tokens.append(curriculum_args.mask_token)

    # Filter out tokens that already exist
    vocab = tokenizer.get_vocab()
    new_tokens = [t for t in special_tokens if t not in vocab]

    if new_tokens:
        tokenizer.add_special_tokens({'additional_special_tokens': new_tokens})
        print(f"Added {len(new_tokens)} special tokens: {new_tokens}")

    return new_tokens


def resize_model_embeddings(model, tokenizer):
    """
    Resize model embeddings to match tokenizer vocabulary.

    Args:
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer (after adding special tokens)
    """
    original_size = model.get_input_embeddings().weight.shape[0]
    new_size = len(tokenizer)

    if new_size != original_size:
        model.resize_token_embeddings(new_size)
        print(f"Resized embeddings from {original_size} to {new_size}")


def archive_datasets(
    train_dataset,
    eval_datasets: Dict[str, Any],
    tokenizer,
    output_dir: str,
    n_samples: int = 10
):
    """
    Archive sample datasets for inspection and reproducibility.

    Args:
        train_dataset: Training dataset
        eval_datasets: Dict of evaluation datasets
        tokenizer: Tokenizer for decoding
        output_dir: Directory to save archives
        n_samples: Number of samples to archive per dataset
    """
    archive_dir = os.path.join(output_dir, 'dataset_archive')
    os.makedirs(archive_dir, exist_ok=True)

    # Archive training samples
    if train_dataset is not None and len(train_dataset) > 0:
        train_samples = []
        for i in range(min(n_samples, len(train_dataset))):
            sample = train_dataset[i]
            decoded = {
                'input_ids': tokenizer.decode(sample['input_ids']),
                'labels': sample['labels'],
                'curriculum_length': sample.get('curriculum_length', 0),
                'input_length': sample.get('input_length', 0),
                'target_length': sample.get('target_length', 0)
            }
            train_samples.append(decoded)

        with open(os.path.join(archive_dir, 'train_samples.json'), 'w') as f:
            json.dump(train_samples, f, indent=2, ensure_ascii=False)

    # Archive evaluation samples
    for name, dataset in eval_datasets.items():
        if dataset is not None and len(dataset) > 0:
            eval_samples = []
            for i in range(min(n_samples, len(dataset))):
                sample = dataset[i]
                decoded = {
                    'input_ids': tokenizer.decode(sample['input_ids']),
                    'labels': sample['labels'],
                    'curriculum_length': sample.get('curriculum_length', 0),
                    'input_length': sample.get('input_length', 0),
                    'target_length': sample.get('target_length', 0)
                }
                eval_samples.append(decoded)

            filename = f'{name}_samples.json'
            with open(os.path.join(archive_dir, filename), 'w') as f:
                json.dump(eval_samples, f, indent=2, ensure_ascii=False)

    print(f"Archived dataset samples to {archive_dir}")


def get_trainable_parameters(model, print_details: bool = True) -> Dict[str, Any]:
    """
    Get statistics about trainable parameters.

    Args:
        model: PyTorch model
        print_details: Whether to print parameter details

    Returns:
        Dict with parameter statistics
    """
    trainable_params = 0
    all_params = 0

    for name, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            if print_details:
                print(f"  Trainable: {name} - {param.numel():,} params")

    stats = {
        'trainable_params': trainable_params,
        'all_params': all_params,
        'trainable_percent': 100 * trainable_params / all_params if all_params > 0 else 0
    }

    print(f"\nTrainable params: {trainable_params:,} / {all_params:,} "
          f"({stats['trainable_percent']:.2f}%)")

    return stats


def freeze_base_model(model, unfreeze_layers: Optional[List[str]] = None):
    """
    Freeze base model parameters, optionally unfreezing specific layers.

    Args:
        model: HuggingFace model
        unfreeze_layers: List of layer name patterns to keep trainable
    """
    unfreeze_layers = unfreeze_layers or []

    for name, param in model.named_parameters():
        should_unfreeze = any(pattern in name for pattern in unfreeze_layers)
        param.requires_grad = should_unfreeze

    print(f"Froze base model, unfroze layers matching: {unfreeze_layers}")


def save_curriculum_config(curriculum_args, output_dir: str):
    """
    Save curriculum configuration for reproducibility.

    Args:
        curriculum_args: ThreeCurriculumArguments instance
        output_dir: Directory to save config
    """
    config = {
        'dropout_static': curriculum_args.dropout_static,
        'dropout_case': curriculum_args.dropout_case,
        'dropout_user': curriculum_args.dropout_user,
        'dropout_schedule': curriculum_args.dropout_schedule,
        'dropout_warmup_steps': curriculum_args.dropout_warmup_steps,
        'channel_dropout_mode': curriculum_args.channel_dropout_mode,
        'dropout_replacement': curriculum_args.dropout_replacement,
        'mask_curriculum_in_loss': curriculum_args.mask_curriculum_in_loss,
        'mask_input_in_loss': curriculum_args.mask_input_in_loss,
    }

    config_path = os.path.join(output_dir, 'curriculum_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"Saved curriculum config to {config_path}")
