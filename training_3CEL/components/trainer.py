"""
Custom Trainer for Three-Curriculum Context-Enhanced Learning (3CEL).

This trainer extends HuggingFace's Trainer with:
- Support for curriculum dropout scheduling
- Multiple evaluation datasets
- Custom metrics for three-curriculum training
"""

import torch
import numpy as np
from typing import Dict, Optional, List, Any, Tuple, Union, TYPE_CHECKING
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler

from transformers import Trainer
from transformers.trainer_utils import EvalLoopOutput, has_length
from transformers.trainer_pt_utils import nested_detach

import transformers
import datasets

transformers.logging.set_verbosity_info()


class ThreeCurriculumTrainer(Trainer):
    """
    Trainer for Three-Curriculum Context-Enhanced Learning.

    Extends HuggingFace Trainer with:
    - Step-based dropout scheduling for curriculum channels
    - Separate collators for training and evaluation
    - Support for multiple evaluation datasets
    - Custom metrics computation
    """

    def __init__(
        self,
        *args,
        eval_collator=None,
        curriculum_args=None,
        **kwargs
    ):
        """
        Args:
            eval_collator: Separate collator for evaluation
            curriculum_args: ThreeCurriculumArguments instance
            *args, **kwargs: Arguments passed to parent Trainer
        """
        super().__init__(*args, **kwargs)

        self.eval_collator = eval_collator
        self.curriculum_args = curriculum_args
        self.label_names = ['labels']

        # Training tracking
        self.current_step = 0
        self.total_steps = None

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        """Get the training data sampler."""
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        if not self.args.shuffle_train:
            print("Using Sequential Sampler (no shuffle)", flush=True)
            return SequentialSampler(self.train_dataset)
        else:
            print("Using Random Sampler", flush=True)
            return RandomSampler(self.train_dataset)

    def training_step(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]]
    ) -> torch.Tensor:
        """
        Override training_step to update curriculum dropout rates.
        """
        # Update step counter for curriculum scheduling
        self.current_step += 1

        # Update dataset's step counter for scheduled dropout
        if hasattr(self.train_dataset, 'set_step'):
            self.train_dataset.set_step(self.current_step)

        # Call parent's training_step
        return super().training_step(model, inputs)

    def get_eval_dataloader(
        self,
        eval_dataset: Optional[Union[str, Dataset]] = None
    ) -> DataLoader:
        """
        Get evaluation dataloader with eval-specific collator.
        """
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")

        dataloader_key = eval_dataset if isinstance(eval_dataset, str) else "eval"

        # Check for cached dataloader
        if (
            hasattr(self, "_eval_dataloaders")
            and dataloader_key in self._eval_dataloaders
            and self.args.dataloader_persistent_workers
        ):
            return self.accelerator.prepare(self._eval_dataloaders[dataloader_key])

        eval_dataset = (
            self.eval_dataset[eval_dataset]
            if isinstance(eval_dataset, str)
            else eval_dataset
            if eval_dataset is not None
            else self.eval_dataset
        )

        # Use eval collator if available
        data_collator = self.eval_collator if self.eval_collator else self.data_collator

        if transformers.utils.is_datasets_available() and isinstance(eval_dataset, datasets.Dataset):
            eval_dataset = self._remove_unused_columns(eval_dataset, description="evaluation")
        else:
            data_collator = self._get_collator_with_removed_columns(
                data_collator, description="evaluation"
            )

        dataloader_params = {
            "batch_size": self.args.eval_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(eval_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_eval_sampler(eval_dataset)
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        eval_dataloader = DataLoader(eval_dataset, **dataloader_params)

        if self.args.dataloader_persistent_workers:
            if hasattr(self, "_eval_dataloaders"):
                self._eval_dataloaders[dataloader_key] = eval_dataloader
            else:
                self._eval_dataloaders = {dataloader_key: eval_dataloader}

        return self.accelerator.prepare(eval_dataloader)

    def prediction_step(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step.
        """
        has_labels = any(inputs.get(k) is not None for k in self.label_names)
        return_loss = inputs.get("return_loss", None)

        if return_loss is None:
            return_loss = self.can_return_loss

        loss_without_labels = len(self.label_names) == 0 and return_loss

        inputs = self._prepare_inputs(inputs)

        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        # Get labels before they might be popped
        if has_labels or loss_without_labels:
            labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None

        with torch.no_grad():
            if has_labels or loss_without_labels:
                with self.compute_loss_context_manager():
                    loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                loss = loss.mean().detach()

                if isinstance(outputs, dict):
                    logits = tuple(v for k, v in outputs.items() if k not in ignore_keys + ["loss"])
                else:
                    logits = outputs[1:]
            else:
                loss = None
                with self.compute_loss_context_manager():
                    outputs = model(**inputs)
                if isinstance(outputs, dict):
                    logits = tuple(v for k, v in outputs.items() if k not in ignore_keys)
                else:
                    logits = outputs

                if self.args.past_index >= 0:
                    self._past = outputs[self.args.past_index - 1]

        if prediction_loss_only:
            return (loss, None, None)

        logits = nested_detach(logits)
        if len(logits) == 1:
            logits = logits[0]

        return (loss, logits, labels)


def compute_metrics_three_curriculum(eval_preds, tokenizer=None):
    """
    Compute metrics for three-curriculum evaluation.

    Metrics computed:
    - next_token_accuracy: Accuracy on target tokens
    - perplexity: Perplexity on target tokens
    """
    predictions, labels, inputs = eval_preds
    metrics = {}

    # Shift predictions and labels for autoregressive comparison
    if len(predictions.shape) > 2:
        # predictions are logits
        predictions = predictions.argmax(dim=-1)

    predictions = predictions[..., :-1]
    labels_shifted = labels[..., 1:]

    # Create mask for valid target positions (not -100)
    valid_mask = labels_shifted != -100

    # Compute accuracy on target tokens only
    correct = (predictions == labels_shifted) & valid_mask
    total_correct = correct.sum().item()
    total_valid = valid_mask.sum().item()

    if total_valid > 0:
        metrics['next_token_accuracy'] = total_correct / total_valid
    else:
        metrics['next_token_accuracy'] = 0.0

    # Per-sample accuracy (proportion of correctly predicted targets)
    sample_correct = correct.sum(dim=-1).float()
    sample_valid = valid_mask.sum(dim=-1).float()
    sample_acc = sample_correct / sample_valid.clamp(min=1)
    metrics['mean_sample_accuracy'] = sample_acc.mean().item()

    # Perfect match rate (samples with 100% accuracy)
    perfect_matches = (sample_correct == sample_valid) & (sample_valid > 0)
    metrics['perfect_match_rate'] = perfect_matches.float().mean().item()

    return metrics


def preprocess_logits_argmax(logits, labels):
    """Preprocess logits by taking argmax for metric computation."""
    return logits.argmax(dim=-1)
