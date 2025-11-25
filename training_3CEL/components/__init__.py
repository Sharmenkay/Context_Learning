"""
Components for Three-Curriculum Context-Enhanced Learning (3CEL).

This package provides the building blocks for training language models
with the three-curriculum architecture:

- arguments: Configuration dataclasses
- data_utils: Dataset classes and curriculum dropout
- data_collator: Batch collation with selective loss masking
- trainer: Custom trainer with curriculum scheduling
- model_utils: Model initialization and special tokens
"""

from .arguments import (
    ModelArguments,
    TrainingArguments,
    DataArguments,
    ThreeCurriculumArguments
)

from .data_utils import (
    ThreeCurriculumSample,
    ThreeCurriculumDataset,
    CurriculumDropout,
    load_three_curriculum_dataset,
    create_synthetic_sample,
    get_dropout_schedule
)

from .data_collator import (
    ThreeCurriculumCollator,
    ThreeCurriculumEvalCollator,
    ThreeCurriculumInferenceCollator
)

from .trainer import (
    ThreeCurriculumTrainer,
    compute_metrics_three_curriculum,
    preprocess_logits_argmax
)

from .model_utils import (
    add_curriculum_tokens,
    resize_model_embeddings,
    archive_datasets,
    get_trainable_parameters,
    freeze_base_model,
    save_curriculum_config
)

__all__ = [
    # Arguments
    'ModelArguments',
    'TrainingArguments',
    'DataArguments',
    'ThreeCurriculumArguments',

    # Data
    'ThreeCurriculumSample',
    'ThreeCurriculumDataset',
    'CurriculumDropout',
    'load_three_curriculum_dataset',
    'create_synthetic_sample',
    'get_dropout_schedule',

    # Collators
    'ThreeCurriculumCollator',
    'ThreeCurriculumEvalCollator',
    'ThreeCurriculumInferenceCollator',

    # Trainer
    'ThreeCurriculumTrainer',
    'compute_metrics_three_curriculum',
    'preprocess_logits_argmax',

    # Model Utils
    'add_curriculum_tokens',
    'resize_model_embeddings',
    'archive_datasets',
    'get_trainable_parameters',
    'freeze_base_model',
    'save_curriculum_config',
]
