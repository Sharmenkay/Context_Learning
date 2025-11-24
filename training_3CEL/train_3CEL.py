"""
Three-Curriculum Context-Enhanced Learning (3CEL) Training Script.

This script implements Algorithm 1 from the paper:
"Architecture Overview: Three-Curriculum Context-Enhanced Learning"

The training procedure:
1. Each sample contains three curriculum channels (cS, cC, cU), task input (x), and target (y)
2. Independent dropout is applied to each curriculum channel
3. Curricula are concatenated: c = concat(cS, cC, cU)
4. Input sequence: sin = concat(c~, x, y1, ..., yT-1)
5. Target sequence: stgt = concat(c~, x, y1, ..., yT)
6. Loss is computed only on target positions (curriculum and input are masked)

Usage:
    python train_3CEL.py --model_name_or_path <model> --train_path <data> ...

    Or with a config file:
    python train_3CEL.py config.json
"""

import os
import sys
import functools
import numpy as np
import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    get_scheduler,
    HfArgumentParser,
    set_seed
)
from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy

from components import (
    ModelArguments,
    TrainingArguments,
    DataArguments,
    ThreeCurriculumArguments,
    ThreeCurriculumDataset,
    ThreeCurriculumSample,
    ThreeCurriculumCollator,
    ThreeCurriculumEvalCollator,
    ThreeCurriculumTrainer,
    add_curriculum_tokens,
    resize_model_embeddings,
    archive_datasets,
    save_curriculum_config,
    compute_metrics_three_curriculum,
    preprocess_logits_argmax,
    load_three_curriculum_dataset
)


def create_demo_dataset(tokenizer, curriculum_args, n_samples=100, training=True):
    """
    Create a demonstration dataset for testing.

    This creates synthetic samples with:
    - Static curriculum: Domain rules/protocols
    - Case curriculum: Situation-specific context
    - User curriculum: Practitioner preferences
    - Task input: Query/problem
    - Target: Expected response
    """
    samples = []

    # Example templates for demonstration
    static_templates = [
        "Protocol: Follow standard procedure A.",
        "Rule: Always verify before proceeding.",
        "Guideline: Prioritize safety measures.",
    ]

    case_templates = [
        "Case context: Patient presents with symptom X.",
        "Situation: High-priority case requiring immediate attention.",
        "Context: Routine check with no complications.",
    ]

    user_templates = [
        "Expertise: Senior practitioner with 10 years experience.",
        "Workload: Currently managing 5 active cases.",
        "Preference: Prefers detailed explanations.",
    ]

    task_templates = [
        "What is the recommended action?",
        "Provide assessment and next steps.",
        "Summarize the situation and suggest approach.",
    ]

    target_templates = [
        "Based on the protocol and context, recommend action A with monitoring.",
        "Assessment complete. Proceed with standard protocol, adjust for case specifics.",
        "Situation analyzed. Follow guideline B with practitioner-adjusted pacing.",
    ]

    for i in range(n_samples):
        np.random.seed(i)

        c_static = tokenizer.encode(
            static_templates[i % len(static_templates)],
            add_special_tokens=False
        )
        c_case = tokenizer.encode(
            case_templates[i % len(case_templates)],
            add_special_tokens=False
        )
        c_user = tokenizer.encode(
            user_templates[i % len(user_templates)],
            add_special_tokens=False
        )
        x = tokenizer.encode(
            task_templates[i % len(task_templates)],
            add_special_tokens=False
        )
        y = tokenizer.encode(
            target_templates[i % len(target_templates)],
            add_special_tokens=False
        )

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
        training=training
    )


def main():
    # Parse arguments
    parser = HfArgumentParser((
        ModelArguments,
        TrainingArguments,
        DataArguments,
        ThreeCurriculumArguments
    ))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # Load from config file
        model_args, training_args, data_args, curriculum_args = \
            parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, training_args, data_args, curriculum_args = \
            parser.parse_args_into_dataclasses()

    # Setup output directory
    os.makedirs(training_args.output_dir, exist_ok=True)

    # Setup logging
    os.environ["WANDB_PROJECT"] = training_args.project_name
    os.environ["WANDB_DIR"] = training_args.output_dir
    if training_args.report_to == 'none':
        os.environ["WANDB_MODE"] = "disabled"

    # Set seed for reproducibility
    set_seed(training_args.seed)
    training_args.remove_unused_columns = False

    print("=" * 60)
    print("Three-Curriculum Context-Enhanced Learning (3CEL)")
    print("=" * 60)
    print(f"Model: {model_args.model_name_or_path}")
    print(f"Output: {training_args.output_dir}")
    print(f"Dropout rates: static={curriculum_args.dropout_static}, "
          f"case={curriculum_args.dropout_case}, "
          f"user={curriculum_args.dropout_user}")
    print("=" * 60)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        legacy=True
    )
    tokenizer.pad_token = tokenizer.eos_token

    # Add curriculum special tokens
    add_curriculum_tokens(tokenizer, curriculum_args)

    # Load or create datasets
    if data_args.train_path and os.path.exists(data_args.train_path):
        print(f"Loading training data from {data_args.train_path}")
        train_dataset = load_three_curriculum_dataset(
            data_args.train_path,
            tokenizer,
            curriculum_args,
            split='train',
            max_samples=data_args.train_samples
        )
    else:
        print("Creating demonstration training dataset")
        train_dataset = create_demo_dataset(
            tokenizer,
            curriculum_args,
            n_samples=data_args.train_samples,
            training=True
        )

    if data_args.eval_path and os.path.exists(data_args.eval_path):
        print(f"Loading evaluation data from {data_args.eval_path}")
        eval_dataset = load_three_curriculum_dataset(
            data_args.eval_path,
            tokenizer,
            curriculum_args,
            split='eval',
            max_samples=data_args.eval_samples
        )
    else:
        print("Creating demonstration evaluation dataset")
        eval_dataset = create_demo_dataset(
            tokenizer,
            curriculum_args,
            n_samples=data_args.eval_samples,
            training=False
        )

    # Compute total training steps
    num_training_steps = (
        len(train_dataset) //
        (training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps)
        * training_args.num_train_epochs
    )
    train_dataset.total_steps = num_training_steps

    print(f"\nDataset sizes:")
    print(f"  Training: {len(train_dataset)}")
    print(f"  Evaluation: {len(eval_dataset)}")
    print(f"  Total training steps: {num_training_steps}")

    # Create evaluation datasets dict (for multiple eval sets)
    eval_datasets = {
        'eval': eval_dataset,
    }

    # Load model
    print(f"\nLoading model: {model_args.model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        torch_dtype=torch.bfloat16,
        device_map='auto'
    )

    # Resize embeddings for new tokens
    resize_model_embeddings(model, tokenizer)

    # Store pretrained weights for regularization
    pretrained_weights = [param.clone().detach() for param in model.parameters()]

    # Custom optimizer with weight decay towards pretrained weights
    class AdamW_PretrainedWD(torch.optim.AdamW):
        """AdamW with weight decay towards pretrained weights."""

        def __init__(self, params, lr, weight_decay_coeff, pretrained_params):
            super().__init__(
                params,
                lr=lr,
                weight_decay=training_args.weight_decay,
                betas=[training_args.adam_beta1, training_args.adam_beta2]
            )
            self.pretrained_params = pretrained_params
            self.weight_decay_coeff = weight_decay_coeff

        def step(self, closure=None):
            if self.weight_decay_coeff > 0:
                for i, param in enumerate(self.param_groups[0]['params']):
                    if param.grad is not None and i < len(self.pretrained_params):
                        param.grad.data += self.weight_decay_coeff * (
                            param.data - self.pretrained_params[i]
                        )
            super().step(closure)

    # Create optimizer and scheduler
    optimizer = AdamW_PretrainedWD(
        model.parameters(),
        lr=training_args.learning_rate,
        weight_decay_coeff=training_args.weight_decay_pretrained,
        pretrained_params=pretrained_weights
    )

    lr_scheduler = get_scheduler(
        name=training_args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=int(num_training_steps * training_args.warmup_ratio),
        num_training_steps=num_training_steps,
    )

    # Create collators
    train_collator = ThreeCurriculumCollator(
        tokenizer=tokenizer,
        mask_curriculum=curriculum_args.mask_curriculum_in_loss,
        mask_input=curriculum_args.mask_input_in_loss,
        max_length=data_args.max_sequence_length
    )

    eval_collator = ThreeCurriculumEvalCollator(
        tokenizer=tokenizer,
        max_length=data_args.max_sequence_length
    )

    # Archive sample datasets
    archive_datasets(train_dataset, eval_datasets, tokenizer, training_args.output_dir)

    # Save curriculum configuration
    save_curriculum_config(curriculum_args, training_args.output_dir)

    # Create trainer
    trainer = ThreeCurriculumTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_datasets,
        data_collator=train_collator,
        eval_collator=eval_collator,
        curriculum_args=curriculum_args,
        compute_metrics=lambda x: compute_metrics_three_curriculum(x, tokenizer),
        optimizers=(optimizer, lr_scheduler),
        preprocess_logits_for_metrics=preprocess_logits_argmax,
    )

    # Setup FSDP if enabled
    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.limit_all_gathers = True
        trainer.accelerator.state.fsdp_plugin.sync_module_states = False

        from torch.distributed.fsdp.fully_sharded_data_parallel import BackwardPrefetch
        trainer.accelerator.state.fsdp_plugin.backward_prefetch = BackwardPrefetch.BACKWARD_PRE

        def fsdp_policy_fn(module):
            return getattr(module, "_fsdp_wrap", False)

        auto_wrap_policy = functools.partial(
            lambda_auto_wrap_policy,
            lambda_fn=fsdp_policy_fn
        )
        trainer.accelerator.state.fsdp_plugin.auto_wrap_policy = auto_wrap_policy

    # Save tokenizer
    tokenizer.save_pretrained(training_args.output_dir)

    # Train
    if training_args.do_train:
        print("\n" + "=" * 60)
        print("Starting training...")
        print("=" * 60)

        train_result = trainer.train()
        trainer.save_model()

        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        print("\n" + "=" * 60)
        print("Training complete!")
        print("=" * 60)

    # Evaluate
    if training_args.do_eval:
        print("\n" + "=" * 60)
        print("Running evaluation...")
        print("=" * 60)

        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    print(f"\nResults saved to: {training_args.output_dir}")


if __name__ == "__main__":
    main()
