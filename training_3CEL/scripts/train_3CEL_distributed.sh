#!/bin/bash
#
# Distributed Training Script for Three-Curriculum CEL
#
# Uses PyTorch FSDP (Fully Sharded Data Parallel) for multi-GPU training.
#
# Usage:
#   ./train_3CEL_distributed.sh [num_gpus] [model_name] [output_dir]
#
# Example:
#   ./train_3CEL_distributed.sh 4 meta-llama/Llama-3.2-3B-Instruct ./outputs/3cel_distributed

set -e

NUM_GPUS=${1:-4}
MODEL_NAME=${2:-"meta-llama/Llama-3.2-3B-Instruct"}
OUTPUT_DIR=${3:-"./outputs/3cel_distributed_$(date +%Y%m%d_%H%M%S)"}
CACHE_DIR=${CACHE_DIR:-"~/.cache/huggingface"}

# Training hyperparameters (adjusted for distributed training)
LEARNING_RATE=2e-5
BATCH_SIZE=2  # Per GPU
GRADIENT_ACCUMULATION=2
NUM_EPOCHS=3
WARMUP_RATIO=0.1
MAX_SEQ_LENGTH=2048

# Curriculum dropout rates
DROPOUT_STATIC=0.5
DROPOUT_CASE=0.5
DROPOUT_USER=0.5

# Dataset sizes
TRAIN_SAMPLES=50000
EVAL_SAMPLES=2000

echo "=============================================="
echo "Distributed Three-Curriculum CEL Training"
echo "=============================================="
echo "GPUs: ${NUM_GPUS}"
echo "Model: ${MODEL_NAME}"
echo "Output: ${OUTPUT_DIR}"
echo "Effective batch size: $((BATCH_SIZE * NUM_GPUS * GRADIENT_ACCUMULATION))"
echo "=============================================="

mkdir -p ${OUTPUT_DIR}

# FSDP configuration
export FSDP_CPU_RAM_EFFICIENT_LOADING=1
export FSDP_USE_ORIG_PARAMS=1

torchrun --nproc_per_node=${NUM_GPUS} \
    --master_port=29500 \
    ../train_3CEL.py \
    --model_name_or_path ${MODEL_NAME} \
    --cache_dir ${CACHE_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --do_train \
    --do_eval \
    --learning_rate ${LEARNING_RATE} \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --per_device_eval_batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION} \
    --num_train_epochs ${NUM_EPOCHS} \
    --warmup_ratio ${WARMUP_RATIO} \
    --lr_scheduler_type cosine \
    --max_sequence_length ${MAX_SEQ_LENGTH} \
    --train_samples ${TRAIN_SAMPLES} \
    --eval_samples ${EVAL_SAMPLES} \
    --dropout_static ${DROPOUT_STATIC} \
    --dropout_case ${DROPOUT_CASE} \
    --dropout_user ${DROPOUT_USER} \
    --dropout_schedule constant \
    --mask_curriculum_in_loss True \
    --mask_input_in_loss True \
    --channel_dropout_mode full \
    --dropout_replacement empty \
    --eval_strategy steps \
    --eval_steps 200 \
    --save_strategy steps \
    --save_steps 1000 \
    --logging_steps 10 \
    --bf16 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_config '{"transformer_layer_cls_to_wrap": ["LlamaDecoderLayer"]}' \
    --seed 42 \
    --report_to wandb \
    --project_name three_curriculum_cel_distributed

echo "=============================================="
echo "Distributed training complete!"
echo "Results saved to: ${OUTPUT_DIR}"
echo "=============================================="
