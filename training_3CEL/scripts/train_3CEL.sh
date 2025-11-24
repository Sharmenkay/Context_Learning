#!/bin/bash
#
# Three-Curriculum Context-Enhanced Learning (3CEL) Training Script
#
# This script trains a language model using the three-curriculum architecture:
# - cS: Static curriculum (expert rules/protocols)
# - cC: Case curriculum (situation-specific context)
# - cU: User curriculum (practitioner preferences)
#
# Usage:
#   ./train_3CEL.sh [model_name] [output_dir]
#
# Example:
#   ./train_3CEL.sh meta-llama/Llama-3.2-1B-Instruct ./outputs/3cel_run1

set -e

# Default arguments
MODEL_NAME=${1:-"meta-llama/Llama-3.2-1B-Instruct"}
OUTPUT_DIR=${2:-"./outputs/3cel_$(date +%Y%m%d_%H%M%S)"}
CACHE_DIR=${CACHE_DIR:-"~/.cache/huggingface"}

# Training hyperparameters
LEARNING_RATE=1e-5
BATCH_SIZE=4
GRADIENT_ACCUMULATION=4
NUM_EPOCHS=3
WARMUP_RATIO=0.1
MAX_SEQ_LENGTH=2048

# Curriculum dropout rates (Algorithm 1)
DROPOUT_STATIC=0.5
DROPOUT_CASE=0.5
DROPOUT_USER=0.5
DROPOUT_SCHEDULE="constant"

# Dataset sizes
TRAIN_SAMPLES=10000
EVAL_SAMPLES=1000

echo "=============================================="
echo "Three-Curriculum Context-Enhanced Learning"
echo "=============================================="
echo "Model: ${MODEL_NAME}"
echo "Output: ${OUTPUT_DIR}"
echo "=============================================="
echo "Curriculum Dropout Rates:"
echo "  Static (cS): ${DROPOUT_STATIC}"
echo "  Case (cC):   ${DROPOUT_CASE}"
echo "  User (cU):   ${DROPOUT_USER}"
echo "=============================================="

# Create output directory
mkdir -p ${OUTPUT_DIR}

# Run training
python ../train_3CEL.py \
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
    --dropout_schedule ${DROPOUT_SCHEDULE} \
    --mask_curriculum_in_loss True \
    --mask_input_in_loss True \
    --channel_dropout_mode full \
    --dropout_replacement empty \
    --eval_strategy steps \
    --eval_steps 100 \
    --save_strategy steps \
    --save_steps 500 \
    --logging_steps 10 \
    --bf16 \
    --seed 42 \
    --report_to wandb \
    --project_name three_curriculum_cel

echo "=============================================="
echo "Training complete!"
echo "Results saved to: ${OUTPUT_DIR}"
echo "=============================================="
