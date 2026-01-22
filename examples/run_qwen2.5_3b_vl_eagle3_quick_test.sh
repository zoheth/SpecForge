#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)

# Quick test script for Qwen2.5-VL-3B on 16G GPU
# Optimized for fast verification with minimal data
NUM_GPUS=${1:-1}
BUILD_DATASET_NUM_PROC=${BUILD_DATASET_NUM_PROC:-64}

# Set CUDA_HOME to virtual environment's CUDA libraries
export CUDA_HOME=$ROOT_DIR/.venv/lib/python3.12/site-packages/nvidia/cuda_nvrtc
export CUDA_PATH=$CUDA_HOME

torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    $ROOT_DIR/scripts/train_eagle3.py \
    --target-model-path /home/x/models/models/Qwen/Qwen2___5-VL-3B-Instruct \
    --draft-model-config $ROOT_DIR/configs/qwen2-5-vl-eagle3.json \
    --train-data-path $ROOT_DIR/cache/dataset/quick_test_train.jsonl \
    --build-dataset-num-proc $BUILD_DATASET_NUM_PROC \
    --output-dir $ROOT_DIR/outputs/Qwen2.5-VL-3B-eagle3-quick-test \
    --num-epochs 1 \
    --batch-size 1 \
    --learning-rate 1e-4 \
    --max-length 4096 \
    --dist-timeout 360 \
    --chat-template qwen2-vl \
    --cache-dir $ROOT_DIR/cache \
    --embedding-key model.embed_tokens.weight \
    --tp-size 1 \
    --is-vlm \
    --min-pixels 25088 \
    --max-pixels 401408
