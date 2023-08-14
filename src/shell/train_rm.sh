#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python ../train_bash.py \
    --model_name_or_path /root/autodl-tmp/chatglm2-6b \
    --stage rm \
    --do_train \
    --dataset comparison_gpt4_zh \
    --dataset_dir /root/gougang/ChatGLM-Efficient-Tuning/data \
    --finetuning_type lora \
    --output_dir covid/rm \
    --overwrite_cache \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --max_source_length 512 \
    --max_target_length 512 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 100 \
    --eval_steps 100 \
    --learning_rate 1e-5 \
    --num_train_epochs 1.0 \
    --dev_ratio 0.05 \
    --evaluation_strategy steps \
    --load_best_model_at_end \
    --plot_loss \
    --fp16

