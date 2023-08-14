#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python ../train_bash.py \
    --model_name_or_path /root/autodl-tmp/chatglm2-6b \
    --stage ppo \
    --do_train \
    --dataset covid_train,covid_dev \
    --dataset_dir /root/gougang/ChatGLM-Efficient-Tuning/data/covid \
    --finetuning_type lora \
    --checkpoint_dir covid/sft \
    --reward_model covid/rm \
    --output_dir covid/ppo \
    --overwrite_cache \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --max_source_length 256 \
    --max_target_length 128 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 100 \
    --learning_rate 1e-5 \
    --num_train_epochs 5.0 \
    --resume_lora_training False \
    --plot_loss

