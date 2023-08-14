#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python ../train_bash.py \
    --model_name_or_path /root/autodl-tmp/chatglm2-6b \
    --stage sft \
    --do_train \
    --dataset covid_train,covid_dev \
    --dataset_dir /root/gougang/ChatGLM-Efficient-Tuning/data/covid \
    --finetuning_type lora \
    --output_dir covid/sft \
    --overwrite_cache \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 16 \
    --max_source_length 1024 \
    --max_target_length 128 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 100 \
    --eval_steps 100 \
    --learning_rate 5e-5 \
    --num_train_epochs 20.0 \
    --dev_ratio 0.05 \
    --evaluation_strategy steps \
    --load_best_model_at_end \
    --plot_loss \
    --fp16

