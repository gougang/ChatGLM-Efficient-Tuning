#!/bin/bash
echo "start execute train_sft..."
./train_sft.sh
echo "end execute train_sft..."

echo "start execute train_rm..."
./train_rm.sh
echo "end execute train_rm..."

echo "start execute train_ppo..."
./train_ppo.sh
echo "end execute train_ppo..."