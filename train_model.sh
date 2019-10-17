#!/usr/bin/env bash
source /etc/profile
export CUDA_VISIBLE_DEVICES='0'
nohup python -u train.py --network resnet 2>&1 > log.log &