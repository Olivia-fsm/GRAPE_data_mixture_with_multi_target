#!/bin/bash
cd ./grape # replace to your directory
pip install -r requirements.txt

export WANDB_API_KEY="put your authorize key here, to find it: https://wandb.ai/authorize"
pip install --upgrade wandb

torchrun --nproc-per-node 2 src/run.py --config_json config/wiki40b/T2-base.json --wandb_proj multi-wiki --wandb_run BASE-T1-T2
