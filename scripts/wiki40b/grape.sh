#!/bin/bash
cd ./grape # replace to your directory
pip install -r requirements.txt

export WANDB_API_KEY="put your authorize key here, to find it: https://wandb.ai/authorize"
pip install --upgrade wandb

torchrun --nproc-per-node 2 src/run.py --config_json config/wiki40b/T1-wiki40b.json --wandb_proj multi-wiki --wandb_run WIKI-GRAPE-T1
torchrun --nproc-per-node 2 src/run.py --config_json config/wiki40b/T2-wiki40b.json --wandb_proj multi-wiki --wandb_run WIKI-GRAPE-T2
torchrun --nproc-per-node 2 src/run.py --config_json config/wiki40b/T1-wiki40b-ema.json --wandb_proj multi-wiki --wandb_run WIKI-EMA-T1
torchrun --nproc-per-node 2 src/run.py --config_json config/wiki40b/T2-wiki40b-ema.json --wandb_proj multi-wiki --wandb_run WIKI-EMA-T2
torchrun --nproc-per-node 2 src/run.py --config_json config/wiki40b/T1-wiki40b-gap.json --wandb_proj multi-wiki --wandb_run WIKI-GAP-T1
torchrun --nproc-per-node 2 src/run.py --config_json config/wiki40b/T2-wiki40b-gap.json --wandb_proj multi-wiki --wandb_run WIKI-GAP-T2

