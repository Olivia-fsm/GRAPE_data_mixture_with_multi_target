#!/bin/bash
cd ./grape # replace to your directory
pip install -r requirements.txt

# export WANDB_API_KEY="put your authorize key here, to find it: https://wandb.ai/authorize"
export WANDB_API_KEY="1dba8bb7f1589f867fa1538683d77eaf4e8209de"
pip install --upgrade wandb

torchrun --nproc_per_node 2 src/run.py --config_json ./config/climb/base.json --wandb_proj CLIMB-GRAPE --wandb_run BASE
torchrun --nproc_per_node 2 src/run.py --config_json ./config/climb/regmix.json --wandb_proj CLIMB-GRAPE --wandb_run REGMIX
torchrun --nproc_per_node 2 src/run.py --config_json ./config/climb/crisp.json --wandb_proj CLIMB-GRAPE --wandb_run CRISP
torchrun --nproc_per_node 2 src/run.py --config_json ./config/climb/climb.json --wandb_proj CLIMB-GRAPE --wandb_run CLIMB
torchrun --nproc_per_node 2 src/run.py --config_json ./config/climb/doge.json --wandb_proj CLIMB-GRAPE --wandb_run DOGE
torchrun --nproc_per_node 2 src/run.py --config_json ./config/climb/grape-mu10.json --wandb_proj CLIMB-GRAPE --wandb_run GRAPE-mu[10]
torchrun --nproc_per_node 2 src/run.py --config_json ./config/climb/grape-mu5.json --wandb_proj CLIMB-GRAPE --wandb_run GRAPE-mu[5]
torchrun --nproc_per_node 2 src/run.py --config_json ./config/climb/clime-grape-mu10.json --wandb_proj CLIMB-GRAPE --wandb_run GRAPE-mu[10]
torchrun --nproc_per_node 2 src/run.py --config_json ./config/climb/clime-grape-mu5.json --wandb_proj CLIMB-GRAPE --wandb_run GRAPE-mu[5]
