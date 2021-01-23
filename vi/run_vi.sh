#!/bin/bash
#SBATCH --partition=batch

env=cp
num_bins_per_dim=100

source ~/virtualenvs/mococo/bin/activate
python3 value_iter.py \
    --env="$env" \
    --num-bins-per-dim="$num_bins_per_dim"
