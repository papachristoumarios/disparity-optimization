#!/usr/bin/env bash
#SBATCH --job-name=opinion-seeding
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=htc

EXPERIMENT_LIST=${1:-1-3}
SIZE=${2:-small}

module load mamba/latest

source activate disparity-optimization

python opinion_seeding.py --experiment_list $EXPERIMENT_LIST --size $SIZE
