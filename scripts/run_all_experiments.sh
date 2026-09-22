#!/bin/bash

rm -rf logs
mkdir -p logs

rm -rf figures
mkdir -p figures

for experiment_list in 1 2 3 4 5 6 7 8 9; do
    echo "Submitting link recommendation experiment $experiment_list"
    sbatch scripts/run_link_experiments.sh $experiment_list small
done

for experiment_list in 1 2 3 4; do
    echo "Submitting opinion seeding experiment $experiment_list"
    sbatch scripts/run_seeding_experiments.sh $experiment_list small
done