# Disparity optimization

Code for reproducing the link-recommendation and opinion-seeding experiments in the paper. The main entry points are `link_recommendation.py` and `opinion_seeding.py`. The notebook `disparity.ipynb` contains the same logic in exploratory form.

## Setup

Use Python 3.11 or newer. From the repository root:

```bash
conda create -n disparity-optimization python=3.11
conda activate disparity-optimization
pip install -r requirements.txt
```


```bash
pip install karateclub
```

## Data

Graphs and opinions live under `data/`. Each dataset directory needs:

- `edges.txt` — two or three whitespace-separated columns: `u v` or `u v weight`
- `opinions.txt` — two columns: `node_id opinion`
- `embeddings.npy` - node embeddings (produced either from true node features or Node2Vec)

Bundled datasets: `reddit`, `twitter`, `polblogs`, and Twitch language networks (`twitch-DE`, `twitch-ES`, etc.). Raw Twitch data can be reprocessed with `data/twitch/preprocess.py` if needed.

## Running experiments

Results (CSV tables and PDF figures) are written to `figures/` by default. Logs from cluster jobs go to `logs/`.

### Link recommendation

| ID | Output prefix (examples) |
|----|--------------------------|
| 0 | `experiment_0_network_statistics` |
| 1 | `experiment_1_link_recommendation_oracle` |
| 2 | `experiment_2_link_recommendation_oracle` |
| 3 | `experiment_3_worst_case_C_oracle` |
| 4 | `experiment_4_robust_link_recommendation_oracle` |
| 5 | `experiment_5_fiedler_gradient_ascent` |
| 6 | `experiment_6_link_recommendation_baselines` |
| 7 | `experiment_7_fiedler_baselines` |
| 8 | `experiment_8_robust_link_recommendation_baselines` |
| 9 | `experiment_9_predictive_model` |

```bash
# Single experiment
python link_recommendation.py --experiment_list 3 --size small

# Several experiments
python link_recommendation.py --experiment_list 1 2 3 --size small

# Inclusive range (runs 4, 5, 6)
python link_recommendation.py --experiment_list 4-6 --size small

# All link experiments
python link_recommendation.py --experiment_list all --size small
```

### Opinion seeding

| ID | Output prefix (examples) |
|----|--------------------------|
| 1 | `experiment_1_opinion_seeding_oracle` |
| 2 | `experiment_2_robust_opinion_seeding_random_scenarios` |
| 3 | `experiment_3_opinion_seeding_baselines` |
| 4 | `experiment_4_robust_opinion_seeding_active_set` |

```bash
python opinion_seeding.py --experiment_list 1 --size small
python opinion_seeding.py --experiment_list 1-4 --size small
python opinion_seeding.py --experiment_list all --size small
```

### Useful CLI flags

Both drivers accept:

| Flag | Default | Description |
|------|---------|-------------|
| `--experiment_list` | `1` | Experiment ID(s), ranges (`4-6`), or `all` |
| `--size` | `all` | Passed through for batch scripts; dataset selection is fixed in `utils.get_datasets()` |
| `--out-dir` | `figures` | Directory for CSV/PDF outputs |
| `--seed` | `0` | Random seed |
| `--cached_results` | off | Skip computation; rebuild plots from existing CSVs in `--out-dir` |
| `--s_type` | `actual` | Opinion vector: `actual` or `adversarial` (link experiments) |

`link_recommendation.py` also exposes `--batch_size`, `--rho`, `--eps`, `opinion_seeding.py` exposes `--b` (budget) and `--ridge` (regularization).

### Running the experiments on a cluster with SLURM

`scripts/run_all_experiments.sh` submits the experiments used in the paper `--size small` which includes the polblogs, twitter, and reddit datasets. The script creates `logs/` and `figures/` before submitting.

```bash
bash scripts/run_all_experiments.sh
```

To run one batch job manually (after loading your environment):

```bash
mkdir -p logs figures
sbatch scripts/run_link_experiments.sh 3 small      # link experiment 3
sbatch scripts/run_seeding_experiments.sh 1 small   # opinion seeding experiment 1
```
