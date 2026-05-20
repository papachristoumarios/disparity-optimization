from __future__ import annotations

import dataclasses
import itertools
import math
from sys import setrecursionlimit
import time
import random
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import networkx as nx
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import pandas as pd
import argparse
import os

import cvxpy as cp

from utils import *

import seaborn as sns
import matplotlib.pyplot as plt
from dataclasses import dataclass

sns.set_theme(
    style="whitegrid",
    palette="magma",
    context="paper",
    font_scale=1.75,
    rc={
        "font.size": 15,
        "axes.labelsize": 17,
        "axes.titlesize": 18,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "legend.fontsize": 14,
        "legend.title_fontsize": 15,
        "figure.titlesize": 19,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    },
)

FIGSIZE = 5.5

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_list', type=str, nargs='+', default=['1'])
    parser.add_argument("--out-dir", type=str, default='figures')
    parser.add_argument("--rho", type=float, default=1.0)
    parser.add_argument('--eta', type=float, default=1.0)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--eps', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--size', default='all', choices=['all', 'small', 'tiny'])
    parser.add_argument('--s_type', default='actual', choices=['actual', 'adversarial'])
    parser.add_argument('--embedding_type', default='precomputed', choices=['node2vec', 'gaussian', 'network_structure', 'precomputed'])
    parser.add_argument('--cached_results', action='store_true')
    parser.add_argument('--b', type=int, default=100)
    return parser.parse_args()

def cholesky_add_node(L, z, z_uu, jitter=1e-12):
    if L is None:
        alpha = float(z_uu)
        if alpha <= 0:
            alpha = jitter
        L_new = np.array([[np.sqrt(alpha)]], dtype=float)
        return L_new, alpha

    z = np.asarray(z, dtype=np.float64).ravel()
    if z.size != L.shape[0]:
        raise ValueError(
            f"cholesky_add_node: len(z)={z.size} != L.shape[0]={L.shape[0]}"
        )

    # Solve L v = z  (so v = L^{-1} z)
    v = solve_triangular(L, z, lower=True, check_finite=False)

    # Schur complement
    alpha = float(z_uu - np.dot(v, v))
    if alpha <= jitter:
        alpha = jitter

    m = L.shape[0]
    L_new = np.zeros((m + 1, m + 1), dtype=float)
    L_new[:m, :m] = L
    L_new[m, :m] = v
    L_new[m, m] = np.sqrt(alpha)
    return L_new, alpha

def marginal_gain_for_node(
    Z: np.ndarray,
    Q: Optional[np.ndarray],
    s: np.ndarray,
    S_idx: np.ndarray,
    u: int,
) -> Tuple[float, Optional[np.ndarray], float]:
    """
    Marginal benefit of adding u to S for F(S) = s^T P_S s:
        gain = (s^T q_u)^2 / alpha,  q_u = Z_{:,u} - Z_{:,S} Z_SS^{-1} Z_{S,u}.
    """
    ui = int(u)
    if len(S_idx) == 0:
        alpha = float(Z[ui, ui])
        if alpha <= 1e-12:
            return 0.0, None, alpha
        q_u = Z[:, ui]
        z_col = None
    else:
        z = Z[np.ix_(S_idx, np.array([ui], dtype=np.intp))].reshape(-1)
        v = solve_with_cholesky(Q, z)
        alpha = float(Z[ui, ui] - np.dot(z, v))
        alpha = max(alpha, 1e-12)
        q_u = Z[:, ui] - Z[:, S_idx] @ v
        z_col = z.astype(np.float64, copy=False)

    gain = (float(np.dot(s, q_u)) ** 2) / alpha
    if not np.isfinite(gain):
        gain = 0.0
    return gain, z_col, alpha



def benefit_fixed_set(Z: np.ndarray, s: np.ndarray, S: Sequence[int], ridge: float = 1e-8) -> float:
    """F(S) = s^T P_S s for a fixed set S."""
    if len(S) == 0:
        return 0.0
    S_idx = np.asarray(S, dtype=np.intp)
    ZSS = 0.5 * (Z[np.ix_(S_idx, S_idx)] + Z[np.ix_(S_idx, S_idx)].T)
    ridge_eff = max(ridge, 1e-8 * len(S_idx))
    ZSS = ZSS + ridge_eff * np.eye(len(S_idx))
    try:
        L = np.linalg.cholesky(ZSS)
    except np.linalg.LinAlgError:
        ZSS = project_psd(ZSS, eps=ridge_eff)
        L = np.linalg.cholesky(ZSS)
    v = Z[np.ix_(S_idx, np.arange(Z.shape[0]))] @ s
    x = solve_with_cholesky(L, v)
    return max(float(v @ x), 0.0)

def benefit_on_ground_set_incremental(
    Z: np.ndarray, s: np.ndarray, ground_set: Sequence[int]
) -> float:
    """
    F(S) on the full ground set via incremental marginals (same as greedy additions).
    Avoids an n x n Cholesky when |S| = n.
    """
    state = FastScenarioState(Z=Z, s=s)
    for u in ground_set:
        S_idx = np.asarray(state.S, dtype=np.intp)
        mg, z_col, _ = marginal_gain_for_node(Z, state.Q, s, S_idx, int(u))
        if mg <= 0:
            continue
        state.benefit += mg
        state.add_node(int(u), z_col)
    return state.benefit


def compute_delta_with_factor(Z, Q, s, S, ridge=1e-8, normalize: bool = False):
    n = Z.shape[0]
    # Preserve insertion order: Q matches Z_SS in this row/column order, not sorted(S).
    S = np.asarray(S, dtype=int)
    bar = np.array([i for i in range(n) if i not in set(S.tolist())], dtype=int)

    if len(S) == 0:
        return np.zeros(n), np.array([])

    ZSS = 0.5 * (Z[np.ix_(S, S)] + Z[np.ix_(S, S)].T)
    ZSS = ZSS + ridge * np.eye(len(S))
    

    ZSbar = Z[np.ix_(S, bar)]
    sbar = s[bar]
    rhs = ZSbar @ sbar

    # delta_S* = - Z_SS^{-1} Z_Sbar s_bar
    delta_S = -solve_with_cholesky(Q, rhs)

    if normalize:
        delta_S = delta_S / np.linalg.norm(delta_S)

    delta = np.zeros(n, dtype=float)
    delta[S] = delta_S
    return delta, delta_S


def opinion_seeding_fast(
    G: nx.Graph, 
    s: np.ndarray, 
    C: np.ndarray, 
    name: str, 
    b: int = 100, 
    seed: int = 0) -> pd.DataFrame:
    L = sparse_laplacian(G)
    L0 = L.copy().toarray()
    
    n = len(G)
    eps = 0.1
    q = max(int(np.log(n) / eps**2), 1)

    L_plus_I = L + sp.identity(n, format="csr")
    U, R, X, M = sketch_solve(L_plus_I, q, seed)

    Z = M * C

    records = []

    S = []
    remaining = list(range(n))
    objective_value = 0
    objective_prev = 0
    Q = None

    start_time = time.time()

    for i in range(b):
        best_u = None
        best_gain = -np.inf
        best_z = None
        best_alpha = None

        S_idx = np.asarray(S, dtype=np.intp)

        for u in remaining:
            ui = int(u)
            gain, z_col, alpha = marginal_gain_for_node(Z, Q, s, S_idx, ui)
            if alpha <= 1e-12 or gain <= 1e-9:
                continue

            if gain > best_gain:
                best_gain = gain
                best_u = ui
                best_z = z_col
                best_alpha = alpha

        if best_u is None or best_gain <= 1e-9:
            break

        if len(S) > 0:
            bu = int(best_u)
            best_z = (
                Z[np.ix_(S_idx, np.array([bu], dtype=np.intp))]
                .reshape(-1)
                .astype(np.float64, copy=False)
            )

        # Update Cholesky factor with the chosen node
        bu = int(best_u)
        if len(S) == 0:
            Q, _ = cholesky_add_node(None, np.array([]), Z[bu, bu])
        else:
            Q, _ = cholesky_add_node(Q, best_z, Z[bu, bu])

        S.append(int(bu))
        remaining.remove(bu)
        objective_value += best_gain
        delta, delta_S = compute_delta_with_factor(Z, Q, s, S)

        if i == 0:
            initial_objective_value = objective_value

        records.append({
            'Step': i,
            'Percent Change': (objective_value - initial_objective_value) / (1e-14 + initial_objective_value) * 100,
            'S': list(S),
            'Delta': delta,
            'Delta_S': delta_S,
        })

        if np.isclose(objective_value, objective_prev, rtol=1e-6):
            break

        objective_prev = objective_value

    eta_time = time.time() - start_time

    df = pd.DataFrame(records)

    final_step = df['Step'].max()

    delta_final = df[df['Step'] == final_step]['Delta'].values[0]
    delta_final_S = df[df['Step'] == final_step]['Delta_S'].values[0]

    return df, delta_final, delta_final_S, eta_time

def experiment_1_opinion_seeding_oracle(args: argparse.Namespace) -> None:
    out_dir = args.out_dir

    datasets = get_datasets(args)
   
    if args.cached_results:
        concat_df = pd.read_csv(f'{out_dir}/experiment_1_opinion_seeding_oracle.csv')
        df_delta = pd.read_csv(f'{out_dir}/experiment_1_opinion_seeding_oracle_delta.csv')
    else:
        concat_df = []
        df_delta = []
        for name, group_types in datasets:
            for group_type in group_types:
                print(f"Running {name} with {group_type}")
                G, s, C_bar = load_dataset(name, group_type)
               
                df, delta_final, delta_final_S, eta_time = opinion_seeding_fast(G, s, C_bar, name, args.b, args.seed)

                delta_final = delta_final / np.linalg.norm(delta_final)
                delta_final_S = delta_final_S / np.linalg.norm(delta_final_S)

                df['Name'] = name
                df['Nominal Partition Type'] = group_type
                df['Number of Nodes'] = G.number_of_nodes()
                df['Batch Size'] = args.batch_size
                df['Learning Rate'] = args.eta
                df['Seed'] = args.seed
                df['Time (s)'] = eta_time
                df['Per Step Time (s)'] = eta_time / args.b

                delta_final_sorted = -np.sort(-delta_final_S)
                
                for rank, delta in enumerate(delta_final_sorted):
                    df_delta.append({
                        'Name': name,
                        'Nominal Partition Type': group_type,
                        'Delta': delta,
                        'Rank': rank + 1,
                    })

                concat_df.append(df)

        concat_df = pd.concat(concat_df, ignore_index=True)
        concat_df.to_csv(f'{out_dir}/experiment_1_opinion_seeding_oracle.csv', index=False)

        df_delta = pd.DataFrame(df_delta)
        df_delta.to_csv(f'{out_dir}/experiment_1_opinion_seeding_oracle_delta.csv', index=False)

    # plot percent change in benefit
    num_names = concat_df['Name'].nunique()
    fig_a, ax_a = plt.subplots(nrows=1, ncols=3, figsize=(FIGSIZE * 3, FIGSIZE), squeeze=False)

    sns.lineplot(x='Step', y='Percent Change', hue='Name', style='Nominal Partition Type', data=concat_df, ax=ax_a[0, 0], markers=True, marker='x', markersize=5)

    sns.lineplot(x='Rank', y='Delta', hue='Name', style='Nominal Partition Type', data=df_delta, ax=ax_a[0, 1], markers=True, marker='x', markersize=5)
    ax_a[0, 1].set_xlabel('Rank')
    ax_a[0, 1].set_ylabel('Normalized Intervention $\\delta_S / ||\\delta_S||$')

    sns.barplot(x='Name', y='Time (s)', data=concat_df, ax=ax_a[0, 2])
    ax_a[0, 2].set_title('Runtime')
    ax_a[0, 2].set_xlabel('')
    ax_a[0, 2].set_ylabel('Time (s)')
    ax_a[0, 2].set_yscale('log')

    fig_a.suptitle('Opinion Seeding Oracle')

    fig_a.tight_layout()
    fig_a.savefig(f'{out_dir}/experiment_1_opinion_seeding_oracle.pdf', dpi=300, bbox_inches='tight')

def experiment_2_robust_opinion_seeding_oracle(args: argparse.Namespace) -> None:
    out_dir = args.out_dir

    datasets = get_datasets(args)

    if args.cached_results:
        concat_df = pd.read_csv(f'{out_dir}/experiment_2_robust_opinion_seeding_oracle.csv')
        df_delta = pd.read_csv(f'{out_dir}/experiment_2_robust_opinion_seeding_oracle_delta.csv')
    else:
        concat_df = []
        df_delta = []
        for name, group_types in datasets:
            for group_type in group_types:
                print(f"Running {name} with {group_type}")
                G, s, C_bar = load_dataset(name, group_type)
                
                df, delta_final, delta_final_S, eta_time = robust_opinion_seeding(G, s, C_bar, args.rho, name, args.b, args.seed)

                delta_final = np.asarray(delta_final, dtype=float).reshape(-1)
                delta_final_S = np.asarray(delta_final_S, dtype=float).reshape(-1)
                norm = np.linalg.norm(delta_final_S)
                if norm > 0:
                    delta_final = delta_final / norm
                    delta_final_S = delta_final_S / norm

                df['Name'] = name
                df['Nominal Partition Type'] = group_type
                df['Number of Nodes'] = G.number_of_nodes()
                df['Batch Size'] = args.batch_size
                df['Learning Rate'] = args.eta
                df['Seed'] = args.seed
                df['Time (s)'] = eta_time
                df['Per Step Time (s)'] = eta_time / args.b

                if delta_final_S.size > 0:
                    delta_final_sorted = -np.sort(-delta_final_S)
                    for rank, delta in enumerate(delta_final_sorted):
                        df_delta.append({
                            'Name': name,
                            'Nominal Partition Type': group_type,
                            'Delta': delta,
                            'Rank': rank + 1,
                        })

                concat_df.append(df)

        concat_df = pd.concat(concat_df, ignore_index=True)
        concat_df.to_csv(f'{out_dir}/experiment_2_robust_opinion_seeding_oracle.csv', index=False)

        df_delta = pd.DataFrame(df_delta)
        df_delta.to_csv(f'{out_dir}/experiment_2_robust_opinion_seeding_oracle_delta.csv', index=False)

    # plot percent change in benefit
    num_names = concat_df['Name'].nunique()
    fig_a, ax_a = plt.subplots(nrows=1, ncols=3, figsize=(FIGSIZE * 3, FIGSIZE), squeeze=False)

    sns.lineplot(x='Step', y='Percent Change', hue='Name', style='Nominal Partition Type', data=concat_df, ax=ax_a[0, 0], markers=True, marker='x', markersize=5)
    ax_a[0, 0].set_title('Percent Change in Benefit')
    ax_a[0, 0].set_xlabel('Step')
    ax_a[0, 0].set_ylabel('Percent Change')

    sns.lineplot(x='Rank', y='Delta', hue='Name', style='Nominal Partition Type', data=df_delta, ax=ax_a[0, 1], markers=True, marker='x', markersize=5)
    ax_a[0, 1].set_title('Delta Final Sorted by Rank')
    ax_a[0, 1].set_xlabel('Rank')
    ax_a[0, 1].set_ylabel('Normalized Intervention $\\delta_S / ||\\delta_S||$')

    sns.barplot(x='Name', y='Time (s)', data=concat_df, ax=ax_a[0, 2])
    ax_a[0, 2].set_title('Runtime')
    ax_a[0, 2].set_xlabel('')
    ax_a[0, 2].set_ylabel('Time (s)')
    ax_a[0, 2].set_yscale('log')

    fig_a.suptitle('Robust Opinion Seeding Oracle')

    fig_a.tight_layout()
    fig_a.savefig(f'{out_dir}/experiment_2_robust_opinion_seeding_oracle.pdf', dpi=300, bbox_inches='tight')
    concat_df.to_csv(f'{out_dir}/experiment_2_robust_opinion_seeding_oracle.csv', index=False)
    df_delta.to_csv(f'{out_dir}/experiment_2_robust_opinion_seeding_oracle_delta.csv', index=False)

def main(args: argparse.Namespace) -> None:
    experiment_dict = dict([
        (1, experiment_1_opinion_seeding_oracle),
        (2, experiment_2_robust_opinion_seeding_oracle),
    ])
    
    experiment_list = set()
    for experiment_range in args.experiment_list:
        if '-' in experiment_range:
            start, end = experiment_range.split('-')
            experiment_list.update(range(int(start), int(end) + 1))
        elif experiment_range.isdigit():
            experiment_list.add(int(experiment_range))
        elif experiment_range == 'all':
            experiment_list.update(experiment_dict.keys())
        else:
            raise ValueError(f"Invalid experiment range: {experiment_range}")

    print(f"Running experiments: {list(experiment_list)}")

    for experiment_id in experiment_list:
       print(f"Running experiment {experiment_id}")
       experiment_dict[experiment_id](args)

if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    main(args)
