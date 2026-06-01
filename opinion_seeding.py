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

OPINION_SEEDING_METHOD_LABELS: Dict[str, str] = {
    "greedy": "Greedy (marginal gain)",
    "random": "Random",
    "max_degree": "Max degree",
    "pagerank": "PageRank",
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_list', type=str, nargs='+', default=['1'])
    parser.add_argument("--out-dir", type=str, default='figures')
    parser.add_argument("--rho", type=float, default=0.2)
    parser.add_argument('--eps', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--size', default='all', choices=['all', 'small', 'tiny'])
    parser.add_argument('--s_type', default='actual', choices=['actual', 'adversarial'])
    parser.add_argument('--embedding_type', default='precomputed', choices=['node2vec', 'gaussian', 'network_structure', 'precomputed'])
    parser.add_argument('--cached_results', action='store_true')
    parser.add_argument('--b', type=int, default=100)
    parser.add_argument('--ridge', type=float, default=2.0)
    return parser.parse_args()


def marginal_gain_for_node(
    Z: np.ndarray,
    Q: Optional[np.ndarray],
    s: np.ndarray,
    S_idx: np.ndarray,
    u: int,
    ridge: float = 0.0,
) -> Tuple[float, Optional[np.ndarray], float]:

    ui = int(u)
    if len(S_idx) == 0:
        alpha = float(Z[ui, ui] + ridge)
        if alpha <= 1e-12:
            return 0.0, None, alpha
        q_u = Z[:, ui]
        z_col = None
    else:
        z = Z[np.ix_(S_idx, np.array([ui], dtype=np.intp))].reshape(-1)
        v = solve_with_cholesky(Q, z)
        alpha = float(Z[ui, ui] + ridge - np.dot(z, v))
        alpha = max(alpha, 1e-12)
        q_u = Z[:, ui] - Z[:, S_idx] @ v
        z_col = z.astype(np.float64, copy=False)

    gain = (float(np.dot(s, q_u)) ** 2) / alpha
    if not np.isfinite(gain):
        gain = 0.0
    return gain, z_col, alpha


def benefit_fixed_set(Z: np.ndarray, s: np.ndarray, S: Sequence[int], ridge: float = 1e-8) -> float:
    """F(S) with (Z_SS + ridge I)^{-1} in the pseudoinverse / projection."""
    if len(S) == 0:
        return 0.0
    S_idx = np.asarray(S, dtype=np.intp)
    ZSS = 0.5 * (Z[np.ix_(S_idx, S_idx)] + Z[np.ix_(S_idx, S_idx)].T)
    ZSS = ZSS + ridge * np.eye(len(S_idx))
    try:
        L = np.linalg.cholesky(ZSS)
    except np.linalg.LinAlgError:
        ZSS = project_psd(ZSS, eps=1e-8)
        L = np.linalg.cholesky(ZSS)
    v = Z[np.ix_(S_idx, np.arange(Z.shape[0]))] @ s
    x = solve_with_cholesky(L, v)
    return max(float(v @ x), 0.0)


def _baseline_node_order(G: nx.Graph, selection: str, seed: int) -> List[int]:
    """Fixed seeding order for non-greedy baselines (nodes are 0..n-1)."""
    n = G.number_of_nodes()
    if selection == "max_degree":
        return sorted(range(n), key=lambda u: G.degree(u), reverse=True)
    if selection == "random":
        order = list(range(n))
        rng = np.random.default_rng(seed)
        rng.shuffle(order)
        return order
    if selection == "pagerank":
        pr = nx.pagerank(G)
        return sorted(range(n), key=lambda u: pr[u], reverse=True)
    raise ValueError(f"Unknown baseline selection: {selection}")


def _append_seeded_node(
    Z: np.ndarray,
    Q: Optional[np.ndarray],
    s: np.ndarray,
    S: List[int],
    bu: int,
    ridge: float = 0.0,
) -> Tuple[Optional[np.ndarray], float, np.ndarray, np.ndarray]:
    """Add node ``bu`` to ``S``, update Cholesky ``Q``, return gain and intervention."""
    S_idx = np.asarray(S, dtype=np.intp)
    gain, z_col, alpha = marginal_gain_for_node(Z, Q, s, S_idx, int(bu), ridge=ridge)

    if len(S) > 0:
        z_col = (
            Z[np.ix_(S_idx, np.array([int(bu)], dtype=np.intp))]
            .reshape(-1)
            .astype(np.float64, copy=False)
        )

    z_uu = float(Z[int(bu), int(bu)] + ridge)
    if len(S) == 0:
        Q_new, _ = cholesky_add_node(None, np.array([]), z_uu)
    else:
        Q_new, _ = cholesky_add_node(Q, z_col, z_uu)

    S.append(int(bu))
    delta, delta_S = compute_delta_with_factor(Z, Q_new, s, S)
    return Q_new, float(gain), delta, delta_S


def opinion_seeding_fast(
    G: nx.Graph, 
    s: np.ndarray, 
    C: np.ndarray, 
    name: str, 
    b: int = 100, 
    seed: int = 0,
    selection: str = "greedy",
    ridge: float = 1e-8,
) -> pd.DataFrame:
    if selection not in OPINION_SEEDING_METHOD_LABELS:
        raise ValueError(f"selection must be one of {list(OPINION_SEEDING_METHOD_LABELS)}")

    L = sparse_laplacian(G)
    L0 = L.copy().toarray()
    
    n = len(G)
    eps = 0.1
    q = max(int(np.log(n) / eps**2), 1)

    L_plus_I = L + sp.identity(n, format="csr")
    U, R, X, M = sketch_solve(L_plus_I, q, seed)

    Z = M * C

    records = []

    S: List[int] = []
    remaining = list(range(n))
    objective_value = 0.0
    objective_prev = 0.0
    Q = None
    initial_objective_value: Optional[float] = None

    baseline_order: Optional[List[int]] = None
    if selection != "greedy":
        baseline_order = _baseline_node_order(G, selection, seed)

    start_time = time.time()

    for i in range(b):
        best_u: Optional[int] = None
        best_gain = -np.inf

        S_idx = np.asarray(S, dtype=np.intp)

        if selection == "greedy":
            for u in remaining:
                ui = int(u)
                gain, _, alpha = marginal_gain_for_node(
                    Z, Q, s, S_idx, ui, ridge=ridge
                )
                if alpha <= 1e-12 or gain <= 1e-9:
                    continue

                if gain > best_gain:
                    best_gain = gain
                    best_u = ui

            if best_u is None or best_gain <= 1e-9:
                break
            bu = int(best_u)
            step_gain = float(best_gain)
        else:
            assert baseline_order is not None
            if i >= len(baseline_order):
                break
            bu = int(baseline_order[i])
            step_gain, _, _ = marginal_gain_for_node(
                Z, Q, s, S_idx, bu, ridge=ridge
            )

        Q, step_gain, delta, delta_S = _append_seeded_node(
            Z, Q, s, S, bu, ridge=ridge
        )
        remaining.remove(bu)
        objective_value += step_gain

        if initial_objective_value is None:
            initial_objective_value = objective_value

        records.append({
            'Step': i,
            'Percent Change': (objective_value - initial_objective_value) / (1e-14 + initial_objective_value) * 100,
            'Objective Value': objective_value,
            'S': list(S),
            'Delta': delta,
            'Delta_S': delta_S,
        })

        if selection == "greedy" and np.isclose(objective_value, objective_prev, rtol=1e-6):
            break

        objective_prev = objective_value

    eta_time = time.time() - start_time

    df = pd.DataFrame(records)

    final_step = df['Step'].max()

    delta_final = df[df['Step'] == final_step]['Delta'].values[0]
    delta_final_S = df[df['Step'] == final_step]['Delta_S'].values[0]

    return df, delta_final, delta_final_S, eta_time

def robust_opinion_seeding_fast(
    G: nx.Graph,
    s: np.ndarray,
    Cs: List[np.ndarray],
    name: str,
    b: int = 100,
    seed: int = 0,
    ridge: float = 1e-8,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, float]:
    if len(Cs) == 0:
        raise ValueError("Cs must be a non-empty list")

    L = sparse_laplacian(G)

    n = len(G)
    eps = 0.1
    q = max(int(np.log(n) / eps**2), 1)

    L_plus_I = L + sp.identity(n, format="csr")
    U, R, X, M = sketch_solve(L_plus_I, q, seed)

    oracles = [
        ScenarioOracle(Z=M * C_scenario, s=s, ridge=ridge) for C_scenario in Cs
    ]
    Z_nom = M * Cs[0]

    start_time = time.time()
    S_ordered, _ = saturate_fast_helper(
        oracles,
        list(range(n)),
        max_budget=b,
    )
    eta_time = time.time() - start_time

    records: List[dict] = []
    S: List[int] = []
    Q = None
    initial_objective_value: Optional[float] = None

    for i, bu in enumerate(S_ordered[:b]):
        Q, _, delta, delta_S = _append_seeded_node(
            Z_nom, Q, s, S, int(bu), ridge=ridge
        )
        objective_value = min(
            oracle.benefit(set(S)) for oracle in oracles
        )

        if initial_objective_value is None:
            initial_objective_value = objective_value

        records.append({
            'Step': i,
            'Percent Change': (objective_value - initial_objective_value) / (1e-14 + initial_objective_value) * 100,
            'Objective Value': objective_value,
            'S': list(S),
            'Delta': delta,
            'Delta_S': delta_S,
        })

    df = pd.DataFrame(records)

    if len(df) == 0:
        delta_final = np.zeros(n, dtype=float)
        delta_final_S = np.array([], dtype=float)
        return df, delta_final, delta_final_S, eta_time

    final_step = df['Step'].max()
    delta_final = df[df['Step'] == final_step]['Delta'].values[0]
    delta_final_S = df[df['Step'] == final_step]['Delta_S'].values[0]

    return df, delta_final, delta_final_S, eta_time

def robust_opinion_seeding_active_set(
    G: nx.Graph,
    s: np.ndarray,
    Cbar: np.ndarray,
    rho: float,
    name: str,
    b: int = 100,
    seed: int = 0,
    K : int = 10,
    ridge: float = 1e-8,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, float]:
    
    L = sparse_laplacian(G)
    L0 = L.copy().toarray()
    s0 = s.copy()
    
    n = len(G)
    eps = 0.1
    q = max(int(np.log(n) / eps**2), 1)
    
    L_plus_I = L + sp.identity(n, format="csr")
    U0, R0, X0, M0 = sketch_solve(L_plus_I, q, seed)
    Z = M0 * Cbar

    initial_disparity = s0.T @ (M0 * Cbar) @ s0
    initial_benefit = 0

    initial_Z_cond = np.linalg.cond(Z + ridge * np.eye(n))
    initial_alpha_rob = 1 + initial_Z_cond * np.log(1)
    initial_b_rob = int(b * initial_alpha_rob)

    print('Initial Z cond', initial_Z_cond)

    active_set = [(Cbar.copy(),  initial_disparity, initial_benefit, initial_Z_cond, initial_b_rob)]

    records_inner = []  
    records_outer = []  
    eta_time = 0

    cbar_fro = float(np.linalg.norm(Cbar, "fro"))
    cbar_spec = float(np.linalg.norm(Cbar, 2))

    records_outer.append({
        'Name': name,
        'Metric': 'Worst Disparity',
        'Percent Change': 0.0,
        'Value': initial_disparity,
        'k': 0,
        'Rho': rho,
        'Number of Nodes': n,
        'Budget': b,
        'Robust Budget': initial_b_rob,
    })
    records_outer.append({
        'Name': name,
        'Metric': 'Benefit',
        'Percent Change': 0.0,
        'Value': initial_benefit,
        'k': 0,
        'Rho': rho,
        'Number of Nodes': n,
        'Budget': b,
        'Robust Budget': initial_b_rob,
    })
    records_outer.append({
        'Name': name,
        'Metric': 'Frobenius deviation',
        'Percent Change': 0.0,
        'Value': 0.0,
        'k': 0,
        'Rho': rho,
        'Number of Nodes': n,
        'Budget': b,
        'Robust Budget': initial_b_rob,
    })
    records_outer.append({
        'Name': name,
        'Metric': 'Spectral deviation',
        'Percent Change': 0.0,
        'Value': 0.0,
        'k': 0,
        'Rho': rho,
        'Number of Nodes': n,
        'Budget': b,
        'Robust Budget': initial_b_rob,
    })

    for k in range(K):
        C0, disparity_current, benefit_current, Z_cond_current, b_rob_current = active_set[-1]
        print(f"Step k = {k + 1} / {K}, current robust budget: {b_rob_current}, benefit: {benefit_current}")

        Cs = [a[0].copy() for a in active_set]

        df_inner, delta_final, delta_final_S, S_eta_time = robust_opinion_seeding_fast(
            G, s, Cs, name, b=b_rob_current, seed=seed, ridge=ridge, 
        )
        df_inner['k'] = k + 1

        eta_time += S_eta_time
        records_inner.append(df_inner)

        s_new = s0.copy() + delta_final

        C_new, C_eta_time = worst_case_C_for_fixed_L(L=L0, X=X0, Cbar=Cbar, rho=rho, C_prev=C0, s=s_new)

        disparity_new = s_new.T @ (M0 * C_new) @ s_new
        benefit_new = s0.T @ (M0 * C_new) @ s0.T - disparity_new

        eta_time += C_eta_time


        Z_cond_new = np.linalg.cond(M0 * C_new + ridge * np.eye(n))
        alpha_rob = 1 + Z_cond_new * np.log(k + 2)
        b_rob_new = max(b_rob_current, int(b * alpha_rob))


        records_outer.append({
            'Name': name,
            'Metric': 'Disparity',
            'Percent Change': (disparity_new - initial_disparity) / (1e-14 + initial_disparity) * 100,
            'Value': disparity_new,
            'k': k + 1,
            'Rho': rho,
            'Number of Nodes': n,
            'Budget': b,
            'Robust Budget': b_rob_new,
        })

        records_outer.append({
            'Name': name,
            'Metric': 'Benefit',
            'Percent Change': (benefit_new - initial_benefit) / (1e-14 + initial_benefit) * 100,
            'Value': benefit_new,
            'k': k + 1,
            'Rho': rho,
            'Number of Nodes': n,
            'Budget': b,
            'Robust Budget': b_rob_new,
        })

        diff_C = C_new - Cbar
        dev_fro = float(np.linalg.norm(diff_C, "fro"))
        dev_spec = float(np.linalg.norm(diff_C, 2))
        
        records_outer.append(
            {
                "Name": name,
                "Metric": "Frobenius deviation",
                "Percent Change": float(100.0 * dev_fro / (cbar_fro + 1e-14)),
                "Value": dev_fro,
                "k": k + 1,
                "Rho": rho,
                "Number of Nodes": n,
                "Budget": b,
                "Robust Budget": b_rob_new,
            }
        )
        records_outer.append(
            {
                "Name": name,
                "Metric": "Spectral deviation",
                "Percent Change": float(100.0 * dev_spec / (cbar_spec + 1e-14)),
                "Value": dev_spec,
                "k": k + 1,
                "Rho": rho,
                "Number of Nodes": n,
                "Budget": b,
                "Robust Budget": b_rob_new,
            }
        )

        delta_final_prev = delta_final.copy()
        delta_final_S_prev = delta_final_S.copy()

        if benefit_new < benefit_current:
            active_set.append((C_new.copy(), disparity_new, benefit_new, Z_cond_new, b_rob_new))
        else:
            break

    df_inner = pd.concat(records_inner, ignore_index=True)
    df_outer = pd.DataFrame(records_outer)

    return df_inner, df_outer, delta_final_prev, delta_final_S_prev, eta_time

def robust_opinion_seeding_random_scenarios(
    G: nx.Graph,
    s: np.ndarray,
    C: np.ndarray,
    rho: float,
    name: str,
    b: int = 100,
    seed: int = 0,
    num_scenarios: int = 3,
    ridge: float = 1e-8,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, float]:
    Cs = generate_scenarios(C, rho, num_scenarios)
    return robust_opinion_seeding_fast(
        G, s, Cs, name, b=b, seed=seed, ridge=ridge,
    )


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
               
                df, delta_final, delta_final_S, eta_time = opinion_seeding_fast(
                    G, s, C_bar, name, args.b, args.seed, ridge=args.ridge
                )

                delta_final = delta_final / np.linalg.norm(delta_final)
                delta_final_S = delta_final_S / np.linalg.norm(delta_final_S)

                df['Name'] = name
                df['Nominal Partition Type'] = group_type
                df['Number of Nodes'] = G.number_of_nodes()
                df['Budget'] = args.b
                df['Seed'] = args.seed
                df['Ridge'] = args.ridge
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

def experiment_2_robust_opinion_seeding_random_scenarios(args: argparse.Namespace) -> None:
    out_dir = args.out_dir

    datasets = get_datasets(args)

    if args.cached_results:
        concat_df = pd.read_csv(f'{out_dir}/experiment_2_robust_opinion_seeding_random_scenarios.csv')
        df_delta = pd.read_csv(f'{out_dir}/experiment_2_robust_opinion_seeding_random_scenarios_delta.csv')
    else:
        concat_df = []
        df_delta = []
        for name, group_types in datasets:
            for group_type in group_types:
                print(f"Running {name} with {group_type}")
                G, s, C_bar = load_dataset(name, group_type)
                
                df, delta_final, delta_final_S, eta_time = robust_opinion_seeding_random_scenarios(
                    G,
                    s,
                    C_bar,
                    args.rho,
                    name,
                    args.b,
                    args.seed,
                    num_scenarios=3,
                    ridge=args.ridge,
                )

                delta_final = np.asarray(delta_final, dtype=float).reshape(-1)
                delta_final_S = np.asarray(delta_final_S, dtype=float).reshape(-1)
                norm = np.linalg.norm(delta_final_S)
                if norm > 0:
                    delta_final = delta_final / norm
                    delta_final_S = delta_final_S / norm

                df['Name'] = name
                df['Nominal Partition Type'] = group_type
                df['Number of Nodes'] = G.number_of_nodes()
                df['Budget'] = args.b
                df['Seed'] = args.seed
                df['Ridge'] = args.ridge
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
        concat_df.to_csv(f'{out_dir}/experiment_2_robust_opinion_seeding_random_scenarios.csv', index=False)

        df_delta = pd.DataFrame(df_delta)
        df_delta.to_csv(f'{out_dir}/experiment_2_robust_opinion_seeding_random_scenarios_delta.csv', index=False)

    # plot percent change in benefit
    num_names = concat_df['Name'].nunique()
    fig_a, ax_a = plt.subplots(nrows=1, ncols=3, figsize=(FIGSIZE * 3, FIGSIZE), squeeze=False)

    sns.lineplot(x='Step', y='Percent Change', hue='Name', style='Nominal Partition Type', data=concat_df, ax=ax_a[0, 0], markers=True, marker='x', markersize=5)
    ax_a[0, 0].set_xlabel('Step')
    ax_a[0, 0].set_ylabel('Percent Change in Benefit')

    sns.lineplot(x='Rank', y='Delta', hue='Name', style='Nominal Partition Type', data=df_delta, ax=ax_a[0, 1], markers=True, marker='x', markersize=5)
    ax_a[0, 1].set_xlabel('Rank')
    ax_a[0, 1].set_ylabel('Normalized Intervention $\\delta_S / ||\\delta_S||$')

    sns.barplot(x='Name', y='Time (s)', data=concat_df, ax=ax_a[0, 2])
    ax_a[0, 2].set_xlabel('')
    ax_a[0, 2].set_ylabel('Time (s)')
    ax_a[0, 2].set_yscale('log')

    fig_a.suptitle('Robust Opinion Seeding Oracle (Random Scenarios)')

    fig_a.tight_layout()
    fig_a.savefig(f'{out_dir}/experiment_2_robust_opinion_seeding_random_scenarios.pdf', dpi=300, bbox_inches='tight')
    concat_df.to_csv(f'{out_dir}/experiment_2_robust_opinion_seeding_random_scenarios.csv', index=False)
    df_delta.to_csv(f'{out_dir}/experiment_2_robust_opinion_seeding_random_scenarios_delta.csv', index=False)


def experiment_4_robust_opinion_seeding_active_set(args: argparse.Namespace) -> None:
    out_dir = args.out_dir

    datasets = get_datasets(args)
    # datasets = [('twitter', ['spectral'])]

    if args.cached_results:
        concat_df_inner = pd.read_csv(f'{out_dir}/experiment_4_robust_opinion_seeding_active_set_inner.csv')
        concat_df_outer = pd.read_csv(f'{out_dir}/experiment_4_robust_opinion_seeding_active_set_outer.csv')
        df_delta = pd.read_csv(f'{out_dir}/experiment_4_robust_opinion_seeding_active_set_delta.csv')
    else:
        concat_df_inner = []
        concat_df_outer = []
        df_delta = []
        rho = args.rho
        for name, group_types in datasets:
            for group_type in group_types:
                print(f"Running {name} with {group_type}")
                G, s, C_bar = load_dataset(name, group_type)
                
                df_inner, df_outer, delta_final, delta_final_S, eta_time = robust_opinion_seeding_active_set(
                    G,
                    s,
                    C_bar,
                    rho,
                    name,
                    args.b,
                    args.seed,
                    K=3,
                    ridge=args.ridge,
                )

                df_inner['Name'] = name 
                df_inner['Nominal Partition Type'] = group_type
                df_inner['Number of Nodes'] = G.number_of_nodes()
                df_inner['Seed'] = args.seed
                df_inner['Time (s)'] = eta_time
                df_inner['Budget'] = args.b
                df_inner['Rho'] = rho
                df_inner['Ridge'] = args.ridge

                df_outer['Name'] = name
                df_outer['Nominal Partition Type'] = group_type
                df_outer['Number of Nodes'] = G.number_of_nodes()
                df_outer['Seed'] = args.seed
                df_outer['Time (s)'] = eta_time
                df_outer['Budget'] = args.b
                df_outer['Rho'] = rho
                df_outer['Ridge'] = args.ridge
                concat_df_inner.append(df_inner)
                concat_df_outer.append(df_outer)

                delta_final_S = delta_final_S / np.linalg.norm(delta_final_S)
                delta_final = delta_final / np.linalg.norm(delta_final)

                if delta_final_S.size > 0:
                    delta_final_sorted = -np.sort(-delta_final_S)
                    for rank, delta in enumerate(delta_final_sorted):
                        df_delta.append({
                            'Name': name,
                            'Nominal Partition Type': group_type,
                            'Delta': delta,
                            'Rank': rank + 1,
                        })


        concat_df_inner = pd.concat(concat_df_inner, ignore_index=True)
        concat_df_outer = pd.concat(concat_df_outer, ignore_index=True)
        concat_df_inner.to_csv(f'{out_dir}/experiment_4_robust_opinion_seeding_active_set_inner.csv', index=False)
        concat_df_outer.to_csv(f'{out_dir}/experiment_4_robust_opinion_seeding_active_set_outer.csv', index=False)

        df_delta = pd.DataFrame(df_delta)
        df_delta.to_csv(f'{out_dir}/experiment_4_robust_opinion_seeding_active_set_delta.csv', index=False)

    fig_a, ax_a = plt.subplots(nrows=1, ncols=3, figsize=(FIGSIZE * 3, FIGSIZE), squeeze=False)
    concat_df_inner_benefit = concat_df_inner[concat_df_inner['k'] == concat_df_inner['k'].max()]

    sns.lineplot(x='Step', y='Objective Value', hue='Name', data=concat_df_inner_benefit, ax=ax_a[0, 0], markers=True, marker='o', markersize=5)
    ax_a[0, 0].set_xlabel('Step')
    ax_a[0, 0].set_ylabel('Benefit Function')

    sns.lineplot(x='Rank', y='Delta', hue='Name', data=df_delta, ax=ax_a[0, 1], markers=True, marker='o', markersize=5)
    ax_a[0, 1].set_xlabel('Rank')
    ax_a[0, 1].set_ylabel(f'Normalized Intervention $\\delta_{{\\hat S_K}} / ||\\delta_{{ \\hat S_K }}||$')

    sns.barplot(x='Name', y='Time (s)', data=concat_df_inner_benefit, ax=ax_a[0, 2])
    ax_a[0, 2].set_xlabel('')
    ax_a[0, 2].set_ylabel('Time (s)')
    ax_a[0, 2].set_yscale('log')

    fig_a.suptitle('Robust Opinion Seeding Active Set (Final Result)')
    fig_a.tight_layout()
    fig_a.savefig(f'{out_dir}/experiment_4_robust_opinion_seeding_active_set_inner_final.pdf', dpi=300, bbox_inches='tight')
    

def experiment_3_opinion_seeding_baselines(args: argparse.Namespace) -> None:
    """Compare greedy opinion seeding with degree, random, and PageRank baselines."""
    out_dir = args.out_dir
    datasets = get_datasets(args)
    selections = ["greedy", "random", "max_degree", "pagerank"]

    if args.cached_results:
        df_all = pd.read_csv(f"{out_dir}/experiment_3_opinion_seeding_baselines.csv")
    else:
        concat_df: List[pd.DataFrame] = []
        for name, group_types in datasets:
            for group_type in group_types:
                print(f"Running {name} with {group_type}")
                G, s, C_bar = load_dataset(name, group_type)
                for sel in selections:
                    print(f"  {OPINION_SEEDING_METHOD_LABELS[sel]}")
                    df, _, _, eta_time = opinion_seeding_fast(
                        G,
                        s,
                        C_bar,
                        name,
                        args.b,
                        args.seed,
                        selection=sel,
                        ridge=args.ridge,
                    )
                    df["Name"] = name
                    df["Nominal Partition Type"] = group_type
                    df["Method"] = OPINION_SEEDING_METHOD_LABELS[sel]
                    df["Selection"] = sel
                    df["Number of Nodes"] = G.number_of_nodes()
                    df["Budget"] = args.b
                    df["Seed"] = args.seed
                    df["Ridge"] = args.ridge
                    df["Time (s)"] = eta_time
                    df["Per Step Time (s)"] = eta_time / max(len(df), 1)
                    concat_df.append(df)

        df_all = pd.concat(concat_df, ignore_index=True)
        df_all = df_all[np.isfinite(df_all["Percent Change"])].copy()

    allowed_methods = {OPINION_SEEDING_METHOD_LABELS[sel] for sel in selections}
    df_all = df_all[df_all["Method"].isin(allowed_methods)].copy()
    df_all.to_csv(f"{out_dir}/experiment_3_opinion_seeding_baselines.csv", index=False)

    step_max = df_all["Step"].max()
    df_last = df_all[df_all["Step"] == step_max].copy()

    g = sns.catplot(
        data=df_last,
        x="Method",
        y="Objective Value",
        hue="Nominal Partition Type",
        col="Name",
        kind="bar",
        sharey=True,
        legend=True,
        height=FIGSIZE,
        aspect=1.2,
    )
    g.set_titles(template="{col_name}")
    g.set_axis_labels("Method", "Objective Value")
    g.fig.suptitle("Opinion seeding baselines (final step)", y=1.02)
    g.tight_layout()
    g.savefig(f"{out_dir}/experiment_3a_opinion_seeding_baselines.pdf", dpi=300, bbox_inches="tight")
    plt.close("all")

    fig, axes = plt.subplots(
        1,
        df_all["Name"].nunique(),
        figsize=(FIGSIZE * df_all["Name"].nunique(), FIGSIZE),
        squeeze=False,
    )
    for i, name in enumerate(df_all["Name"].unique()):
        cell = df_all[df_all["Name"] == name]
        sns.lineplot(
            data=cell,
            x="Step",
            y="Objective Value",
            hue="Method",
            style="Nominal Partition Type",
            ax=axes[0, i],
            markers=True,
            marker="x",
            markersize=5,
            legend=(i == df_all["Name"].nunique() - 1),
        )
        axes[0, i].set_title(name)
        axes[0, i].set_xlabel("Step")
        if i == 0:
            axes[0, i].set_ylabel("Objective Value")
        else:
            axes[0, i].set_ylabel("")
    fig.suptitle("Opinion seeding baselines")
    fig.tight_layout()
    fig.savefig(f"{out_dir}/experiment_3b_opinion_seeding_baselines.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main(args: argparse.Namespace) -> None:
    experiment_dict = dict([
        (1, experiment_1_opinion_seeding_oracle),
        (2, experiment_2_robust_opinion_seeding_random_scenarios),
        (3, experiment_3_opinion_seeding_baselines),
        (4, experiment_4_robust_opinion_seeding_active_set),
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
