from __future__ import annotations

import argparse
import random
from re import S
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import seaborn as sns
import time
from tqdm import tqdm
from utils import *
import os
rng = np.random.default_rng(0)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

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


def _pct_change_vs_ref(curr: float, ref: float) -> float:
    if not np.isfinite(curr):
        return 0.0
    if not np.isfinite(ref) or abs(ref) < 1e-20:
        return 0.0
    return float((curr - ref) / (abs(ref) + 1e-14) * 100.0)


def _append_auxiliary_inner_records(
    records: List[dict],
    i: int,
    M: np.ndarray,
    Cbar_for_groups: Optional[np.ndarray],
    cum_abs_weight: float,
    edges_removed_cum: int,
    intra_abs_cum: float,
    inter_abs_cum: float,
    init_mean_cross: float,
    init_mean_within: float,
    initial_edge_mass: float,
    initial_num_edges: int,
) -> None:
    if Cbar_for_groups is None:
        return
    mc, mw = mean_M_cross_within_pairs(M, Cbar_for_groups)
    records.append(
        {
            "Step": i,
            "Metric": "Mean M cross-group",
            "Percent Change": _pct_change_vs_ref(mc, init_mean_cross),
            "Value": mc,
        }
    )
    records.append(
        {
            "Step": i,
            "Metric": "Mean M within-group",
            "Percent Change": _pct_change_vs_ref(mw, init_mean_within),
            "Value": mw,
        }
    )
    den_m = initial_edge_mass + 1e-14
    den_e = float(initial_num_edges) + 1e-14
    records.append(
        {
            "Step": i,
            "Metric": "Edges removed (cumulative)",
            "Percent Change": float(100.0 * edges_removed_cum / den_e),
            "Value": float(edges_removed_cum),
        }
    )
    records.append(
        {
            "Step": i,
            "Metric": "Intra-group abs weight Δ (cum.)",
            "Percent Change": float(100.0 * intra_abs_cum / den_m),
            "Value": float(intra_abs_cum),
        }
    )
    records.append(
        {
            "Step": i,
            "Metric": "Inter-group abs weight Δ (cum.)",
            "Percent Change": float(100.0 * inter_abs_cum / den_m),
            "Value": float(inter_abs_cum),
        }
    )

    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_list', type=str, nargs='+', default=['1'])
    parser.add_argument("--out-dir", type=str, default='figures')
    parser.add_argument("--rho", type=float, default=1.0)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--eps', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--size', default='all', choices=['all', 'small', 'tiny'])
    parser.add_argument('--s_type', default='actual', choices=['actual', 'adversarial'])
    parser.add_argument('--embedding_type', default='precomputed', choices=['node2vec', 'gaussian', 'network_structure', 'precomputed'])
    parser.add_argument('--cached_results', action='store_true')
    parser.add_argument('--betweenness_refresh', default=100, type=int)
    return parser.parse_args()


def robust_link_recommendation(G: nx.Graph, Cbar: np.ndarray, rho: float, name: str, q: int, K: int = 10, T_L: int = 50, T_C: int = 20, eta_L: float = 1.0, eta_C: float = 2.0, batch_size: int = 100, seed: int = 0) -> None:
    L0 = sparse_laplacian(G)
    L0_plus_I = L0 + sp.identity(L0.shape[0], format="csr")
    U0, R0, X0, M0 = sketch_solve(L0_plus_I, q, seed)
    H = G.copy()
    n = G.number_of_nodes()

    initial_surrogate_disparity = float(top_eigenpair(X0 * Cbar)[0])
    initial_disparity = float(top_eigenpair(M0 * Cbar)[0])
    initial_polarization = float(top_eigenpair(M0)[0])


    active_set = [(Cbar.copy(), L0, M0, X0, initial_surrogate_disparity, initial_disparity, initial_polarization)]


    records_outer = []
    records_inner = []

    records_outer.append({
        'Name': name,
        'Metric': 'Worst Surrogate Disparity',
        'Percent Change': 0.0,
        'Value': initial_surrogate_disparity,
        'k' : 0,
        'Rho': rho,
        'Number of Nodes': n,
    })
    records_outer.append({
        'Name': name,
        'Metric': 'Worst Disparity',
        'Percent Change': 0.0,
        'Value': initial_disparity,
        'k' : 0,
        'Rho': rho,
        'Number of Nodes': n,
    })
    records_outer.append({
        'Name': name,
        'Metric': 'Worst Polarization',
        'Percent Change': 0.0,
        'Value': initial_polarization,
        'k' : 0,
        'Rho': rho,
        'Number of Nodes': n,
    })
    cbar_fro = float(np.linalg.norm(Cbar, "fro"))
    cbar_spec = float(np.linalg.norm(Cbar, 2))
    records_outer.append(
        {
            "Name": name,
            "Metric": "Frobenius deviation",
            "Percent Change": 0.0,
            "Value": 0.0,
            "k": 0,
            "Rho": rho,
            "Number of Nodes": n,
        }
    )
    records_outer.append(
        {
            "Name": name,
            "Metric": "Spectral deviation",
            "Percent Change": 0.0,
            "Value": 0.0,
            "k": 0,
            "Rho": rho,
            "Number of Nodes": n,
        }
    )

    eta_time = 0

    for k in range(K):
        C0, L0, M0, X0, worst_surrogate_disparity_current, worst_disparity_current, worst_polarization_current = active_set[-1]

        df_inner, L, X, M, H, L_eta_time = link_recommendation(
            G=G,
            C=C0,
            s=None,
            name=name,
            T_L=T_L,
            batch_size=batch_size,
            eta=np.sqrt(G.number_of_edges() / 2),
            seed=seed,
            Cbar_for_groups=Cbar,
        )
        records_inner.append(df_inner)
        eta_time += L_eta_time

        C_new, C_eta_time = worst_case_C_for_fixed_L(L=L, X=X, Cbar=Cbar, rho=rho, C_prev=C0)

        worst_surrogate_disparity_new = float(top_eigenpair(X * C_new)[0])
        worst_disparity_new = float(top_eigenpair(M * C_new)[0])
        worst_polarization_new = float(top_eigenpair(M)[0])

        eta_time += C_eta_time

        if np.isclose(worst_surrogate_disparity_new, worst_surrogate_disparity_current, rtol=1e-6):
            break

        active_set.append((C_new, L, M, X, worst_surrogate_disparity_new, worst_disparity_new, worst_polarization_new))

        records_outer.append({
            'Name': name,
            'Metric': 'Worst Surrogate Disparity',
            'Percent Change': (worst_surrogate_disparity_new - initial_surrogate_disparity) / initial_surrogate_disparity * 100,
            'Value': worst_surrogate_disparity_new,
            'k' : k + 1,
            'Rho': rho,
            'Number of Nodes': n,
        })
        records_outer.append({
            'Name': name,
            'Metric': 'Worst Disparity',
            'Percent Change': (worst_disparity_new - initial_disparity) / initial_disparity * 100,
            'Value': worst_disparity_new,
            'k' : k + 1,
            'Rho': rho,
            'Number of Nodes': n,
        })
        records_outer.append({
            'Name': name,
            'Metric': 'Worst Polarization',
            'Percent Change': (worst_polarization_new - initial_polarization) / initial_polarization * 100,
            'Value': worst_polarization_new,
            'k' : k + 1,
            'Rho': rho,
            'Number of Nodes': n,
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
            }
        )

    df_outer = pd.DataFrame(records_outer)
    df_inner = pd.concat(records_inner, ignore_index=True)
    return df_outer, df_inner, eta_time, M0, M


def link_recommendation(
    G: nx.Graph,
    s: np.ndarray,
    C: np.ndarray,
    name: str,
    T_L: int = 100,
    batch_size: int = 100,
    eta: float = 1,
    seed: int = 0,
    Cbar_for_groups: Optional[np.ndarray] = None,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, nx.Graph]:
    nodelist = list(G.nodes())
    B = nx.incidence_matrix(G, nodelist=nodelist, oriented=True).toarray()
    edge_to_col = {}
    for j, e in enumerate(G.edges()):
        u_e, v_e = e
        edge_to_col[(u_e, v_e)] = j
        edge_to_col[(v_e, u_e)] = j
    L = sparse_laplacian(G)
    L0 = L.copy().toarray()

    H = G.copy()

    n = len(G.nodes())
    eps = 0.1
    q = max(int(np.log(n) / eps**2), 1)
    rng = np.random.default_rng(seed)
    L_plus_I = L + sp.identity(n, format="csr")
    U, R, X, M = sketch_solve(L_plus_I, q, seed)

    T_refresh = int(np.sqrt(G.number_of_edges()))

    records = []

    start_time = time.time()

    progress_bar = tqdm(total=T_L, desc="Link Recommendations")


    if s is not None:
        initial_disparity = s.T @ (M * C) @ s
        initial_polarization = s.T @ M @ s
        initial_surrogate_disparity = s.T @ (X * C) @ s
        s_type = 'actual'
    else:
        initial_disparity, s = top_eigenpair(M * C)
        initial_polarization, s = top_eigenpair(M)
        initial_surrogate_disparity, s = top_eigenpair(X * C)
        s_type = 'adversarial'


    C_mtx = C * np.outer(s, s)

    initial_L0_fro = np.linalg.norm(L0, 'fro')

    surrogate_disparity_prev = initial_surrogate_disparity

    if Cbar_for_groups is not None:
        init_mean_cross, init_mean_within = mean_M_cross_within_pairs(M, Cbar_for_groups)
        initial_edge_mass = float(sum(float(H[u][v]["weight"]) for u, v in H.edges()))
        initial_num_edges = int(H.number_of_edges())
    else:
        init_mean_cross = float("nan")
        init_mean_within = float("nan")
        initial_edge_mass = 0.0
        initial_num_edges = 0

    cum_abs_weight = 0.0
    edges_removed_cum = 0
    intra_abs_cum = 0.0
    inter_abs_cum = 0.0

    # use tqdm to show progress
    for i in range(T_L):
        # pick one edge
        edges = random.sample(list(H.edges()), k=batch_size)

        cols = np.fromiter((edge_to_col[e] for e in edges), dtype=np.intp, count=len(edges))
        Be = B[:, cols]
        Xe = (X @ Be).T

        leverage_arr = np.apply_along_axis(lambda x: x.T @ C_mtx @ x, axis=1, arr=Xe)

        idx_plus = int(np.argmax(leverage_arr))
        idx_minus = int(np.argmin(leverage_arr))
        edge_plus = edges[idx_plus]
        edge_minus = edges[idx_minus]
        u_plus, v_plus = edge_plus
        u_minus, v_minus = edge_minus

        eta_current = eta / np.sqrt(i + 1)
        weight_change = max(0.0, min(eta_current, H[u_plus][v_plus]['weight'], H[u_minus][v_minus]['weight']))

        if Cbar_for_groups is not None:
            cum_abs_weight += 2.0 * weight_change
            bp = edge_inter_intra_bucket(int(u_plus), int(v_plus), Cbar_for_groups)
            bm = edge_inter_intra_bucket(int(u_minus), int(v_minus), Cbar_for_groups)
            if bp == "intra":
                intra_abs_cum += weight_change
            else:
                inter_abs_cum += weight_change
            if bm == "intra":
                intra_abs_cum += weight_change
            else:
                inter_abs_cum += weight_change

        # H is undirected; each edge has one shared weight dict.
        H[u_plus][v_plus]['weight'] += weight_change
        H[u_minus][v_minus]['weight'] -= weight_change

        if H[u_minus][v_minus]['weight'] <= 1e-12:
            if Cbar_for_groups is not None:
                edges_removed_cum += 1
            H.remove_edge(u_minus, v_minus)

        b_uv_plus = B[:, edge_to_col[edge_plus]].reshape((n, 1))
        b_uv_minus = B[:, edge_to_col[edge_minus]].reshape((n, 1))

        L = L + weight_change * (b_uv_plus @ b_uv_plus.T - b_uv_minus @ b_uv_minus.T)

        if (i + 1) % T_refresh == 0:
            L_plus_I = L + sp.identity(n, format="csr")
            U, R, X, M = sketch_solve(L_plus_I, q, seed)
        else:
            U, R, X, M = sketch_U_sherman_morrison_two_rank(
                U, R, q, weight_change, b_uv_plus.ravel(), b_uv_minus.ravel()
            )

        
        Z_tilde = X * C
        Z = M * C

        if s is not None:
            surrogate_disparity = s.T @ Z_tilde @ s
            disparity = s.T @ Z @ s
            polarization = s.T @ M @ s
        else:
            surrogate_disparity, _ = top_eigenpair(Z_tilde)
            disparity, _ = top_eigenpair(Z)
            polarization, _ = top_eigenpair(M)

        # if convergence is detected, break
        if np.isclose(surrogate_disparity, surrogate_disparity_prev, rtol=1e-6):
            break
        
        diff_L_fro = np.linalg.norm(L - L0, 'fro') / initial_L0_fro * 100

        records.append({
            'Step': i,
            'Metric': 'Surrogate',
            'Percent Change': (surrogate_disparity - initial_surrogate_disparity) / initial_surrogate_disparity * 100,
        })
        records.append({
            'Step': i,
            'Metric': 'Disparity',
            'Percent Change': (disparity - initial_disparity) / initial_disparity * 100,
        })
        records.append({
            'Step': i,
            'Metric': 'Polarization',
            'Percent Change': (polarization - initial_polarization) / initial_polarization * 100,
        })

        records.append({
            'Step': i,
            'Metric': f'$\\|L_{{t}} - L_{{0}}\\|_F$ / $\\|L_{{0}}\\|_F$',
            'Percent Change': diff_L_fro,
        })

        _append_auxiliary_inner_records(
            records,
            i,
            M,
            Cbar_for_groups,
            cum_abs_weight,
            edges_removed_cum,
            intra_abs_cum,
            inter_abs_cum,
            init_mean_cross,
            init_mean_within,
            initial_edge_mass,
            initial_num_edges,
        )

        progress_bar.set_description(f"Name: {name}")
        progress_bar.update(1)
        progress_bar.refresh()

        surrogate_disparity_prev = surrogate_disparity

    eta_time = time.time() - start_time

    progress_bar.close()

    df = pd.DataFrame(records)
    
    return df, L, X, M, H, eta_time


def _laplacian_edge_weight_delta(n: int, u: int, v: int, delta_w: float) -> sp.csr_matrix:
    """Sparse rank-one update delta_w * (e_u - e_v)(e_u - e_v)^T for the Laplacian."""
    rows = np.array([u, u, v, v], dtype=np.int32)
    cols = np.array([u, v, u, v], dtype=np.int32)
    data = np.array([delta_w, -delta_w, -delta_w, delta_w], dtype=np.float64)
    return sp.csr_matrix((data, (rows, cols)), shape=(n, n))


def algebraic_connectivity_and_fiedler_vector(L: sp.csr_matrix) -> Tuple[float, np.ndarray]:
    """
    Second smallest eigenvalue of L (algebraic connectivity) and a corresponding unit eigenvector.
    For a connected graph this is the Fiedler value / Fiedler vector pair.
    """
    n = L.shape[0]
    L64 = L.astype(np.float64)
    if n <= 1:
        return 0.0, np.zeros(max(n, 1), dtype=np.float64)
    k = min(2, n - 1)
    if k < 2:
        vals, vecs = spla.eigsh(L64, k=1, which="SA")
        return float(vals[0]), vecs[:, 0].astype(np.float64)
    vals, vecs = spla.eigsh(L64, k=2, which="SA")
    lam2 = float(vals[1])
    v = vecs[:, 1].astype(np.float64)
    nv = np.linalg.norm(v)
    if nv > 0:
        v = v / nv
    return lam2, v


def fiedler_maximizing_link_recommendation(
    G: nx.Graph,
    s: Optional[np.ndarray],
    C: np.ndarray,
    name: str,
    T_L: int = 100,
    batch_size: int = 100,
    eta: float = 1,
    seed: int = 0,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, nx.Graph, float]:
    """
    Gradient ascent on edge weights: at each step, move mass from an edge with smallest
    (v_u - v_v)^2 to one with largest, where v is the Fiedler vector of L. This follows
    the partial derivative dλ_2/d w_e = (v_u - v_v)^2 for the algebraic connectivity λ_2.
    """
    nodelist = list(G.nodes())
    B = nx.incidence_matrix(G, nodelist=nodelist, oriented=True).toarray()
    edge_to_col = {}
    for j, e in enumerate(G.edges()):
        u_e, v_e = e
        edge_to_col[(u_e, v_e)] = j
        edge_to_col[(v_e, u_e)] = j
    L = sparse_laplacian(G)
    L0 = L.copy()

    H = G.copy()

    n = len(G.nodes())
    eps_sketch = 0.1
    q = max(int(np.log(n) / eps_sketch**2), 1)
    rng = np.random.default_rng(seed)
    R = rng.choice([-1.0, 1.0], size=(n, q)).astype(np.float64)
    L_plus_I = L + sp.identity(n, format="csr")
    U, R, X, M = sketch_solve(L_plus_I, q, seed)

    T_refresh = max(int(np.sqrt(max(G.number_of_edges(), 1))), 1)

    records = []

    start_time = time.time()

    progress_bar = tqdm(total=T_L, desc="Fiedler gradient ascent")

    lam2_0, _ = algebraic_connectivity_and_fiedler_vector(L)

    if s is not None:
        initial_disparity = s.T @ (M * C) @ s
        initial_polarization = s.T @ M @ s
        initial_surrogate_disparity = s.T @ (X * C) @ s
    else:
        initial_disparity = float(top_eigenpair(M * C)[0])
        initial_polarization = float(top_eigenpair(M)[0])
        initial_surrogate_disparity = float(top_eigenpair(X * C)[0])

    initial_L0_fro = float(sp.linalg.norm(L0, "fro"))
    lam2_den = max(abs(lam2_0), 1e-14)

    for i in range(T_L):
        edge_list = list(H.edges())
        if len(edge_list) < 2:
            break
        k_batch = min(batch_size, len(edge_list))
        edges = random.sample(edge_list, k=k_batch)

        _, v = algebraic_connectivity_and_fiedler_vector(L)

        grad_scores = []
        for e in edges:
            u_e, v_e = e
            grad_scores.append((float(v[int(u_e)] - v[int(v_e)]) ** 2, e))
        grad_scores.sort(key=lambda t: t[0])
        edge_minus = grad_scores[0][1]
        edge_plus = grad_scores[-1][1]

        eta_current = eta / np.sqrt(i + 1)
        weight_change = max(
            0.0,
            min(eta_current, H[edge_plus[0]][edge_plus[1]]["weight"], H[edge_minus[0]][edge_minus[1]]["weight"]),
        )

        H[edge_plus[0]][edge_plus[1]]["weight"] += weight_change
        H[edge_minus[0]][edge_minus[1]]["weight"] -= weight_change

        if H[edge_minus[0]][edge_minus[1]]["weight"] <= 1e-12:
            H.remove_edge(edge_minus[0], edge_minus[1])

        b_uv_plus = B[:, edge_to_col[edge_plus]].reshape((n, 1))
        b_uv_minus = B[:, edge_to_col[edge_minus]].reshape((n, 1))

        u_p, v_p = int(edge_plus[0]), int(edge_plus[1])
        u_m, v_m = int(edge_minus[0]), int(edge_minus[1])
        L = L + _laplacian_edge_weight_delta(n, u_p, v_p, weight_change)
        L = L + _laplacian_edge_weight_delta(n, u_m, v_m, -weight_change)
        L.eliminate_zeros()

        if (i + 1) % T_refresh == 0:
            L_plus_I = L + sp.identity(n, format="csr")
            U, R, X, M = sketch_solve(L_plus_I, q, seed)
        
        else:
            U, R, X, M = sketch_U_sherman_morrison_two_rank(
                U, R, q, weight_change, b_uv_plus.ravel(), b_uv_minus.ravel()
            )

        Z_tilde = X * C
        Z = M * C

        lam2_new, _ = algebraic_connectivity_and_fiedler_vector(L)

        if s is not None:
            surrogate_disparity = s.T @ Z_tilde @ s
            disparity = s.T @ Z @ s
            polarization = s.T @ M @ s
        else:
            surrogate_disparity = float(top_eigenpair(Z_tilde)[0])
            disparity = float(top_eigenpair(Z)[0])
            polarization = float(top_eigenpair(M)[0])

        diff_L_fro = float(sp.linalg.norm(L - L0, "fro")) / initial_L0_fro * 100

        records.append(
            {
                "Step": i,
                "Metric": "Fiedler $\\lambda_2$",
                "Percent Change": (lam2_new - lam2_0) / lam2_den * 100,
            }
        )
        records.append(
            {
                "Step": i,
                "Metric": "Surrogate",
                "Percent Change": (surrogate_disparity - initial_surrogate_disparity)
                / (abs(initial_surrogate_disparity) + 1e-14)
                * 100,
            }
        )
        records.append(
            {
                "Step": i,
                "Metric": "Disparity",
                "Percent Change": (disparity - initial_disparity) / (abs(initial_disparity) + 1e-14) * 100,
            }
        )
        records.append(
            {
                "Step": i,
                "Metric": "Polarization",
                "Percent Change": (polarization - initial_polarization)
                / (abs(initial_polarization) + 1e-14)
                * 100,
            }
        )
        
        progress_bar.set_description(f"Name: {name}")
        progress_bar.update(1)
        progress_bar.refresh()

    eta_time = time.time() - start_time

    progress_bar.close()

    df = pd.DataFrame(records)

    return df, L, X, M, H, eta_time


# --- Generalized reweighting + baselines (experiments 6–8) -----------------

METHOD_LABELS: Dict[str, str] = {
    "leverage": "Oracle (leverage)",
    "fiedler_grad": "Oracle (Fiedler grad)",
    "random": "Random rewiring",
    "max_degree": "Max-degree heuristic",
    "max_betweenness": "Max-betweenness heuristic",
}


def _edge_betweenness_unweighted(H: nx.Graph) -> Dict[Tuple[int, int], float]:
    """Undirected edge betweenness (topology only; ignores weights)."""
    return nx.edge_betweenness_centrality(H, normalized=False)


def _select_edges_for_transfer(
    selection: str,
    edges: List[Tuple],
    H: nx.Graph,
    L: sp.csr_matrix,
    U: np.ndarray,
    C: np.ndarray,
    B: np.ndarray,
    edge_to_col: dict,
    n: int,
    q: int,
    rng: np.random.Generator,
    betweenness_scores: Optional[Dict[Tuple[int, int], float]],
) -> Tuple[Tuple, Tuple]:
    """Pick (edge_plus, edge_minus) from candidate ``edges`` for weight transfer."""
    if len(edges) < 2:
        raise ValueError("Need at least two candidate edges.")

    if selection == "leverage":
        cols = np.fromiter((edge_to_col[e] for e in edges), dtype=np.intp, count=len(edges))
        T = B[:, cols]
        coef = U.T @ T
        norm_r_sq = np.sum(coef * coef, axis=0)
        quad = np.sum(T * (C @ T), axis=0)
        leverage_arr = norm_r_sq * quad
        idx_plus = int(np.argmax(leverage_arr))
        idx_minus = int(np.argmin(leverage_arr))
        return edges[idx_plus], edges[idx_minus]

    if selection == "fiedler_grad":
        _, v = algebraic_connectivity_and_fiedler_vector(L)
        scored = []
        for e in edges:
            u_e, v_e = int(e[0]), int(e[1])
            scored.append((float(v[u_e] - v[v_e]) ** 2, e))
        scored.sort(key=lambda t: t[0])
        return scored[-1][1], scored[0][1]

    if selection == "random":
        perm = rng.permutation(len(edges))
        return edges[int(perm[0])], edges[int(perm[1])]

    if selection == "max_degree":
        scored = []
        for e in edges:
            u_e, v_e = int(e[0]), int(e[1])
            s = H.degree(u_e) + H.degree(v_e)
            scored.append((float(s), e))
        scored.sort(key=lambda t: t[0])
        return scored[-1][1], scored[0][1]

    if selection == "max_betweenness":
        if betweenness_scores is None:
            betweenness_scores = _edge_betweenness_unweighted(H)

        def edge_bc(e: Tuple) -> float:
            u, v = int(e[0]), int(e[1])
            return float(
                max(
                    betweenness_scores.get((u, v), 0.0),
                    betweenness_scores.get((v, u), 0.0),
                )
            )

        scored = [(edge_bc(e), e) for e in edges]
        scored.sort(key=lambda t: t[0])
        return scored[-1][1], scored[0][1]

    raise ValueError(f"Unknown selection rule: {selection}")


def generalized_link_reweighting(
    G: nx.Graph,
    s: Optional[np.ndarray],
    C: np.ndarray,
    name: str,
    T_L: int = 100,
    batch_size: int = 100,
    eta: float = 1.0,
    seed: int = 0,
    selection: str = "leverage",
    track_fiedler: bool = False,
    betweenness_refresh: int = 1,
    Cbar_for_groups: Optional[np.ndarray] = None,
) -> Tuple[pd.DataFrame, sp.csr_matrix, np.ndarray, np.ndarray, nx.Graph, float]:
    """
    Same mass-transfer dynamics as ``link_recommendation`` / ``fiedler_maximizing_link_recommendation``,
    with edge choice controlled by ``selection``:

    - ``leverage``: disparity-oracle (original link recommendation)
    - ``fiedler_grad``: Fiedler-value gradient scores
    - ``random``: uniform random pair from the batch
    - ``max_degree``: move weight toward higher-degree endpoints
    - ``max_betweenness``: move weight toward higher edge-betweenness (topology); refresh cadence via
      ``betweenness_refresh``
    """
    if selection not in METHOD_LABELS:
        raise ValueError(f"selection must be one of {list(METHOD_LABELS)}")

    nodelist = list(G.nodes())
    B = nx.incidence_matrix(G, nodelist=nodelist, oriented=True).toarray()
    edge_to_col = {}
    for j, e in enumerate(G.edges()):
        u_e, v_e = e
        edge_to_col[(u_e, v_e)] = j
        edge_to_col[(v_e, u_e)] = j

    L = sparse_laplacian(G)
    L0 = L.copy()
    H = G.copy()
    n = len(G.nodes())

    eps_sketch = 0.1
    q = max(int(np.log(n) / eps_sketch**2), 1)
    rng = np.random.default_rng(seed)
    R = rng.choice([-1.0, 1.0], size=(n, q)).astype(np.float64)
    L_plus_I = L + sp.identity(n, format="csr")
    U, R, X, M = sketch_solve(L_plus_I, q, seed)

    T_refresh = max(int(np.sqrt(max(G.number_of_edges(), 1))), 1)
    records = []
    start_time = time.time()

    lam2_0 = 0.0
    lam2_den = 1.0
    if track_fiedler:
        lam2_0, _ = algebraic_connectivity_and_fiedler_vector(L)
        lam2_den = max(abs(lam2_0), 1e-14)

    if s is not None:
        initial_disparity = float(s.T @ (M * C) @ s)
        initial_polarization = float(s.T @ M @ s)
        initial_surrogate_disparity = float(s.T @ (X * C) @ s)
    else:
        initial_disparity = float(top_eigenpair(M * C)[0])
        initial_polarization = float(top_eigenpair(M)[0])
        initial_surrogate_disparity = float(top_eigenpair(X * C)[0])

    initial_L0_fro = float(sp.linalg.norm(L0, "fro"))

    if Cbar_for_groups is not None:
        init_mean_cross, init_mean_within = mean_M_cross_within_pairs(M, Cbar_for_groups)
        initial_edge_mass = float(sum(float(H[u][v]["weight"]) for u, v in H.edges()))
        initial_num_edges = int(H.number_of_edges())
    else:
        init_mean_cross = float("nan")
        init_mean_within = float("nan")
        initial_edge_mass = 0.0
        initial_num_edges = 0
    cum_abs_weight = 0.0
    edges_removed_cum = 0
    intra_abs_cum = 0.0
    inter_abs_cum = 0.0

    betweenness_cache: Optional[Dict[Tuple[int, int], float]] = None
    if selection == "max_betweenness" and betweenness_refresh <= 0:
        raise ValueError("betweenness_refresh must be >= 1")

    desc = f"{METHOD_LABELS.get(selection, selection)}"
    progress_bar = tqdm(total=T_L, desc=desc)

    for i in range(T_L):
        edge_list = list(H.edges())
        if len(edge_list) < 2:
            break
        k_batch = min(batch_size, len(edge_list))
        idx = rng.choice(len(edge_list), size=k_batch, replace=False)
        edges = [edge_list[j] for j in idx]

        if selection == "max_betweenness":
            if betweenness_cache is None or (i % betweenness_refresh == 0):
                betweenness_cache = _edge_betweenness_unweighted(H)
        else:
            betweenness_cache = None

        edge_plus, edge_minus = _select_edges_for_transfer(
            selection,
            edges,
            H,
            L,
            U,
            C,
            B,
            edge_to_col,
            n,
            q,
            rng,
            betweenness_cache,
        )

        eta_current = eta / np.sqrt(i + 1)
        weight_change = max(
            0.0,
            min(
                eta_current,
                H[edge_plus[0]][edge_plus[1]]["weight"],
                H[edge_minus[0]][edge_minus[1]]["weight"],
            ),
        )

        if Cbar_for_groups is not None:
            cum_abs_weight += 2.0 * weight_change
            bp = edge_inter_intra_bucket(int(edge_plus[0]), int(edge_plus[1]), Cbar_for_groups)
            bm = edge_inter_intra_bucket(int(edge_minus[0]), int(edge_minus[1]), Cbar_for_groups)
            if bp == "intra":
                intra_abs_cum += weight_change
            else:
                inter_abs_cum += weight_change
            if bm == "intra":
                intra_abs_cum += weight_change
            else:
                inter_abs_cum += weight_change

        H[edge_plus[0]][edge_plus[1]]["weight"] += weight_change
        H[edge_minus[0]][edge_minus[1]]["weight"] -= weight_change

        if H[edge_minus[0]][edge_minus[1]]["weight"] <= 1e-12:
            if Cbar_for_groups is not None:
                edges_removed_cum += 1
            H.remove_edge(edge_minus[0], edge_minus[1])

        b_uv_plus = B[:, edge_to_col[edge_plus]].reshape((n, 1))
        b_uv_minus = B[:, edge_to_col[edge_minus]].reshape((n, 1))

        u_p, v_p = int(edge_plus[0]), int(edge_plus[1])
        u_m, v_m = int(edge_minus[0]), int(edge_minus[1])
        L = L + _laplacian_edge_weight_delta(n, u_p, v_p, weight_change)
        L = L + _laplacian_edge_weight_delta(n, u_m, v_m, -weight_change)
        L.eliminate_zeros()

        if (i + 1) % T_refresh == 0:
            L_plus_I = L + sp.identity(n, format="csr")
            U, R, X, M = sketch_solve(L_plus_I, q, seed)
        else:
            U, R, X, M = sketch_U_sherman_morrison_two_rank(
                U, R, q, weight_change, b_uv_plus.ravel(), b_uv_minus.ravel()
            )
        

        Z_tilde = X * C
        Z = M * C

        lam2_new = 0.0
        if track_fiedler:
            lam2_new, _ = algebraic_connectivity_and_fiedler_vector(L)

        if s is not None:
            surrogate_disparity = float(s.T @ Z_tilde @ s)
            disparity = float(s.T @ Z @ s)
            polarization = float(s.T @ M @ s)
        else:
            surrogate_disparity = float(top_eigenpair(Z_tilde)[0])
            disparity = float(top_eigenpair(Z)[0])
            polarization = float(top_eigenpair(M)[0])

        diff_L_fro = float(sp.linalg.norm(L - L0, "fro")) / initial_L0_fro * 100

        if track_fiedler:
            records.append(
                {
                    "Step": i,
                    "Metric": "Fiedler $\\lambda_2$",
                    "Percent Change": (lam2_new - lam2_0) / lam2_den * 100,
                }
            )

        if track_fiedler:
            records.append(
                {
                    "Step": i,
                    "Metric": "Surrogate",
                    "Percent Change": (surrogate_disparity - initial_surrogate_disparity)
                    / (abs(initial_surrogate_disparity) + 1e-14)
                    * 100,
                }
            )
            records.append(
                {
                    "Step": i,
                    "Metric": "Disparity",
                    "Percent Change": (disparity - initial_disparity)
                    / (abs(initial_disparity) + 1e-14)
                    * 100,
                }
            )
            records.append(
                {
                    "Step": i,
                    "Metric": "Polarization",
                    "Percent Change": (polarization - initial_polarization)
                    / (abs(initial_polarization) + 1e-14)
                    * 100,
                }
            )
        else:
            records.append(
                {
                    "Step": i,
                    "Metric": "Surrogate",
                    "Percent Change": (surrogate_disparity - initial_surrogate_disparity)
                    / (initial_surrogate_disparity + 1e-14)
                    * 100,
                }
            )
            records.append(
                {
                    "Step": i,
                    "Metric": "Disparity",
                    "Percent Change": (disparity - initial_disparity)
                    / (initial_disparity + 1e-14)
                    * 100,
                }
            )
            records.append(
                {
                    "Step": i,
                    "Metric": "Polarization",
                    "Percent Change": (polarization - initial_polarization)
                    / (initial_polarization + 1e-14)
                    * 100,
                }
            )

        records.append(
            {
                "Step": i,
                "Metric": f"$\\|L_{{t}} - L_{{0}}\\|_F$ / $\\|L_{{0}}\\|_F$",
                "Percent Change": diff_L_fro,
            }
        )

        _append_auxiliary_inner_records(
            records,
            i,
            M,
            Cbar_for_groups,
            cum_abs_weight,
            edges_removed_cum,
            intra_abs_cum,
            inter_abs_cum,
            init_mean_cross,
            init_mean_within,
            initial_edge_mass,
            initial_num_edges,
        )

        progress_bar.set_description(
            f"Name: {name}"
        )
        progress_bar.update(1)
        progress_bar.refresh()

    eta_time = time.time() - start_time
    progress_bar.close()

    df = pd.DataFrame(records)
    return df, L, X, M, H, eta_time


def robust_link_recommendation_baseline(
    G: nx.Graph,
    Cbar: np.ndarray,
    rho: float,
    name: str,
    q: int,
    inner_selection: str,
    norm_constraint: str = "spectral_ball",
    K: int = 10,
    T_L: int = 50,
    T_C: int = 20,
    eta_L: float = 1.0,
    eta_C: float = 2.0,
    batch_size: int = 100,
    seed: int = 0,
    betweenness_refresh: int = 1,
) -> Tuple[pd.DataFrame, pd.DataFrame, float]:
    """Like ``robust_link_recommendation`` but inner Laplacian updates use ``generalized_link_reweighting``."""
    L0 = sparse_laplacian(G)
    L0_plus_I = L0 + sp.identity(L0.shape[0], format="csr")
    U0, R0, X0, M0 = sketch_solve(L0_plus_I, q, seed)
    n = G.number_of_nodes()

    initial_surrogate_disparity = float(top_eigenpair(X0 * Cbar)[0])
    initial_disparity = float(top_eigenpair(M0 * Cbar)[0])
    initial_polarization = float(top_eigenpair(M0)[0])

    active_set = [
        (Cbar.copy(), L0, M0, X0, initial_surrogate_disparity, initial_disparity, initial_polarization)
    ]

    records_outer = []
    records_inner = []

    records_outer.append(
        {
            "Name": name,
            "Metric": "Worst Surrogate Disparity",
            "Percent Change": 0.0,
            "Value": initial_surrogate_disparity,
            "k": 0,
            "Norm Constraint": norm_constraint,
            "Rho": rho,
            "Number of Nodes": n,
        }
    )
    records_outer.append(
        {
            "Name": name,
            "Metric": "Worst Disparity",
            "Percent Change": 0.0,
            "Value": initial_disparity,
            "k": 0,
            "Norm Constraint": norm_constraint,
            "Rho": rho,
            "Number of Nodes": n,
        }
    )
    records_outer.append(
        {
            "Name": name,
            "Metric": "Worst Polarization",
            "Percent Change": 0.0,
            "Value": initial_polarization,
            "k": 0,
            "Norm Constraint": norm_constraint,
            "Rho": rho,
            "Number of Nodes": n,
        }
    )
    cbar_fro = float(np.linalg.norm(Cbar, "fro"))
    cbar_spec = float(np.linalg.norm(Cbar, 2))
    records_outer.append(
        {
            "Name": name,
            "Metric": "Frobenius deviation",
            "Percent Change": 0.0,
            "Value": 0.0,
            "k": 0,
            "Norm Constraint": norm_constraint,
            "Rho": rho,
            "Number of Nodes": n,
        }
    )
    records_outer.append(
        {
            "Name": name,
            "Metric": "Spectral deviation",
            "Percent Change": 0.0,
            "Value": 0.0,
            "k": 0,
            "Norm Constraint": norm_constraint,
            "Rho": rho,
            "Number of Nodes": n,
        }
    )

    eta_time = 0.0

    for k in range(K):
        C0, L0, M0, X0, worst_surrogate_disparity_current, worst_disparity_current, worst_polarization_current = (
            active_set[-1]
        )

        df_inner, L, X, M, H, L_eta_time = generalized_link_reweighting(
            G=G.copy(),
            s=None,
            C=C0,
            name=name,
            T_L=T_L,
            batch_size=batch_size,
            eta=eta_L,
            seed=seed,
            selection=inner_selection,
            track_fiedler=False,
            betweenness_refresh=betweenness_refresh,
            Cbar_for_groups=Cbar,
        )
        records_inner.append(df_inner)
        eta_time += L_eta_time

        C_new, C_eta_time = worst_case_C_for_fixed_L(
            L=L,
            X=X,
            Cbar=Cbar,
            rho=rho,
            C_prev=C0,
        )

        worst_surrogate_disparity_new = float(top_eigenpair(X * C_new)[0])
        worst_disparity_new = float(top_eigenpair(M * C_new)[0])
        worst_polarization_new = float(top_eigenpair(M)[0])

        eta_time += C_eta_time

        if worst_surrogate_disparity_new <= worst_surrogate_disparity_current:
            break

        active_set.append(
            (C_new, L, M, X, worst_surrogate_disparity_new, worst_disparity_new, worst_polarization_new)
        )

        records_outer.append(
            {
                "Name": name,
                "Metric": "Worst Surrogate Disparity",
                "Percent Change": (worst_surrogate_disparity_new - initial_surrogate_disparity)
                / initial_surrogate_disparity
                * 100,
                "Value": worst_surrogate_disparity_new,
                "k": k + 1,
                "Norm Constraint": norm_constraint,
                "Rho": rho,
                "Number of Nodes": n,
            }
        )
        records_outer.append(
            {
                "Name": name,
                "Metric": "Worst Disparity",
                "Percent Change": (worst_disparity_new - initial_disparity) / initial_disparity * 100,
                "Value": worst_disparity_new,
                "k": k + 1,
                "Norm Constraint": norm_constraint,
                "Rho": rho,
                "Number of Nodes": n,
            }
        )
        records_outer.append(
            {
                "Name": name,
                "Metric": "Worst Polarization",
                "Percent Change": (worst_polarization_new - initial_polarization) / initial_polarization * 100,
                "Value": worst_polarization_new,
                "k": k + 1,
                "Norm Constraint": norm_constraint,
                "Rho": rho,
                "Number of Nodes": n,
            }
        )
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
                "Norm Constraint": norm_constraint,
                "Rho": rho,
                "Number of Nodes": n,
            }
        )
        records_outer.append(
            {
                "Name": name,
                "Metric": "Spectral deviation",
                "Percent Change": float(100.0 * dev_spec / (cbar_spec + 1e-14)),
                "Value": dev_spec,
                "k": k + 1,
                "Norm Constraint": norm_constraint,
                "Rho": rho,
                "Number of Nodes": n,
            }
        )

    df_outer = pd.DataFrame(records_outer)
    df_inner = pd.concat(records_inner, ignore_index=True)
    return df_outer, df_inner, eta_time


def get_iteration_parameters(n, eps):
    temp = max(1, int(1 / eps**2))
    q = max(int(np.log(n) / eps**2), 1)
    K = temp
    T_L = temp
    T_C = temp

    return T_L, T_C, K, q


_FROBENIUS_METRIC_LABEL = '$\\|L_{t} - L_{0}\\|_F$ / $\\|L_{0}\\|_F$'

_EXPERIMENT_4_AUX_INNER_METRICS = frozenset(
    {
        "Mean M cross-group",
        "Mean M within-group",
        "Edges removed (cumulative)",
        "Intra-group abs weight Δ (cum.)",
        "Inter-group abs weight Δ (cum.)",
    }
)

_EXPERIMENT_4_AUX_INNER_METRICS_ORDER = (
    "Mean M cross-group",
    "Mean M within-group",
    "Edges removed (cumulative)",
    "Intra-group abs weight Δ (cum.)",
    "Inter-group abs weight Δ (cum.)",
)

_EXPERIMENT_4_OUTER_C_DEV_METRICS = frozenset(
    {
        "Frobenius deviation",
        "Spectral deviation",
    }
)


def _experiment_4_forward_fill_meta(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ['Name', 'Rho', 'Norm Constraint', 'Number of Nodes', 'Time (s)', 'Per Step Time (s)']:
        if col in out.columns:
            out[col] = out[col].ffill()
    return out


def _experiment_4_inner_last_run_for_metrics(df: pd.DataFrame, inner_metrics: frozenset) -> pd.DataFrame:
    """Keep only the final outer-iteration inner trajectory per (Name, Rho) for the given metrics."""
    d = df[df["Metric"].isin(inner_metrics)].copy()
    if d.empty:
        return d
    d = d.sort_index()
    pieces = []
    for (_, _), g in d.groupby(["Name", "Rho"], sort=False):
        step = g["Step"].to_numpy()
        if len(step) == 0:
            continue
        run_id = np.cumsum(np.r_[0, (np.diff(step) < 0).astype(int)])
        last_run = int(run_id.max())
        g2 = g.assign(_run_id=run_id)
        pieces.append(g2[g2["_run_id"] == last_run].drop(columns=["_run_id"]))
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()


def _experiment_4_inner_last_run(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the final outer-iteration inner trajectory per (Name, Rho)."""
    return _experiment_4_inner_last_run_for_metrics(
        df,
        frozenset({"Surrogate", "Disparity", "Polarization", _FROBENIUS_METRIC_LABEL}),
    )

def experiment_0_network_statistics(args: argparse.Namespace):
    out_dir = args.out_dir
    datasets = get_datasets(args)

    records = []
    records_rho = []

    rho_values = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])

    for name, group_types in datasets:
        for group_type in group_types:
            G, s, Cbar, labels = load_dataset(name, group_type, return_labels=True)
            n = G.number_of_nodes()
            m = G.number_of_edges()
            d = G.degree()
            d_avg = np.mean(d)
            d_max = np.max(d)

            # fiedler value
            L = sparse_laplacian(G)
            eigenvalues, _ = spla.eigsh(L, k=2, which="SA")
            fiedler_value = eigenvalues[1]

            eps = 0.1
            q = max(int(np.log(n) / eps**2), 1)
            L_plus_I = L + sp.identity(n, format="csr")

            U, R, X, M = sketch_solve(L_plus_I, q, seed=args.seed)
            Z = M * Cbar
            Z_tilde = X * Cbar

            realized_polarization = s.T @ M @ s
            realized_disparity = s.T @ Z @ s
            realized_surrogate = s.T @ Z_tilde @ s

            records.append({
                'Name': name,
                '# Nodes': n,
                '# Edges': m,
                'Avg. Degree': d_avg,
                'Max Degree': d_max,
                '$\\lambda_2$': fiedler_value,
                'Realized Polarization': realized_polarization,
                'Realized Disparity': realized_disparity,
                'Realized Surrogate': realized_surrogate,
            })

            

    df = pd.DataFrame(records)


    n = df['# Nodes'].max()
    lambda_2_range = np.arange(0, 1, 0.01)
    rho_range = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    
    # define a function
    def lb(x, y):
        return (1 - 2 * y)**2 * (1 + x)**2

    X, Y = np.meshgrid(lambda_2_range, rho_range)
    Z = lb(X, Y)

    fig, ax = plt.subplots(figsize=(2 * FIGSIZE, FIGSIZE))
    
    fig.colorbar(ax.contourf(X, Y, Z, cmap='viridis'))
    ax.contour(X, Y, Z, colors='black', linestyles='dashed', linewidths=0.5)

    for name in df['Name'].unique():
        df_name = df[df['Name'] == name]
        ax.axvline(df_name['$\\lambda_2$'].values[0])
        ax.text(df_name['$\\lambda_2$'].values[0], 0.4, name, rotation=90, ha='right', va='center', fontsize=16, color='white')

    ax.annotate('increased\nrisk', xy=(lambda_2_range[-1] - 0.05, 0.05), xytext=(0.5, 0.25), arrowprops=dict(arrowstyle='->'), fontsize=16, color='white', ha='right', va='center')
    ax.set_xlabel('Well-connectedness / Fiedler Value ($\\lambda_2$)')
    ax.set_xticks([])
    ax.set_ylabel('Classifier error probability ($\\rho$)')
    ax.set_title('Ratio Lower Bound')
    sns.despine(fig)
    # fig.tight_layout()
    fig.savefig(f'{out_dir}/experiment_0_network_statistics_lower_bound.pdf', dpi=300, bbox_inches='tight')

    df.to_latex(f'{out_dir}/experiment_0_network_statistics.tex', index=False, float_format='%.2f')
    

def experiment_1_link_recommendation_oracle(args: argparse.Namespace):
    out_dir = args.out_dir
    datasets = get_datasets(args)

    if args.cached_results:
        concat_df = pd.read_csv(f'{out_dir}/experiment_1_link_recommendation_oracle.csv')
    else:
        concat_df = []
        for name, group_types in datasets:
            for group_type in group_types:
                G, s, Cbar = load_dataset(name, group_type)
                T_L, T_C, K, q = get_iteration_parameters(G.number_of_nodes(), args.eps)
                T_C = 1
                K = 1
                eta = np.sqrt(G.number_of_edges() / 2)
                df, L, X, M, H, eta_time = link_recommendation(G, s=None if args.s_type == 'actual' else s, C=Cbar, name=name, T_L=T_L, batch_size=args.batch_size, eta=eta, seed=args.seed)

                df['Name'] = name
                df['Nominal Partition Type'] = group_type
                df['Number of Link Recommendations'] = T_L
                df['Number of Worst Case Solves'] = T_C
                df['Number of Outer Iterations'] = K
                df['Number of Sketch Vectors'] = q
                df['Number of Nodes'] = G.number_of_nodes()
                df['Batch Size'] = args.batch_size
                df['Seed'] = args.seed
                df['Time (s)'] = eta_time
                df['Per Step Time (s)'] = eta_time / T_L
                concat_df.append(df)

        concat_df = pd.concat(concat_df, ignore_index=True)
        concat_df = concat_df[np.isfinite(concat_df['Percent Change'])].copy()

    num_names = concat_df['Name'].nunique()
    num_nominal_partition_types = concat_df['Nominal Partition Type'].nunique()

    fig_a, ax_a = plt.subplots(nrows=1, ncols=(1 + num_names), figsize=(FIGSIZE * (1 + num_names), FIGSIZE), squeeze=False)
    fig_c, ax_c = plt.subplots(nrows=1, ncols=num_names, figsize=(FIGSIZE * num_names, FIGSIZE), squeeze=False)

    alpha = 0.1
    min_percent_change = (1 + alpha) * concat_df['Percent Change'].min()
    max_percent_change = (1 + alpha) * concat_df['Percent Change'].max()

    for i, name in enumerate(concat_df['Name'].unique()):
        df_name = concat_df[concat_df['Name'] == name].copy()
        df_a = df_name[df_name['Step'] == df_name['Step'].max()].reset_index(drop=True)
        sns.barplot(x='Nominal Partition Type', y='Percent Change', hue='Metric', data=df_a, ax=ax_a[0, i], dodge=True, legend=(i == num_names - 1))
        ax_a[0, i].set_title(name)
        ax_a[0, i].set_ylim(min_percent_change, max_percent_change)

        sns.barplot(x='Nominal Partition Type', y='Per Step Time (s)', data=df_name, ax=ax_c[0, i], dodge=True, legend=(i == num_names - 1))
       
        ax_c[0, i].set_title(name)


    # sort concat_df by Number of Nodes
    concat_df = concat_df.sort_values(by='Number of Nodes')

    sns.barplot(x='Name', y='Time (s)', data=concat_df, ax=ax_a[0, -1], legend=True, dodge=True)

    ax_a[0, -1].set_title('Runtime')
    ax_a[0, -1].set_xlabel('Number of Nodes')
    ax_a[0, -1].set_ylabel('Time (s)')
    ax_a[0, -1].set_yscale('log')

    fig_a.suptitle('Link Recommendation Oracle (Step 1)')
    fig_a.tight_layout()
    fig_a.savefig(f'{out_dir}/experiment_1a_link_recommendation_oracle.pdf', dpi=300, bbox_inches='tight')

    fig_c.suptitle('Per Step Time of Link Recommendation Oracle (Step 1)')
    fig_c.tight_layout()
    fig_c.savefig(f'{out_dir}/experiment_1c_link_recommendation_oracle.pdf', dpi=300, bbox_inches='tight')

    concat_df.to_csv(f'{out_dir}/experiment_1_link_recommendation_oracle.csv', index=False)

def experiment_2_link_recommendation_oracle(args: argparse.Namespace):
    out_dir = args.out_dir

    datasets = get_datasets(args)

    if args.cached_results:
        concat_df = pd.read_csv(f'{out_dir}/experiment_2_link_recommendation_oracle.csv')
    else:
        concat_df = []

        p_values = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

        for name, group_types in datasets:
            for group_type in set(group_types) - {'polarization'}:
                print(f"Running {name} with {group_type}")
                G, s, Cbar, labels = load_dataset(name, group_type, return_labels=True)
                T_L, T_C, K, q = get_iteration_parameters(G.number_of_nodes(), args.eps)

                for p in p_values:
                    C = corr_from_labels(labels, rho=p)
                    eta = np.sqrt(G.number_of_edges() / 2)
                    df, L, X, M, H, eta_time = link_recommendation(G, s=s if args.s_type == 'actual' else None, C=C, name=name, T_L=T_L, batch_size=args.batch_size, eta=eta, seed=args.seed)

                    df['Name'] = name
                    df['Nominal Partition Type'] = group_type
                    df['Number of Link Recommendations'] = T_L
                    df['Number of Worst Case Solves'] = T_C
                    df['Number of Outer Iterations'] = K
                    df['Number of Sketch Vectors'] = q
                    df['Number of Nodes'] = G.number_of_nodes()
                    df['Batch Size'] = args.batch_size
                    df['Seed'] = args.seed
                    df['Time (s)'] = eta_time
                    df['Per Step Time (s)'] = eta_time / T_L
                    df['Parameter Value'] = p
                    df['Scenario Type'] = 'Classifier Error ($p$)'

                    concat_df.append(df)

        concat_df = pd.concat(concat_df, ignore_index=True)
        concat_df = concat_df[np.isfinite(concat_df['Percent Change'])].copy()

    num_names = concat_df['Name'].nunique()
    num_parameter_types = concat_df['Scenario Type'].nunique()

    fig_a, ax_a = plt.subplots(nrows=num_parameter_types, ncols=num_names, figsize=(FIGSIZE * num_names, FIGSIZE * num_parameter_types), squeeze=False)

    for i, name in enumerate(concat_df['Name'].unique()):
        for j, scenario_type in enumerate(concat_df['Scenario Type'].unique()):
            df_cell = concat_df[
                (concat_df['Name'] == name)
                & (concat_df['Scenario Type'] == scenario_type)
                & (concat_df['Metric'].isin(['Disparity', 'Polarization']))
            ].copy()
            step_max = concat_df['Step'].max()
            df_cell = df_cell[df_cell['Step'] == step_max]
            df_cell = df_cell.sort_values('Parameter Value')
            sns.lineplot(
                x='Parameter Value',
                y='Percent Change',
                hue='Metric',
                style='Nominal Partition Type',
                data=df_cell,
                ax=ax_a[j, i],
                legend=(i == num_names - 1) and (j == num_parameter_types - 1),
                markers=True,
                marker='x',
                markersize=5,
            )
            if i == 0:
                ax_a[j, i].set_ylabel('Percent Change')
            if j == 0:
                ax_a[j, i].set_title(name)

            ax_a[j, i].set_xlabel(scenario_type)   

    fig_a.suptitle('Impact of Platform Intervention as a Function of Uncertainty')
    fig_a.tight_layout()
    fig_a.savefig(f'{out_dir}/experiment_2a_link_recommendation_oracle.pdf', dpi=300, bbox_inches='tight')

    concat_df.to_csv(f'{out_dir}/experiment_2_link_recommendation_oracle.csv', index=False)

def experiment_3_worst_case_C_oracle(args: argparse.Namespace):
    out_dir = args.out_dir

    csv_path = f'{out_dir}/experiment_3_{args.s_type}_worst_case_C_oracle.csv'

    if args.cached_results:
        concat_df = pd.read_csv(csv_path)
    else:
        datasets = get_datasets(args)

        rho_values = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

        records = []

        for name, group_types in datasets:
            for group_type in set(group_types) - {'polarization'}:
                print(f"Running {name} with {group_type}")
                G, s, Cbar = load_dataset(name, group_type)
                T_L, T_C, K, q = get_iteration_parameters(G.number_of_nodes(), args.eps)
                L = sparse_laplacian(G)
                L_plus_I = L + sp.identity(L.shape[0], format="csr")
                U, R, X, M = sketch_solve(L_plus_I, q, seed=args.seed)

                for rho in rho_values:
                    print(f'Classifier error rho = {rho}')

                    if args.s_type == 'actual':
                        C_wc, eta_time = worst_case_C_for_fixed_L(L, X, Cbar, rho, s=s, C_prev=None)

                        worst_case_C_disparity = s.T @ (M * C_wc) @ s
                        worst_case_C_surrogate = s.T @ (X * C_wc) @ s

                        worst_case_Cbar_disparity = s.T @ (M * Cbar) @ s
                        worst_case_Cbar_surrogate = s.T @ (X * Cbar) @ s

                    elif args.s_type == 'adversarial':
                        C_wc, eta_time = worst_case_C_for_fixed_L(L, X, Cbar, rho, s=None, C_prev=None)

                        worst_case_C_disparity, _ = top_eigenpair(M * C_wc)
                        worst_case_C_surrogate, _ = top_eigenpair(X * C_wc)

                        worst_case_Cbar_disparity, _ = top_eigenpair(M * Cbar)
                        worst_case_Cbar_surrogate, _ = top_eigenpair(X * Cbar)

                    disparity_change = (worst_case_C_disparity - worst_case_Cbar_disparity) / (worst_case_Cbar_disparity + 1e-10) * 100
                    surrogate_change = (worst_case_C_surrogate - worst_case_Cbar_surrogate) / (worst_case_Cbar_surrogate + 1e-10) * 100

                    n = G.number_of_nodes()

                    records.append({
                        'Name': name,
                        'Nominal Partition Type': group_type,
                        'Rho': rho,
                        'Metric': 'Disparity',
                        'Value': disparity_change,
                        'Number of Nodes': n,
                    })
                    records.append({
                        'Name': name,
                        'Nominal Partition Type': group_type,
                        'Rho': rho,
                        'Metric': 'Surrogate',
                        'Value': surrogate_change,
                        'Number of Nodes': n,
                    })

                    records.append({
                        'Name': name,
                        'Nominal Partition Type': group_type,
                        'Rho': rho,
                        'Metric': 'Time (s)',
                        'Value': eta_time,
                        'Number of Nodes': n,
                    })
                    records.append({
                        'Name': name,
                        'Nominal Partition Type': group_type,
                        'Rho': rho,
                        'Metric': 'Per Step Time (s)',
                        'Value': eta_time / T_C,
                        'Number of Nodes': n,
                    })

        concat_df = pd.DataFrame(records)

    num_names = concat_df['Name'].nunique()

    fig_a, ax_a = plt.subplots(nrows=1, ncols=(1 + num_names), figsize=(FIGSIZE * (1 + num_names), FIGSIZE), squeeze=False)

    min_y_lim = np.inf
    max_y_lim = 0

    for i, name in enumerate(concat_df['Name'].unique()):
        df_name = concat_df[(concat_df['Name'] == name) & (concat_df['Metric'].isin(['Disparity', 'Surrogate']))].copy()
        sns.lineplot(x='Rho', y='Value', hue='Metric', style='Nominal Partition Type', markers=True, marker='x', markersize=5, data=df_name, ax=ax_a[0, i], legend=(i == num_names - 1))
        ax_a[0, i].set_title(name)
        ax_a[0, i].set_xlabel('$\\rho$')
        ax_a[0, i].set_ylabel('Percent Change')

        min_y_lim = min(min_y_lim, df_name['Value'].min())
        max_y_lim = max(max_y_lim, df_name['Value'].max())

    for i in range(num_names):
        ax_a[0, i].set_ylim(min_y_lim, max_y_lim)


    df_time = concat_df[concat_df['Metric'].isin(['Time (s)'])].copy()
    sns.barplot(x='Name', y='Value', hue='Rho', data=df_time, ax=ax_a[0, -1], legend=True, dodge=True)
    ax_a[0, -1].set_title('Runtime')
    ax_a[0, -1].set_xlabel('')
    ax_a[0, -1].set_ylabel('Time (s)')
    ax_a[0, -1].set_yscale('log')

    fig_a.suptitle(f'Correlation matrix update oracle (Step 2)')
    fig_a.tight_layout()
    fig_a.savefig(f'{out_dir}/experiment_3_{args.s_type}_worst_case_C_oracle.pdf', dpi=300, bbox_inches='tight')
    concat_df.to_csv(csv_path, index=False)


def experiment_4_robust_link_recommendation_oracle(args: argparse.Namespace):
    out_dir = args.out_dir

    datasets = get_datasets(args)

    rho_values = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

    if args.cached_results:
        concat_df = pd.read_csv(f'{out_dir}/experiment_4_robust_link_recommendation_oracle.csv')
        df_opinions = pd.read_csv(f'{out_dir}/experiment_4_final_opinion_change.csv')
    else:
        concat_df = []
        df_opinions = []

        for name, group_types in datasets:
            for group_type in group_types:
                print(f"Running {name} with {group_type}")
                G, s, Cbar = load_dataset(name, group_type)
                T_L, T_C, K, q = get_iteration_parameters(G.number_of_nodes(), args.eps)

                K = 3

                for rho in rho_values:
                    print(f'Classifier error rho = {rho}')

                    df_inner, df_outer, eta_time, M0, M = robust_link_recommendation(G=G.copy(), Cbar=Cbar, rho=rho, name=name, q=q, T_L=T_L, T_C=T_C, K=K, batch_size=args.batch_size, eta_L=1.0, eta_C=2*rho, seed=args.seed)
                    
                    df_inner['Time (s)'] = eta_time
                    df_outer['Time (s)'] = eta_time
                    df_inner['Per Step Time (s)'] = eta_time / (K * T_C * T_L)
                    df_outer['Per Step Time (s)'] = eta_time / (K * T_C * T_L)
                    concat_df.append(df_inner)
                    concat_df.append(df_outer)

                    z_init = M0 @ s
                    z_final = M @ s
                    z_change = z_final - z_init
                    z_change = z_change / np.linalg.norm(z_change)
                    for i in range(z_change.shape[0]):
                        df_opinions.append({
                            'Name': name,
                            'Rho': rho,
                            'Value': z_change[i],
                            'User ID': i,
                        })

        concat_df = pd.concat(concat_df, ignore_index=True)
        concat_df = concat_df[np.isfinite(concat_df['Percent Change'])].copy()
        df_opinions = pd.DataFrame(df_opinions)

    plot_df = _experiment_4_forward_fill_meta(concat_df)
    df_inner_last = _experiment_4_inner_last_run(plot_df)
    if df_inner_last.empty:
        concat_df.to_csv(f'{out_dir}/experiment_4_robust_link_recommendation_oracle.csv', index=False)
        return

    num_names = plot_df['Name'].nunique()
    rho_values_sorted = sorted(plot_df['Rho'].dropna().unique())
    num_rhos = len(rho_values_sorted)

    fig_a, ax_a = plt.subplots(nrows=1, ncols=(1 + num_names), figsize=(FIGSIZE * (1 + num_names), FIGSIZE), squeeze=False)
    fig_b, ax_b = plt.subplots(
        nrows=num_rhos,
        ncols=num_names,
        figsize=(FIGSIZE * num_names, FIGSIZE * num_rhos),
        squeeze=False,
        sharey=True,
    )
    fig_c, ax_c = plt.subplots(nrows=1, ncols=num_names, figsize=(FIGSIZE * num_names, FIGSIZE), squeeze=False, sharey=True)

    alpha = 0.1
    min_percent_change = (1 + alpha) * df_inner_last['Percent Change'].min()
    max_percent_change = (1 + alpha) * df_inner_last['Percent Change'].max()

    idx_max = df_inner_last.groupby(['Name', 'Rho'])['Step'].transform('max')
    df_bar = df_inner_last[df_inner_last['Step'] == idx_max].copy()

    for i, name in enumerate(plot_df['Name'].unique()):
        df_name = plot_df[plot_df['Name'] == name].copy()
        sns.barplot(x='Rho', y='Percent Change', hue='Metric', data=df_bar[(df_bar['Name'] == name) & (df_bar['Metric'].isin(['Surrogate', 'Disparity']))], ax=ax_a[0, i], dodge=True, legend=(i == num_names - 1))
        ax_a[0, i].set_title(name)
        # ax_a[0, i].set_ylim(min_percent_change, max_percent_change)

        df_time = df_name.drop_duplicates(subset=['Rho'], keep='first')
        sns.barplot(x='Rho', y='Per Step Time (s)', data=df_time, ax=ax_c[0, i], dodge=True, legend=(i == num_names - 1))
        ax_c[0, i].set_title(name)

        for j, rho in enumerate(rho_values_sorted):
            df_b = df_inner_last[(df_inner_last['Name'] == name) & (df_inner_last['Rho'] == rho) & (df_inner_last['Metric'].isin(['Surrogate', 'Disparity']))].copy()
            sns.lineplot(
                x='Step',
                y='Percent Change',
                hue='Metric',
                data=df_b,
                ax=ax_b[j, i],
                legend=(i == num_names - 1) and (j == num_rhos - 1),
            )
            ax_b[j, i].set_ylim(min_percent_change, max_percent_change)
            if i == 0:
                ax_b[j, i].set_ylabel(f'$\\rho = {rho}$')
            if j == num_rhos - 1:
                ax_b[j, i].set_xlabel('Step')
            if j == 0:
                ax_b[j, i].set_title(name)

    plot_df_sorted = plot_df.sort_values(by='Number of Nodes')
    df_rt = plot_df_sorted.drop_duplicates(subset=['Name', 'Rho'], keep='first')
    sns.barplot(x='Name', y='Time (s)', hue='Rho', data=df_rt, ax=ax_a[0, -1], legend=True, dodge=True)
    ax_a[0, -1].set_title('Runtime')
    ax_a[0, -1].set_xlabel('')
    ax_a[0, -1].set_ylabel('Time (s)')
    ax_a[0, -1].set_yscale('log')

    fig_a.suptitle('Robust Link Recommendation Oracle')
    fig_a.tight_layout()
    fig_a.savefig(f'{out_dir}/experiment_4a_robust_link_recommendation_oracle.pdf', dpi=300, bbox_inches='tight')

    fig_b.suptitle('Robust Link Recommendation Oracle')
    fig_b.tight_layout()
    fig_b.savefig(f'{out_dir}/experiment_4b_robust_link_recommendation_oracle.pdf', dpi=300, bbox_inches='tight')

    fig_c.suptitle('Per Step Time of Robust Link Recommendation Oracle')
    fig_c.tight_layout()
    fig_c.savefig(f'{out_dir}/experiment_4c_robust_link_recommendation_oracle.pdf', dpi=300, bbox_inches='tight')

    # --- Auxiliary inner metrics: last inner step of final outer iteration (bar by rho) ---
    df_aux_last = _experiment_4_inner_last_run_for_metrics(plot_df, _EXPERIMENT_4_AUX_INNER_METRICS)
    if not df_aux_last.empty:
        idx_max_aux = df_aux_last.groupby(["Name", "Rho"])["Step"].transform("max")
        df_aux_bar = df_aux_last[df_aux_last["Step"] == idx_max_aux].copy()
        name_list = list(plot_df["Name"].unique())
        fig_d, ax_d = plt.subplots(
            nrows=1,
            ncols=num_names,
            figsize=(FIGSIZE * num_names, FIGSIZE),
            squeeze=False,
        )
        for i, name in enumerate(name_list):
            sub = df_aux_bar[df_aux_bar["Name"] == name].copy()
            sns.barplot(
                x="Rho",
                y="Percent Change",
                hue="Metric",
                hue_order=list(_EXPERIMENT_4_AUX_INNER_METRICS_ORDER),
                data=sub,
                ax=ax_d[0, i],
                dodge=True,
                legend=(i == num_names - 1),
            )
            ax_d[0, i].set_title(name)
            ax_d[0, i].set_xlabel(r"$\rho$")
            if i == 0:
                ax_d[0, i].set_ylabel("Percent change")
        

        fig_d.tight_layout()
        
        # move legend to the outside of the plot
        ax_d[0, i].legend(loc='upper left', bbox_to_anchor=(1, 1))

        fig_d.savefig(
            f"{out_dir}/experiment_4d_robust_link_recommendation_oracle.pdf",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig_d)

    # --- Outer-loop $C$ deviation from $\bar C$: final outer iteration (bar by rho) ---
    df_c_dev = plot_df[plot_df["Metric"].isin(_EXPERIMENT_4_OUTER_C_DEV_METRICS)].dropna(subset=["k"])
    if not df_c_dev.empty:
        idx_max_k = df_c_dev.groupby(["Name", "Rho", "Metric"])["k"].transform("max")
        df_c_bar = df_c_dev[df_c_dev["k"] == idx_max_k].copy()
        metric_order = ["Frobenius deviation", "Spectral deviation"]
        name_list = list(plot_df["Name"].unique())
        fig_e, ax_e = plt.subplots(
            nrows=1,
            ncols=num_names,
            figsize=(FIGSIZE * num_names, FIGSIZE),
            squeeze=False,
        )
        for i, name in enumerate(name_list):
            sub = df_c_bar[df_c_bar["Name"] == name].copy()
            sns.barplot(
                x="Rho",
                y="Value",
                hue="Metric",
                hue_order=metric_order,
                data=sub,
                ax=ax_e[0, i],
                dodge=True,
                legend=(i == num_names - 1),
            )
            ax_e[0, i].set_title(name)
            ax_e[0, i].set_xlabel(r"$\rho$")
            if i == 0:
                ax_e[0, i].set_ylabel(r"$\|C_{{new}} - \bar C\|$")
            else:
                ax_e[0, i].set_ylabel('')
        
        # move legend to the outside of the plot
        if i == num_names - 1:
            ax_e[0, i].legend(loc='upper left', bbox_to_anchor=(1, 1))
        fig_e.tight_layout()
        fig_e.savefig(
            f"{out_dir}/experiment_4e_robust_link_recommendation_oracle.pdf",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig_e)

    # lineplot of final opinion change by node id for each name hue by rho
    fig_f, ax_f = plt.subplots(nrows=1, ncols=(1 + num_names), figsize=(FIGSIZE * (1 + num_names), FIGSIZE), squeeze=False)
    for i, name in enumerate(name_list):
        df_name = df_opinions[df_opinions["Name"] == name].copy()
        sns.lineplot(x="User ID", y="Value", hue="Rho", data=df_name, ax=ax_f[0, i], markers=True, marker='o', markersize=5)
        ax_f[0, i].set_title(name)
        ax_f[0, i].set_xlabel("User ID")
        if i == 0:
            ax_f[0, i].set_ylabel("Normalized Opinion Change")
        else:
            ax_f[0, i].set_ylabel('')
        ax_f[0, i].set_ylim(-1, 1)

    sns.barplot(x="Rho", y="Value", hue="Name", dodge=True, data=df_opinions, ax=ax_f[0, -1], palette='viridis')
    ax_f[0, -1].set_xlabel("$\\rho$")
    ax_f[0, -1].set_ylabel("")
    ax_f[0, -1].set_title('average')

    fig_f.tight_layout()
    fig_f.savefig(f'{out_dir}/experiment_4f_robust_link_recommendation_oracle.pdf', dpi=300, bbox_inches='tight')


    df_opinions.to_csv(f'{out_dir}/experiment_4_final_opinion_change.csv', index=False)
    concat_df.to_csv(f'{out_dir}/experiment_4_robust_link_recommendation_oracle.csv', index=False)



def experiment_5_fiedler_gradient_ascent(args: argparse.Namespace):
    """
    Same layout as experiment 1, but edge updates follow gradient ascent on algebraic
    connectivity (Fiedler value λ_2 of L).
    """
    out_dir = args.out_dir

    datasets = get_datasets(args)

    if args.cached_results:
        concat_df = pd.read_csv(f"{out_dir}/experiment_5_fiedler_gradient_ascent.csv")
    else:
        concat_df = []

        for name, group_types in datasets:
            for group_type in set(group_types) - {'polarization'}:
                G, s, Cbar = load_dataset(name, group_type)
                T_L, T_C, K, q = get_iteration_parameters(G.number_of_nodes(), args.eps)
                T_C = 1
                K = 1
                df, L, X, M, H, eta_time = fiedler_maximizing_link_recommendation(
                    G,
                    s,
                    Cbar,
                    name,
                    T_L=T_L,
                    batch_size=args.batch_size,
                    eta=np.sqrt(G.number_of_edges() / 2),
                    seed=args.seed,
                )

                df["Name"] = name
                df["Nominal Partition Type"] = group_type
                df["Number of Link Recommendations"] = T_L
                df["Number of Sketch Vectors"] = q
                df["Number of Nodes"] = G.number_of_nodes()
                df["Batch Size"] = args.batch_size
                df["Seed"] = args.seed
                df["Time (s)"] = eta_time
                df["Per Step Time (s)"] = eta_time / T_L
                concat_df.append(df)

        concat_df = pd.concat(concat_df, ignore_index=True)
        concat_df = concat_df[np.isfinite(concat_df["Percent Change"])].copy()

    num_names = concat_df["Name"].nunique()
    partition_types_for_lines = sorted(
        set(concat_df["Nominal Partition Type"].unique()) - {"polarization"},
        key=lambda x: (str(x) != "spectral", str(x)),
    )
    n_line_rows = max(1, len(partition_types_for_lines))

    fig_a, ax_a = plt.subplots(
        nrows=1,
        ncols=(1 + num_names),
        figsize=(FIGSIZE * (1 + num_names), FIGSIZE),
        squeeze=False,
    )
    fig_b, ax_b = plt.subplots(
        nrows=n_line_rows,
        ncols=num_names,
        figsize=(FIGSIZE * num_names, FIGSIZE * n_line_rows),
        squeeze=False,
        sharey=True,
    )
    fig_c, ax_c = plt.subplots(
        nrows=1,
        ncols=num_names,
        figsize=(FIGSIZE * num_names, FIGSIZE),
        squeeze=False,
        sharey=True,
    )

    alpha = 0.1
    min_percent_change = (1 + alpha) * concat_df["Percent Change"].min()
    max_percent_change = (1 + alpha) * concat_df["Percent Change"].max()

    for i, name in enumerate(concat_df["Name"].unique()):
        df_name = concat_df[concat_df["Name"] == name].copy()
        df_a = df_name[df_name["Step"] == df_name["Step"].max()].reset_index(drop=True)
        sns.barplot(
            x="Nominal Partition Type",
            y="Percent Change",
            hue="Metric",
            data=df_a,
            ax=ax_a[0, i],
            dodge=True,
            legend=(i == num_names - 1),
        )
        ax_a[0, i].set_title(name)
        ax_a[0, i].set_ylim(min_percent_change, max_percent_change)

        sns.barplot(
            x="Nominal Partition Type",
            y="Per Step Time (s)",
            data=df_name,
            ax=ax_c[0, i],
            dodge=True,
            legend=(i == num_names - 1),
        )

        ax_c[0, i].set_title(name)

        for j, nominal_partition_type in enumerate(partition_types_for_lines):
            df_b = df_name[df_name["Nominal Partition Type"] == nominal_partition_type].copy()
            sns.lineplot(
                x="Step",
                y="Percent Change",
                hue="Metric",
                data=df_b,
                ax=ax_b[j, i],
                legend=(i == num_names - 1) and (j == n_line_rows - 1),
            )
            ax_b[j, i].set_ylim(min_percent_change, max_percent_change)
            if i == 0:
                ax_b[j, i].set_ylabel(nominal_partition_type)
            if j == 0:
                ax_b[j, i].set_xlabel("Step")
            if i == 0:
                ax_b[j, i].set_title(name)

        ax_b[0, i].set_title(name)

    concat_df = concat_df.sort_values(by="Number of Nodes")

    sns.scatterplot(
        x="Number of Nodes",
        y="Time (s)",
        data=concat_df,
        ax=ax_a[0, -1],
        style='Name',
        markersize=15,
    )

    alpha = 0.05

    min_number_of_nodes = int((1 - alpha) * concat_df["Number of Nodes"].min())
    max_number_of_nodes = int((1 + alpha) * concat_df["Number of Nodes"].max())

    ax_a[0, -1].set_title("Runtime of Robust Link Recommendation")
    ax_a[0, -1].set_xlabel("Number of Nodes")
    ax_a[0, -1].set_ylabel("Runtime (s)")
    ax_a[0, -1].set_xlim(min_number_of_nodes, max_number_of_nodes)

    ax_a[0, -1].set_yscale("log")

    fig_a.suptitle("Robust Link Recommendation (Unconstrained U)")
    fig_a.tight_layout()
    fig_a.savefig(f"{out_dir}/experiment_5a_fiedler_gradient_ascent.pdf", dpi=300, bbox_inches="tight")

    fig_b.suptitle("Robust Link Recommendation (Unconstrained U)")
    fig_b.tight_layout()
    fig_b.savefig(f"{out_dir}/experiment_5b_fiedler_gradient_ascent.pdf", dpi=300, bbox_inches="tight")

    fig_c.suptitle("Per step time — Robust Link Recommendation (Unconstrained U)")
    fig_c.tight_layout()
    fig_c.savefig(f"{out_dir}/experiment_5c_fiedler_gradient_ascent.pdf", dpi=300, bbox_inches="tight")

    concat_df.to_csv(f"{out_dir}/experiment_5_fiedler_gradient_ascent.csv", index=False)


def experiment_6_link_recommendation_baselines(args: argparse.Namespace) -> None:
    """Disparity-focused reweighting baselines: random vs max-degree."""
    out_dir = args.out_dir
    datasets = get_datasets(args)
    selections = ["random", "max_degree"]

    if args.cached_results:
        df_all = pd.read_csv(f"{out_dir}/experiment_6_link_recommendation_baselines.csv")
    else:
        concat_df: List[pd.DataFrame] = []

        for name, group_types in datasets:
            for group_type in set(group_types) - {'polarization'}:
                G, s, Cbar, labels = load_dataset(name, group_type, return_labels=True)
                T_L, _, _, q = get_iteration_parameters(G.number_of_nodes(), args.eps)
                for sel in selections:
                    print(f"Exp6 {name} {group_type} {METHOD_LABELS[sel]}")
                    df, _, _, _, _, eta_time = generalized_link_reweighting(
                        G,
                        s,
                        Cbar,
                        name,
                        T_L=T_L,
                        batch_size=args.batch_size,
                        eta=np.sqrt(G.number_of_edges() / 2),
                        seed=args.seed,
                        selection=sel,
                        track_fiedler=False,
                        betweenness_refresh=args.betweenness_refresh,
                    )
                    df["Name"] = name
                    df["Nominal Partition Type"] = group_type
                    df["Method"] = METHOD_LABELS[sel]
                    df["Number of Link Recommendations"] = T_L
                    df["Number of Sketch Vectors"] = q
                    df["Number of Nodes"] = G.number_of_nodes()
                    df["Batch Size"] = args.batch_size
                    df["Seed"] = args.seed
                    df["Time (s)"] = eta_time
                    df["Per Step Time (s)"] = eta_time / T_L
                    concat_df.append(df)

        df_all = pd.concat(concat_df, ignore_index=True)
        df_all = df_all[np.isfinite(df_all["Percent Change"])].copy()
    allowed_methods = {METHOD_LABELS[sel] for sel in selections}
    df_all = df_all[df_all["Method"].isin(allowed_methods)].copy()
    df_all.to_csv(f"{out_dir}/experiment_6_link_recommendation_baselines.csv", index=False)

    step_max = df_all["Step"].max()
    df_last = df_all[df_all["Step"] == step_max].copy()
    metrics = ["Surrogate", "Disparity", "Polarization"]
    df_bar = df_last[df_last["Metric"].isin(metrics)].copy()

    # format as the other catplot
    df_bar["Method"] = df_bar["Method"].replace({
        "Random rewiring": "Random",
        "Max-degree heuristic": "Degree",
    })

    num_names = df_bar["Name"].nunique()

    g = sns.catplot(
        data=df_bar,
        x="Method",
        y="Percent Change",
        hue="Metric",
        col="Name",
        kind="bar",
        sharey=True,
        legend=True,
        height=5.5,
        aspect=num_names / 2.5,
    )

    g.set_titles(template="{col_name}")
    g.tight_layout()
    g.savefig(f"{out_dir}/experiment_6a_link_recommendation_baselines.pdf", dpi=300, bbox_inches="tight")
    plt.close("all")


    partition_types_for_lines = sorted(
        set(df_all["Nominal Partition Type"].unique()) - {"polarization"},
        key=lambda x: (str(x) != "spectral", str(x)),
    )
    df_line = df_all[
        (df_all["Metric"] == "Disparity") & (df_all["Nominal Partition Type"].isin(partition_types_for_lines))
    ].copy()
    if len(df_line) > 0:
        n_names = df_all["Name"].nunique()
        n_rows = max(1, len(partition_types_for_lines))
        fig, axes = plt.subplots(
            n_rows,
            n_names,
            figsize=(FIGSIZE * n_names, FIGSIZE * n_rows),
            squeeze=False,
            sharey=True,
        )
        for i, name in enumerate(df_all["Name"].unique()):
            for j, npt in enumerate(partition_types_for_lines):
                cell = df_line[(df_line["Name"] == name) & (df_line["Nominal Partition Type"] == npt)]
                sns.lineplot(
                    data=cell,
                    x="Step",
                    y="Percent Change",
                    hue="Method",
                    ax=axes[j, i],
                    legend=(i == n_names - 1) and (j == n_rows - 1),
                )
                if j == 0:
                    axes[j, i].set_title(name)
                if i == 0:
                    axes[j, i].set_ylabel(npt)
        fig.suptitle("Disparity percent change vs step (baselines)")
        fig.tight_layout()
        fig.savefig(f"{out_dir}/experiment_6b_link_recommendation_baselines.pdf", dpi=300, bbox_inches="tight")
        plt.close(fig)


def experiment_7_fiedler_baselines(args: argparse.Namespace) -> None:
    """Robust link recommendation baselines (general correlation matrix): random vs max-degree."""
    out_dir = args.out_dir
    datasets = get_datasets(args)
    selections = ["random", "max_degree"]

    if args.cached_results:
        df_all = pd.read_csv(f"{out_dir}/experiment_7_fiedler_baselines.csv")
    else:
        concat_df: List[pd.DataFrame] = []

        for name, group_types in datasets:
            for group_type in group_types:
                G, s, Cbar = load_dataset(name, group_type)
                T_L, _, _, q = get_iteration_parameters(G.number_of_nodes(), args.eps)
                for sel in selections:
                    print(f"Exp7 {name} {group_type} {METHOD_LABELS[sel]}")
                    df, _, _, _, _, eta_time = generalized_link_reweighting(
                        G,
                        s,
                        Cbar,
                        name,
                        T_L=T_L,
                        batch_size=args.batch_size,
                        eta=np.sqrt(G.number_of_edges() / 2),
                        seed=args.seed,
                        selection=sel,
                        track_fiedler=True,
                        betweenness_refresh=args.betweenness_refresh,
                    )
                    df["Name"] = name
                    df["Nominal Partition Type"] = group_type
                    df["Method"] = METHOD_LABELS[sel]
                    df["Number of Link Recommendations"] = T_L
                    df["Number of Sketch Vectors"] = q
                    df["Number of Nodes"] = G.number_of_nodes()
                    df["Batch Size"] = args.batch_size
                    df["Seed"] = args.seed
                    df["Time (s)"] = eta_time
                    df["Per Step Time (s)"] = eta_time / T_L
                    concat_df.append(df)

        df_all = pd.concat(concat_df, ignore_index=True)
        df_all = df_all[np.isfinite(df_all["Percent Change"])].copy()
    allowed_methods = {METHOD_LABELS[sel] for sel in selections}
    df_all = df_all[df_all["Method"].isin(allowed_methods)].copy()
    df_all.to_csv(f"{out_dir}/experiment_7_fiedler_baselines.csv", index=False)

    step_max = df_all["Step"].max()
    df_last = df_all[df_all["Step"] == step_max].copy()
    # metrics = [r"Fiedler $\lambda_2$", "Surrogate", "Disparity", "Polarization"]
    metrics = ["Surrogate", "Disparity", "Polarization"]
    df_bar = df_last[df_last["Metric"].isin(metrics)].copy()

    # rename methods to shorter names
    df_bar["Method"] = df_bar["Method"].replace({
        "Random rewiring": "Random",
        "Max-degree heuristic": "Degree",
    })

    num_names = df_bar["Name"].nunique()

    g = sns.catplot(
        data=df_bar,
        x="Method",
        y="Percent Change",
        hue="Metric",
        col="Name",
        kind="bar",
        sharey=True,
        legend=True,
        height=5.5,
        aspect=num_names / 2.5,
    )

    g.set_titles(template="{col_name}")
    g.tight_layout()
    g.savefig(f"{out_dir}/experiment_7a_fiedler_baselines.pdf", dpi=300, bbox_inches="tight")
    plt.close("all")

    df_line = df_all[
        (df_all["Metric"] == r"Fiedler $\lambda_2$")
        & (df_all["Nominal Partition Type"].isin(set(df_all["Nominal Partition Type"].unique()) - {"polarization"}))
    ].copy()
    if len(df_line) > 0:
        partition_types_for_lines = sorted(
            set(df_all["Nominal Partition Type"].unique()) - {"polarization"},
            key=lambda x: (str(x) != "spectral", str(x)),
        )
        n_names = df_all["Name"].nunique()
        n_rows = max(1, len(partition_types_for_lines))
        fig, axes = plt.subplots(
            n_rows,
            n_names,
            figsize=(FIGSIZE * n_names, FIGSIZE * n_rows),
            squeeze=False,
            sharey=True,
        )
        for i, name in enumerate(df_all["Name"].unique()):
            for j, npt in enumerate(partition_types_for_lines):
                cell = df_line[(df_line["Name"] == name) & (df_line["Nominal Partition Type"] == npt)]
                sns.lineplot(
                    data=cell,
                    x="Step",
                    y="Percent Change",
                    hue="Method",
                    ax=axes[j, i],
                    legend=(i == n_names - 1) and (j == n_rows - 1),
                )
                if j == 0:
                    axes[j, i].set_title(name)
                if i == 0:
                    axes[j, i].set_ylabel(npt)
        fig.suptitle(r"Fiedler $\lambda_2$ percent change vs step (baselines)")
        fig.tight_layout()
        fig.savefig(f"{out_dir}/experiment_7b_fiedler_baselines.pdf", dpi=300, bbox_inches="tight")
        plt.close(fig)


def experiment_8_robust_link_recommendation_baselines(args: argparse.Namespace) -> None:
    """Robust outer loop with different inner edge-selection heuristics."""
    out_dir = args.out_dir
    datasets = get_datasets(args)
    rho_values = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    inner_selections = ["random", "max_degree"]
    norm_constraints = ["classifier_error"]

    if args.cached_results:
        df_all = pd.read_csv(f"{out_dir}/experiment_8_robust_link_recommendation_baselines.csv")
    else:
        concat_df: List[pd.DataFrame] = []

        for name, group_types in datasets:
            for group_type in group_types:
                G, s, Cbar = load_dataset(name, group_type)
                T_L, T_C, K, q = get_iteration_parameters(G.number_of_nodes(), args.eps)
                for rho in rho_values:
                    for norm_constraint in norm_constraints:
                        for sel in inner_selections:
                            print(f"Exp8 {name} {group_type} rho={rho} {METHOD_LABELS[sel]}")
                            df_outer, df_inner, eta_time = robust_link_recommendation_baseline(
                                G=G,
                                Cbar=Cbar,
                                rho=rho,
                                name=name,
                                q=q,
                                inner_selection=sel,
                                norm_constraint=norm_constraint,
                                T_L=T_L,
                                T_C=T_C,
                                K=K,
                                batch_size=args.batch_size,
                                eta_L=np.sqrt(G.number_of_edges() / 2),
                                eta_C=2 * rho,
                                seed=args.seed,
                                betweenness_refresh=args.betweenness_refresh,
                            )
                            label = METHOD_LABELS[sel]
                            df_inner["Method"] = label
                            df_inner["Rho"] = rho
                            df_outer["Method"] = label
                            df_outer["Rho"] = rho
                            df_inner["Name"] = name
                            df_inner["Nominal Partition Type"] = group_type
                            df_outer["Name"] = name
                            df_outer["Nominal Partition Type"] = group_type
                            df_inner["Time (s)"] = eta_time
                            df_outer["Time (s)"] = eta_time
                            steps = max(K * T_L * T_C, 1)
                            df_inner["Per Step Time (s)"] = eta_time / steps
                            df_outer["Per Step Time (s)"] = eta_time / steps
                            concat_df.append(df_inner)
                            concat_df.append(df_outer)

        df_all = pd.concat(concat_df, ignore_index=True)
    allowed_methods = {METHOD_LABELS[sel] for sel in inner_selections}
    df_all = df_all[df_all["Method"].isin(allowed_methods)].copy()
    df_all.to_csv(f"{out_dir}/experiment_8_robust_link_recommendation_baselines.csv", index=False)

    df_outer = df_all[df_all["Metric"].str.startswith("Worst", na=False)].copy()
    if len(df_outer) == 0:
        return

    # remove worst polarization metric
    df_outer = df_outer[df_outer["Metric"] != "Worst Polarization"]

    df_outer = df_outer[df_outer["Nominal Partition Type"].isin(set(df_outer["Nominal Partition Type"].unique()) - {"polarization"})].copy()

    g = sns.relplot(
        data=df_outer,
        x="Rho",
        y="Percent Change",
        hue="Method",
        style="Metric",
        col="Name",
        kind="line",
        markers=True,
        dashes=False,
        facet_kws={"sharey": True},
    )
    g.set_titles(template="{col_name}")
    g.figure.suptitle("Comparison with baselines")
    g.tight_layout()
    g.savefig(f"{out_dir}/experiment_8a_robust_link_recommendation_baselines.pdf", dpi=300, bbox_inches="tight")
    plt.close("all")

def experiment_9_predictive_model(args: argparse.Namespace) -> None:
    out_dir = args.out_dir
    datasets = [('polblogs', ['label'])]  
    train_data_fractions = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

    if args.cached_results:
        df_all = pd.read_csv(f"{out_dir}/experiment_9_predictive_model.csv")
    else:
        concat_df: List[pd.DataFrame] = []
        rng_local = np.random.default_rng(args.seed)

        for name, group_types in datasets:
            for group_type in group_types:
                G, s, Cbar, labels = load_dataset(name, group_type, return_labels=True)
                embeddings_filename = f"data/{name}/embeddings.npy"
                labels = (labels > 0).astype(float)
                labels = labels.reshape(-1, 1)
                
                if args.embedding_type == 'precomputed' and os.path.exists(embeddings_filename):
                    embeddings = np.load(embeddings_filename)
                elif args.embedding_type == 'node2vec':
                    print("training embeddings for ", name, group_type)
                    from karateclub import Node2Vec
                    model = Node2Vec(dimensions=64, walk_length=30, workers=4)
                    model.fit(G)
                    embeddings = model.get_embedding()
                    np.save(embeddings_filename, embeddings)
                elif args.embedding_type == 'gaussian':
                    embeddings = np.random.normal(0, 1, size=(len(G.nodes()), 32), random_state=args.seed)
                elif args.embedding_type == 'network_structure':                    
                    degree = np.array([G.degree(i) for i in G.nodes()]).reshape(-1, 1)
                    betweeness = nx.betweenness_centrality(G)
                    clustering = nx.clustering(G)
                    betweeness_scores = np.array([betweeness[i] for i in G.nodes()]).reshape(-1, 1)
                    clustering_coefficients = np.array([clustering[i] for i in G.nodes()]).reshape(-1, 1)
                    embeddings = np.concatenate([degree, betweeness_scores, clustering_coefficients], axis=1)
                else:
                    raise ValueError(f"Invalid embedding type: {args.embedding_type}")
        
                n_labels = len(labels)
                for train_data_fraction in train_data_fractions:
                    labels = labels.flatten()

                    X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(
                        embeddings,
                        labels,
                        np.arange(n_labels),
                        test_size=1 - train_data_fraction,
                        stratify=labels,
                        random_state=args.seed,
                    )

                    n_cv_splits = min(5, int(np.bincount(y_train.astype(int)).min()))

                    # train logistic regression model with k-fold cross-validation
                    model = LogisticRegression(max_iter=1000)
                    model.fit(X_train, y_train)
                    cv = StratifiedKFold(n_splits=n_cv_splits, shuffle=True, random_state=args.seed)
                    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
                    accuracy = float(cv_scores.mean())
                    accuracy_std = float(cv_scores.std())
                    print(f"CV Accuracy ({n_cv_splits}-fold) for {name} {group_type} {train_data_fraction}: {accuracy} ± {accuracy_std}")

                    y_pred = model.predict(X_test)

                    test_accuracy = float(np.mean(y_pred == y_test))


                    labels_predictions = np.zeros(n_labels)
                    labels_predictions[test_indices] = 2 * y_pred - 1
                    labels_predictions[train_indices] = 2 * y_train - 1

                    Cbar_pred = np.outer(labels_predictions, labels_predictions)

                    rho = (1 - train_data_fraction) * (1 - (accuracy - accuracy_std))

                    print(f"Test Accuracy for {name} {group_type} {train_data_fraction}: {test_accuracy}, rho={rho}")

                    T_L, T_C, K, q = get_iteration_parameters(G.number_of_nodes(), args.eps)
                    
                    K = 3

                    df_inner, df_outer, eta_time, _, _ = robust_link_recommendation(
                        G=G.copy(),
                        Cbar=Cbar_pred,
                        rho=rho,
                        name=name,
                        q=q,
                        K=K,
                        T_L=T_L,
                        T_C=T_C,
                        eta_L=np.sqrt(G.number_of_edges() / 2),
                        eta_C=2 * rho,
                        batch_size=args.batch_size,
                        seed=args.seed,
                    )

                    steps = max(K * T_C * T_L, 1)
                    for df in (df_outer, df_inner):
                        df["Name"] = name
                        df["Nominal Partition Type"] = group_type
                        df["Accuracy"] = test_accuracy
                        df["Rho"] = rho
                        df["Fraction of Training Data"] = train_data_fraction * 100
                        df["Number of Link Recommendations"] = T_L
                        df["Number of Sketch Vectors"] = q
                        df["Number of Nodes"] = G.number_of_nodes()
                        df["Batch Size"] = args.batch_size
                        df["Seed"] = args.seed
                        df["Time (s)"] = eta_time
                        df["Per Step Time (s)"] = eta_time / steps
                        df["Method"] = "Predictive Model"

                    concat_df.append(df_outer)
                    concat_df.append(df_inner)

        df_all = pd.concat(concat_df, ignore_index=True) if concat_df else pd.DataFrame()

    if df_all.empty:
        return

    df_all.to_csv(f"{out_dir}/experiment_9_predictive_model.csv", index=False)

    plot_df = df_all.copy()
    frac_values_sorted = sorted(plot_df["Fraction of Training Data"].dropna().unique())
    num_fracs = len(frac_values_sorted)
    name_list = list(plot_df["Name"].dropna().unique())
    num_names = len(name_list)
    if num_names == 0:
        return

    df_outer = plot_df[plot_df["Metric"].str.startswith("Worst", na=False)].dropna(subset=["k"]).copy()
    if df_outer.empty:
        return
    df_outer = df_outer[df_outer["Metric"].isin(["Worst Surrogate Disparity", "Worst Disparity"])]
    if df_outer.empty:
        return

    idx_max = df_outer.groupby(["Name", "Fraction of Training Data", "Metric"])["k"].transform("max")
    df_bar = df_outer[df_outer["k"] == idx_max].copy()

    alpha = 0.1
    min_percent_change = (1 + alpha) * df_outer["Percent Change"].min()
    max_percent_change = (1 + alpha) * df_outer["Percent Change"].max()

    fig_a, ax_a = plt.subplots(
        nrows=1,
        ncols=(2 + num_names),
        figsize=(FIGSIZE * (2 + num_names), FIGSIZE),
        squeeze=False,
    )
    fig_b, ax_b = plt.subplots(
        nrows=num_fracs,
        ncols=num_names,
        figsize=(FIGSIZE * num_names, FIGSIZE * num_fracs),
        squeeze=False,
        sharey=True,
    )
    fig_c, ax_c = plt.subplots(
        nrows=1,
        ncols=num_names,
        figsize=(FIGSIZE * num_names, FIGSIZE),
        squeeze=False,
        sharey=True,
    )

    for i, name in enumerate(name_list):
        sub_bar = df_bar[df_bar["Name"] == name].copy()
        sns.barplot(
            x="Fraction of Training Data",
            y="Percent Change",
            hue="Metric",
            data=sub_bar,
            ax=ax_a[0, i],
            dodge=True,
            legend=(i == num_names - 1),
        )
        ax_a[0, i].set_title(name)
        ax_a[0, i].set_xlabel("Fraction of training data")
        # ax_a[0, i].set_ylim(min_percent_change, max_percent_change)

        df_time = plot_df[plot_df["Name"] == name].drop_duplicates(
            subset=["Fraction of Training Data"],
            keep="first",
        )
        sns.barplot(
            x="Fraction of Training Data",
            y="Per Step Time (s)",
            data=df_time,
            ax=ax_c[0, i],
            dodge=True,
            legend=False,
        )
        ax_c[0, i].set_title(name)
        ax_c[0, i].set_xlabel("Fraction of training data")

        for j, frac in enumerate(frac_values_sorted):
            df_line = df_outer[
                (df_outer["Name"] == name)
                & (df_outer["Fraction of Training Data"] == frac)
            ].copy()
            sns.lineplot(
                x="k",
                y="Percent Change",
                hue="Metric",
                data=df_line,
                ax=ax_b[j, i],
                legend=(i == num_names - 1) and (j == num_fracs - 1),
            )
            ax_b[j, i].set_ylim(min_percent_change, max_percent_change)
            if i == 0:
                ax_b[j, i].set_ylabel(f"train frac = {frac:.1f}")
            if j == num_fracs - 1:
                ax_b[j, i].set_xlabel("Outer iteration (k)")
            if j == 0:
                ax_b[j, i].set_title(name)

    sns.scatterplot(x='Fraction of Training Data', y='Rho', hue='Name', data=plot_df, ax=ax_a[0, -2], legend=True)
    ax_a[0, -2].set_xlabel("Fraction of training data")
    ax_a[0, -2].set_ylabel("$\\hat \\rho$")
    ax_a[0, -2].set_ylim(0.0, 0.5)

    sns.scatterplot(x='Fraction of Training Data', y='Accuracy', hue='Name', data=plot_df, ax=ax_a[0, -1], legend=True)
    ax_a[0, -1].set_xlabel("Fraction of training data")
    ax_a[0, -1].set_ylabel("Test Accuracy")
    ax_a[0, -1].set_ylim(0.9, 1.0)

    plot_df_sorted = plot_df.sort_values(by="Number of Nodes")
    df_rt = plot_df_sorted.drop_duplicates(subset=["Name", "Fraction of Training Data"], keep="first")
    
    fig_a.tight_layout()
    fig_a.savefig(f"{out_dir}/experiment_9a_predictive_model.pdf", dpi=300, bbox_inches="tight")

    fig_b.suptitle("Predictive-model robust recommendation")
    fig_b.tight_layout()
    fig_b.savefig(f"{out_dir}/experiment_9b_predictive_model.pdf", dpi=300, bbox_inches="tight")

    fig_c.suptitle("Per-step time of predictive-model robust recommendation")
    fig_c.tight_layout()
    fig_c.savefig(f"{out_dir}/experiment_9c_predictive_model.pdf", dpi=300, bbox_inches="tight")

    plt.close("all")

def main(args: argparse.Namespace) -> None:
    experiment_dict = dict([
        (0, experiment_0_network_statistics),
        (1, experiment_1_link_recommendation_oracle),
        (2, experiment_2_link_recommendation_oracle),
        (3, experiment_3_worst_case_C_oracle),
        (4, experiment_4_robust_link_recommendation_oracle),
        (5, experiment_5_fiedler_gradient_ascent),
        (6, experiment_6_link_recommendation_baselines),
        (7, experiment_7_fiedler_baselines),
        (8, experiment_8_robust_link_recommendation_baselines),
        (9, experiment_9_predictive_model)
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

