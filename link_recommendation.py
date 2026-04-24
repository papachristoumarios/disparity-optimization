from __future__ import annotations

import argparse
import random
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

sns.set_theme(style="whitegrid")
sns.set_palette("deep")

FIGSIZE = 4

def get_datasets(args: argparse.Namespace):
    if args.size == 'tiny':
        datasets = [('reddit', ['random'])]
    elif args.size == 'small':
        datasets = [('reddit', ['random']), ('twitter', ['random']), ('polblogs', ['random'])]
    elif args.size == 'all':
        datasets = [('twitter', ['spectral', 'label', 'polarization', 'random']), ('polblogs', ['spectral', 'label', 'polarization', 'random']), ('reddit', ['spectral', 'label', 'polarization', 'random'])]
    else:
        raise ValueError(f"Invalid size: {args.size}")

    return datasets
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=str, default='figures')
    parser.add_argument("--rho", type=float, default=1.0)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument('--eta', type=float, default=1.0)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--eps', type=float, default=0.1)
    parser.add_argument('--T', type=int, default=100)
    parser.add_argument('--K', type=int, default=10)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--b', type=int, default=100)
    parser.add_argument('--size', default='all', choices=['all', 'small', 'tiny'])
    parser.add_argument(
        '--betweenness-refresh',
        type=int,
        default=1,
        help='Recompute edge betweenness every k steps (1 = every step; larger speeds up max-betweenness baseline).',
    )
    return parser.parse_args()


def robust_link_recommendation(G: nx.Graph, Cbar: np.ndarray, rho: float, name: str, q: int, norm_constraint: str = 'spectral_ball', K: int = 10, T_L: int = 50, T_C: int = 20, eta_L: float = 1.0, eta_C: float = 2.0, batch_size: int = 100, seed: int = 0) -> None:
    L0 = sparse_laplacian(G)
    U0, X0 = sketch_solve_X(L0, q, rng)
    M0 = X0 @ X0
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
        'Norm Constraint': norm_constraint,
        'Rho': rho,
        'Number of Nodes': n,
    })
    records_outer.append({
        'Name': name,
        'Metric': 'Worst Disparity',
        'Percent Change': 0.0,
        'Value': initial_disparity,
        'k' : 0,
        'Norm Constraint': norm_constraint,
        'Rho': rho,
        'Number of Nodes': n,
    })
    records_outer.append({
        'Name': name,
        'Metric': 'Worst Polarization',
        'Percent Change': 0.0,
        'Value': initial_polarization,
        'k' : 0,
        'Norm Constraint': norm_constraint,
        'Rho': rho,
        'Number of Nodes': n,
    })

    eta_time = 0

    for k in range(K):
        C0, L0, M0, X0, worst_surrogate_disparity_current, worst_disparity_current, worst_polarization_current = active_set[-1]

        df_inner, L, X, M, H, L_eta_time = link_recommendation(G=G, C=C0, s=None, name=name, T_L=T_L, batch_size=batch_size, eta=eta_L, seed=seed)
        records_inner.append(df_inner)
        eta_time += L_eta_time

        
        C_new, C_eta_time = worst_case_C_for_fixed_L(L=L, X=X, Cbar=Cbar, rho=rho, norm_constraint=norm_constraint, T_C=T_C, C0=C0, step0=eta_C)

        worst_surrogate_disparity_new = float(top_eigenpair(X * C_new)[0])
        worst_disparity_new = float(top_eigenpair(M * C_new)[0])
        worst_polarization_new = float(top_eigenpair(M)[0])

        eta_time += C_eta_time

        if worst_surrogate_disparity_new <= worst_surrogate_disparity_current:
            break

        active_set.append((C_new, L, M, X, worst_surrogate_disparity_new, worst_disparity_new, worst_polarization_new))

        records_outer.append({
            'Name': name,
            'Metric': 'Worst Surrogate Disparity',
            'Percent Change': (worst_surrogate_disparity_new - initial_surrogate_disparity) / initial_surrogate_disparity * 100,
            'Value': worst_surrogate_disparity_new,
            'k' : k + 1,
            'Norm Constraint': norm_constraint,
            'Rho': rho,
            'Number of Nodes': n,
        })
        records_outer.append({
            'Name': name,
            'Metric': 'Worst Disparity',
            'Percent Change': (worst_disparity_new - initial_disparity) / initial_disparity * 100,
            'Value': worst_disparity_new,
            'k' : k + 1,
            'Norm Constraint': norm_constraint,
            'Rho': rho,
            'Number of Nodes': n,
        })
        records_outer.append({
            'Name': name,
            'Metric': 'Worst Polarization',
            'Percent Change': (worst_polarization_new - initial_polarization) / initial_polarization * 100,
            'Value': worst_polarization_new,
            'k' : k + 1,
            'Norm Constraint': norm_constraint,
            'Rho': rho,
            'Number of Nodes': n,
        })

    df_outer = pd.DataFrame(records_outer)
    df_inner = pd.concat(records_inner, ignore_index=True)
    return df_outer, df_inner, eta_time


def link_recommendation(G: nx.Graph, s: np.ndarray, C: np.ndarray, name: str, T_L: int = 100, batch_size: int = 100, eta: float = 1, seed: int = 0) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, nx.Graph]:
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
    R = rng.choice([-1.0, 1.0], size=(n, q)).astype(np.float64)
    L_plus_I = L + sp.identity(n, format="csr")
    U = sketch_solve(L_plus_I, R)
    X = (U @ U.T) / q
    M = X @ X

    T_refresh = int(np.sqrt(G.number_of_edges()))

    records = []

    start_time = time.time()

    progress_bar = tqdm(total=T_L, desc="Link Recommendations")

    if s:   
        initial_disparity = s.T @ (M * C) @ s
        initial_polarization = s.T @ M @ s
        initial_surrogate_disparity = s.T @ (X * C) @ s
    else:
        initial_disparity = float(top_eigenpair(M * C)[0])
        initial_polarization = float(top_eigenpair(M)[0])
        initial_surrogate_disparity = float(top_eigenpair(X * C)[0])

    initial_L0_fro = np.linalg.norm(L0, 'fro')

    # use tqdm to show progress
    for i in range(T_L):
        # pick one edge
        edges = random.sample(list(H.edges()), k=batch_size)

        cols = np.fromiter((edge_to_col[e] for e in edges), dtype=np.intp, count=len(edges))
        T = B[:, cols]
        coef = U.T @ T
        norm_r_sq = np.sum(coef * coef, axis=0)
        quad = np.sum(T * (C @ T), axis=0)
        leverage_arr = norm_r_sq * quad
        idx_plus = int(np.argmax(leverage_arr))
        idx_minus = int(np.argmin(leverage_arr))
        edge_plus = edges[idx_plus]
        edge_minus = edges[idx_minus]
        u_plus, v_plus = edge_plus
        u_minus, v_minus = edge_minus

        eta_current = eta / np.sqrt(i + 1)
        weight_change = max(0.0, min(eta_current, H[u_plus][v_plus]['weight'], H[u_minus][v_minus]['weight']))


        # H is undirected; each edge has one shared weight dict.
        H[u_plus][v_plus]['weight'] += weight_change
        H[u_minus][v_minus]['weight'] -= weight_change

        if H[u_minus][v_minus]['weight'] <= 1e-12:
            H.remove_edge(u_minus, v_minus)

        b_uv_plus = B[:, edge_to_col[edge_plus]].reshape((n, 1))
        b_uv_minus = B[:, edge_to_col[edge_minus]].reshape((n, 1))

        L = L + weight_change * (b_uv_plus @ b_uv_plus.T - b_uv_minus @ b_uv_minus.T)

        if (i + 1) % T_refresh == 0:
            L_plus_I = L + sp.identity(n, format="csr")
            U = sketch_solve(L_plus_I, R)
        else:
            U = sketch_U_sherman_morrison_two_rank(
                U, q, weight_change, b_uv_plus.ravel(), b_uv_minus.ravel()
            )

        X = (U @ U.T) / q
        M = X.T @ X # simplify with U 
        
        Z_tilde = X * C
        Z = M * C

        if s:
            surrogate_disparity = s.T @ Z_tilde @ s
            disparity = s.T @ Z @ s
            polarization = s.T @ M @ s
        else:
            surrogate_disparity = float(top_eigenpair(Z_tilde)[0])
            disparity = float(top_eigenpair(Z)[0])
            polarization = float(top_eigenpair(M)[0])
        
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

        progress_bar.set_description(f"Name: {name}, S: {surrogate_disparity:.2g}, D: {disparity:.2g}, P: {polarization:.2g}")
        progress_bar.update(1)
        progress_bar.refresh()

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
    U = sketch_solve(L_plus_I, R)
    X = (U @ U.T) / q
    M = X @ X

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
            U = sketch_solve(L_plus_I, R)
        else:
            U = sketch_U_sherman_morrison_two_rank(
                U, q, weight_change, b_uv_plus.ravel(), b_uv_minus.ravel()
            )

        X = (U @ U.T) / q
        M = X @ X

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
        # records.append(
        #     {
        #         "Step": i,
        #         "Metric": f"$\\|L_{{t}} - L_{{0}}\\|_F$ / $\\|L_{{0}}\\|_F$",
        #         "Percent Change": diff_L_fro,
        #     }
        # )

        progress_bar.set_description(
            f"Name: {name}, λ2: {lam2_new:.4g}, S: {surrogate_disparity:.2g}, D: {disparity:.2g}, P: {polarization:.2g}"
        )
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
    U = sketch_solve(L_plus_I, R)
    X = (U @ U.T) / q
    M = X @ X

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
            U = sketch_solve(L_plus_I, R)
        else:
            U = sketch_U_sherman_morrison_two_rank(
                U, q, weight_change, b_uv_plus.ravel(), b_uv_minus.ravel()
            )

        X = (U @ U.T) / q
        M = X @ X
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

        lam2_s = f"{lam2_new:.4g}" if track_fiedler else ""
        progress_bar.set_description(
            f"{desc} | {name} λ2:{lam2_s} S:{surrogate_disparity:.2g} D:{disparity:.2g} P:{polarization:.2g}"
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
    U0, X0 = sketch_solve_X(L0, q, rng)
    M0 = X0 @ X0
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

    eta_time = 0.0

    for k in range(K):
        C0, L0, M0, X0, worst_surrogate_disparity_current, worst_disparity_current, worst_polarization_current = (
            active_set[-1]
        )

        df_inner, L, X, M, H, L_eta_time = generalized_link_reweighting(
            G=G,
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
        )
        records_inner.append(df_inner)
        eta_time += L_eta_time

        C_new, C_eta_time = worst_case_C_for_fixed_L(
            L=L,
            X=X,
            Cbar=Cbar,
            rho=rho,
            norm_constraint=norm_constraint,
            T_C=T_C,
            C0=C0,
            step0=eta_C,
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

def experiment_1_link_recommendation_oracle(args: argparse.Namespace):
    out_dir = args.out_dir

    datasets = get_datasets(args)

    concat_df = []

    for name, group_types in datasets:
        for group_type in group_types:
            G, s, Cbar = load_dataset(name, group_type)
            T_L, T_C, K, q = get_iteration_parameters(G.number_of_nodes(), args.eps)
            T_C = 1
            K = 1
            df, L, X, M, H, eta_time = link_recommendation(G, s, Cbar, name, T_L=T_L, batch_size=args.batch_size, eta=args.eta, seed=args.seed)

            df['Name'] = name
            df['Nominal Partition Type'] = group_type
            df['Number of Link Recommendations'] = T_L
            df['Number of Worst Case Solves'] = T_C
            df['Number of Outer Iterations'] = K
            df['Number of Sketch Vectors'] = q
            df['Number of Nodes'] = G.number_of_nodes()
            df['Batch Size'] = args.batch_size
            df['Learning Rate'] = args.eta
            df['Seed'] = args.seed
            df['Time (s)'] = eta_time
            df['Per Step Time (s)'] = eta_time / T_L
            concat_df.append(df)

    concat_df = pd.concat(concat_df, ignore_index=True)
    concat_df = concat_df[np.isfinite(concat_df['Percent Change'])].copy()

    num_names = concat_df['Name'].nunique()
    num_nominal_partition_types = concat_df['Nominal Partition Type'].nunique()

    fig_a, ax_a = plt.subplots(nrows=1, ncols=(1 + num_names), figsize=(FIGSIZE * (1 + num_names), FIGSIZE), squeeze=False)
    fig_b, ax_b = plt.subplots(nrows=num_nominal_partition_types - 1, ncols=num_names, figsize=(FIGSIZE * num_names, FIGSIZE * (num_nominal_partition_types - 1)), squeeze=False, sharey=True)
    fig_c, ax_c = plt.subplots(nrows=1, ncols=num_names, figsize=(FIGSIZE * num_names, FIGSIZE), squeeze=False, sharey=True)

    alpha = 0.1
    min_percent_change = (1 + alpha) * concat_df['Percent Change'].min()
    max_percent_change = (1 + alpha) * concat_df['Percent Change'].max()

    for i, name in enumerate(concat_df['Name'].unique()):
        df_name = concat_df[concat_df['Name'] == name].copy()
        df_a = df_name[df_name['Step'] == df_name['Step'].max()].reset_index(drop=True)
        sns.barplot(x='Nominal Partition Type', y='Percent Change', hue='Metric', data=df_a, ax=ax_a[0, i], dodge=True, palette="deep", legend=(i == num_names - 1))
        ax_a[0, i].set_title(name)
        ax_a[0, i].set_ylim(min_percent_change, max_percent_change)


        sns.barplot(x='Nominal Partition Type', y='Per Step Time (s)', data=df_name, ax=ax_c[0, i], dodge=True, legend=(i == num_names - 1))
       
        ax_c[0, i].set_title(name)

        for j, nominal_partition_type in enumerate(set(concat_df['Nominal Partition Type'].unique()) - {'polarization'}):
            df_b = df_name[df_name['Nominal Partition Type'] == nominal_partition_type].copy()
            sns.lineplot(x='Step', y='Percent Change', hue='Metric', data=df_b, ax=ax_b[j, i], legend=(i == num_names - 1) and (j == num_nominal_partition_types - 2))
            ax_b[j, i].set_ylim(min_percent_change, max_percent_change)
            if i == 0:
                ax_b[j, i].set_ylabel(nominal_partition_type)
            if j == 0:
                ax_b[j, i].set_xlabel('Step')
            if i == 0:
                ax_b[j, i].set_title(name)

        ax_b[0, i].set_title(name)

    # sort concat_df by Number of Nodes
    concat_df = concat_df.sort_values(by='Number of Nodes')

    sns.lineplot(x='Number of Nodes', y='Time (s)', data=concat_df, ax=ax_a[0, -1], markers=True, marker='x', markersize=5)

    alpha = 0.05

    min_number_of_nodes = int((1 - alpha) * concat_df['Number of Nodes'].min())
    max_number_of_nodes = int((1 + alpha) * concat_df['Number of Nodes'].max())

    n_range = np.arange(min_number_of_nodes, max_number_of_nodes, 50)
    nominal_runtime = n_range * np.log(n_range) / args.eps**4
    ax_a[0, -1].plot(n_range, nominal_runtime, label='Upper bound $O(n \\log n / \\epsilon^4)$', color='red', linestyle='--')
    ax_a[0, -1].legend(loc='upper right')
    ax_a[0, -1].set_title('Runtime of Link Recommendation Oracle')
    ax_a[0, -1].set_xlabel('Number of Nodes')
    ax_a[0, -1].set_ylabel('Runtime (s)')
    ax_a[0, -1].set_xlim(min_number_of_nodes, max_number_of_nodes)

    ax_a[0, -1].set_yscale('log')

    fig_a.suptitle('Link Recommendation Oracle')
    fig_a.tight_layout()
    fig_a.savefig(f'{out_dir}/experiment_1a_link_recommendation_oracle.pdf', dpi=300, bbox_inches='tight')

    fig_b.suptitle('Link Recommendation Oracle')
    fig_b.tight_layout()
    fig_b.savefig(f'{out_dir}/experiment_1b_link_recommendation_oracle.pdf', dpi=300, bbox_inches='tight')

    fig_c.suptitle('Per Step Time of Link Recommendation Oracle')
    fig_c.tight_layout()
    fig_c.savefig(f'{out_dir}/experiment_1c_link_recommendation_oracle.pdf', dpi=300, bbox_inches='tight')

    concat_df.to_csv(f'{out_dir}/experiment_1_link_recommendation_oracle.csv', index=False)

def experiment_2_link_recommendation_oracle(args: argparse.Namespace):
    out_dir = args.out_dir

    datasets = get_datasets(args)
    
    concat_df = []

    p_values = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    eps_values = np.array([0.1, 0.5, 1, 5, 10])

    for name, group_types in datasets:
        for group_type in group_types:
            print(f"Running {name} with {group_type}")
            G, s, Cbar = load_dataset(name, group_type)
            T_L, T_C, K, q = get_iteration_parameters(G.number_of_nodes(), args.eps)

            for p in p_values:
                C = generate_correlation_matrix_scenario(Cbar, mode='classifier_error', p=p)
                df, L, X, M, H, eta_time = link_recommendation(G, s, C, name, T_L=T_L, batch_size=args.batch_size, eta=1.0, seed=args.seed)

                df['Name'] = name
                df['Nominal Partition Type'] = group_type
                df['Number of Link Recommendations'] = T_L
                df['Number of Worst Case Solves'] = T_C
                df['Number of Outer Iterations'] = K
                df['Number of Sketch Vectors'] = q
                df['Number of Nodes'] = G.number_of_nodes()
                df['Batch Size'] = args.batch_size
                df['Learning Rate'] = args.eta
                df['Seed'] = args.seed
                df['Time (s)'] = eta_time
                df['Per Step Time (s)'] = eta_time / T_L
                df['Parameter Value'] = p
                df['Scenario Type'] = 'Classifier Error ($p$)'
            
                concat_df.append(df)

            for eps in eps_values:
                C = generate_correlation_matrix_scenario(Cbar, mode='differential_privacy', epsilon=eps)
                df, L, X, M, H, eta_time = link_recommendation(G, s, C, name, T_L=T_L, batch_size=args.batch_size, eta=1.0, seed=args.seed)

                df['Name'] = name
                df['Nominal Partition Type'] = group_type
                df['Number of Link Recommendations'] = T_L
                df['Number of Worst Case Solves'] = T_C
                df['Number of Outer Iterations'] = K
                df['Number of Sketch Vectors'] = q
                df['Number of Nodes'] = G.number_of_nodes()
                df['Batch Size'] = args.batch_size
                df['Learning Rate'] = args.eta
                df['Seed'] = args.seed
                df['Time (s)'] = eta_time
                df['Per Step Time (s)'] = eta_time / T_L
                df['Parameter Value'] = eps
                df['Scenario Type'] = 'Privacy Budget ($\\epsilon$)'

                concat_df.append(df)

    concat_df = pd.concat(concat_df, ignore_index=True)
    concat_df = concat_df[np.isfinite(concat_df['Percent Change'])].copy()

    num_names = concat_df['Name'].nunique()
    num_parameter_types = concat_df['Scenario Type'].nunique()

    fig_a, ax_a = plt.subplots(nrows=num_parameter_types, ncols=num_names, figsize=(FIGSIZE * num_names, FIGSIZE * num_parameter_types), squeeze=False, sharey=True)

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

            if scenario_type == 'Privacy Budget ($\\epsilon$)':
                ax_a[j, i].set_xscale('log')

    fig_a.suptitle('Impact of Platform Intervention as a Function of Uncertainty')
    fig_a.tight_layout()
    fig_a.savefig(f'{out_dir}/experiment_2a_link_recommendation_oracle.pdf', dpi=300, bbox_inches='tight')

    concat_df.to_csv(f'{out_dir}/experiment_2_link_recommendation_oracle.csv', index=False)

def experiment_3_worst_case_C_oracle(args: argparse.Namespace, s_type='actual'):
    out_dir = args.out_dir

    datasets = get_datasets(args)

    concat_df = []

    rho_values = np.array([0.1, 0.2, 0.4, 0.4, 0.5])
    norm_constraints = ['spectral_ball']

    rng = np.random.default_rng(0)

    records = []

    for name, group_types in datasets:
        for group_type in group_types:
            print(f"Running {name} with {group_type}")
            G, s, Cbar = load_dataset(name, group_type)
            T_L, T_C, K, q = get_iteration_parameters(G.number_of_nodes(), args.eps)
            L = sparse_laplacian(G)
            U, X = sketch_solve_X(L, q, rng)
            M = X @ X

            for norm_constraint in norm_constraints:
                for rho in rho_values:
                    print(f'rho = {rho}')

                    if s_type == 'actual':
                        C_wc, eta_time = worst_case_C_for_fixed_L(L, X, Cbar, rho, s=s, step0=2*rho, tol=1e-10, T_C=T_C, norm_constraint=norm_constraint)

                        worst_case_C_disparity = s.T @ (M * C_wc) @ s
                        worst_case_C_surrogate = s.T @ (X * C_wc) @ s

                        worst_case_Cbar_disparity = s.T @ (M * Cbar) @ s
                        worst_case_Cbar_surrogate = s.T @ (X * Cbar) @ s

                    elif s_type == 'adversarial':
                        C_wc, eta_time = worst_case_C_for_fixed_L(L, X, Cbar, rho, s=None, step0=2*rho, tol=1e-10, T_C=T_C, norm_constraint=norm_constraint)

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
                        'Norm Constraint': norm_constraint,
                        'Rho': rho,
                        'Metric': 'Disparity Percent Change',
                        'Value': disparity_change,
                        'Number of Nodes': n,
                    })
                    records.append({
                        'Name': name,
                        'Nominal Partition Type': group_type,
                        'Norm Constraint': norm_constraint,
                        'Rho': rho,
                        'Metric': 'Surrogate Percent Change',
                        'Value': surrogate_change,
                        'Number of Nodes': n,
                    })

                    records.append({
                        'Name': name,
                        'Nominal Partition Type': group_type,
                        'Norm Constraint': norm_constraint,
                        'Rho': rho,
                        'Metric': 'Time (s)',
                        'Value': eta_time,
                        'Number of Nodes': n,
                    })
                    records.append({
                        'Name': name,
                        'Nominal Partition Type': group_type,
                        'Norm Constraint': norm_constraint,
                        'Rho': rho,
                        'Metric': 'Per Step Time (s)',
                        'Value': eta_time / T_C,
                        'Number of Nodes': n,
                    })

    concat_df = pd.DataFrame(records)

    num_names = concat_df['Name'].nunique()

    fig_a, ax_a = plt.subplots(nrows=1, ncols=(1 + num_names), figsize=(FIGSIZE * (1 + num_names), FIGSIZE), squeeze=False, sharey=True)

    for i, name in enumerate(concat_df['Name'].unique()):
        df_name = concat_df[(concat_df['Name'] == name) & (concat_df['Metric'].isin(['Disparity Percent Change', 'Surrogate Percent Change']))].copy()
        sns.lineplot(x='Rho', y='Value', hue='Metric', style='Nominal Partition Type', markers=True, marker='x', markersize=5, data=df_name, ax=ax_a[0, i], legend=(i == num_names - 1))
        ax_a[0, i].set_title(name)
        ax_a[0, i].set_xlabel('$\\rho$')
        ax_a[0, i].set_ylabel('Percent Change')

    df_time = concat_df[concat_df['Metric'].isin(['Time (s)'])].copy()
    sns.lineplot(x='Number of Nodes', y='Value', hue='Rho', markers=True, marker='x', markersize=5, data=df_time, ax=ax_a[0, -1], legend=False)
    ax_a[0, -1].set_title('Time')
    ax_a[0, -1].set_xlabel('Number of Nodes')
    ax_a[0, -1].set_ylabel('Time (s)')

    fig_a.suptitle(f'Worst Case C Oracle (s = {s_type})')
    fig_a.tight_layout()
    fig_a.savefig(f'{out_dir}/experiment_3_{s_type}_worst_case_C_oracle.pdf', dpi=300, bbox_inches='tight')
    concat_df.to_csv(f'{out_dir}/experiment_3_{s_type}_worst_case_C_oracle.csv', index=False)


def experiment_4_robust_link_recommendation_oracle(args: argparse.Namespace):
    out_dir = args.out_dir

    datasets = get_datasets(args)

    rho_values = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    norm_constraints = ['spectral_ball']

    concat_df = []

    for name, group_types in datasets:
        for group_type in group_types:
            print(f"Running {name} with {group_type}")
            G, s, Cbar = load_dataset(name, group_type)   
            T_L, T_C, K, q = get_iteration_parameters(G.number_of_nodes(), args.eps)


            for rho in rho_values:
                for norm_constraint in norm_constraints:
                    print(f'rho = {rho}, norm_constraint = {norm_constraint}')

                    df_inner, df_outer, eta_time = robust_link_recommendation(G=G, Cbar=Cbar, rho=rho, name=name, q=q, norm_constraint=norm_constraint, T_L=T_L, T_C=T_C, K=K, batch_size=args.batch_size, eta_L=1.0, eta_C=2*rho, seed=args.seed)
                    df_inner['Time (s)'] = eta_time
                    df_outer['Time (s)'] = eta_time
                    df_inner['Per Step Time (s)'] = eta_time / (K * T_C * T_L)
                    df_outer['Per Step Time (s)'] = eta_time / (K * T_C * T_L)

                    concat_df.append(df_inner)
                    concat_df.append(df_outer)

    concat_df = pd.concat(concat_df, ignore_index=True)
    concat_df.to_csv(f'{out_dir}/experiment_4_robust_link_recommendation_oracle.csv', index=False)


def experiment_5_fiedler_gradient_ascent(args: argparse.Namespace):
    """
    Same layout as experiment 1, but edge updates follow gradient ascent on algebraic
    connectivity (Fiedler value λ_2 of L).
    """
    out_dir = args.out_dir

    datasets = get_datasets(args)

    concat_df = []

    for name, group_types in datasets:
        for group_type in group_types:
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
                eta=args.eta,
                seed=args.seed,
            )

            df["Name"] = name
            df["Nominal Partition Type"] = group_type
            df["Number of Link Recommendations"] = T_L
            df["Number of Sketch Vectors"] = q
            df["Number of Nodes"] = G.number_of_nodes()
            df["Batch Size"] = args.batch_size
            df["Learning Rate"] = args.eta
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
            palette="deep",
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
    )

    alpha = 0.05

    min_number_of_nodes = int((1 - alpha) * concat_df["Number of Nodes"].min())
    max_number_of_nodes = int((1 + alpha) * concat_df["Number of Nodes"].max())

    ax_a[0, -1].set_title("Runtime of Robust Link Recommendation")
    ax_a[0, -1].set_xlabel("Number of Nodes")
    ax_a[0, -1].set_ylabel("Runtime (s)")
    ax_a[0, -1].set_xlim(min_number_of_nodes, max_number_of_nodes)

    ax_a[0, -1].set_yscale("log")

    fig_a.suptitle("Fiedler value maximization (gradient ascent on $L$)")
    fig_a.tight_layout()
    fig_a.savefig(f"{out_dir}/experiment_5a_fiedler_gradient_ascent.pdf", dpi=300, bbox_inches="tight")

    fig_b.suptitle("Fiedler value maximization (gradient ascent on $L$)")
    fig_b.tight_layout()
    fig_b.savefig(f"{out_dir}/experiment_5b_fiedler_gradient_ascent.pdf", dpi=300, bbox_inches="tight")

    fig_c.suptitle("Per step time — Fiedler gradient ascent")
    fig_c.tight_layout()
    fig_c.savefig(f"{out_dir}/experiment_5c_fiedler_gradient_ascent.pdf", dpi=300, bbox_inches="tight")

    concat_df.to_csv(f"{out_dir}/experiment_5_fiedler_gradient_ascent.csv", index=False)


def experiment_6_link_recommendation_baselines(args: argparse.Namespace) -> None:
    """Disparity-focused reweighting: oracle vs random / max-degree / max-betweenness."""
    out_dir = args.out_dir
    datasets = get_datasets(args)
    selections = ["leverage", "random", "max_degree", "max_betweenness"]
    concat_df: List[pd.DataFrame] = []

    for name, group_types in datasets:
        for group_type in group_types:
            G, s, Cbar = load_dataset(name, group_type)
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
                    eta=args.eta,
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
                df["Learning Rate"] = args.eta
                df["Seed"] = args.seed
                df["Time (s)"] = eta_time
                df["Per Step Time (s)"] = eta_time / T_L
                concat_df.append(df)

    df_all = pd.concat(concat_df, ignore_index=True)
    df_all = df_all[np.isfinite(df_all["Percent Change"])].copy()
    df_all.to_csv(f"{out_dir}/experiment_6_link_recommendation_baselines.csv", index=False)

    step_max = df_all["Step"].max()
    df_last = df_all[df_all["Step"] == step_max].copy()
    metrics = ["Surrogate", "Disparity", "Polarization"]
    df_bar = df_last[df_last["Metric"].isin(metrics)].copy()

    g = sns.catplot(
        data=df_bar,
        x="Method",
        y="Percent Change",
        hue="Nominal Partition Type",
        col="Name",
        row="Metric",
        kind="bar",
        sharey=False,
        height=2.8,
        aspect=1.4,
        legend=True,
    )
    g.figure.suptitle("Link recommendation (disparity): baselines vs oracle", y=1.02)
    g.set_xticklabels(rotation=25, ha="right")
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
    """Fiedler maximization: oracle vs random / max-degree / max-betweenness."""
    out_dir = args.out_dir
    datasets = get_datasets(args)
    selections = ["fiedler_grad", "random", "max_degree", "max_betweenness"]
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
                    eta=args.eta,
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
                df["Learning Rate"] = args.eta
                df["Seed"] = args.seed
                df["Time (s)"] = eta_time
                df["Per Step Time (s)"] = eta_time / T_L
                concat_df.append(df)

    df_all = pd.concat(concat_df, ignore_index=True)
    df_all = df_all[np.isfinite(df_all["Percent Change"])].copy()
    df_all.to_csv(f"{out_dir}/experiment_7_fiedler_baselines.csv", index=False)

    step_max = df_all["Step"].max()
    df_last = df_all[df_all["Step"] == step_max].copy()
    metrics = [r"Fiedler $\lambda_2$", "Surrogate", "Disparity", "Polarization"]
    df_bar = df_last[df_last["Metric"].isin(metrics)].copy()

    g = sns.catplot(
        data=df_bar,
        x="Method",
        y="Percent Change",
        hue="Nominal Partition Type",
        col="Name",
        row="Metric",
        kind="bar",
        sharey=False,
        height=2.8,
        aspect=1.4,
        legend=True,
    )
    g.figure.suptitle("Fiedler maximization: baselines vs oracle", y=1.02)
    g.set_xticklabels(rotation=25, ha="right")
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
    inner_selections = ["leverage", "random", "max_degree", "max_betweenness"]
    norm_constraints = ["spectral_ball"]
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
                            eta_L=1.0,
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
    df_all.to_csv(f"{out_dir}/experiment_8_robust_link_recommendation_baselines.csv", index=False)

    df_outer = df_all[df_all["Metric"].str.startswith("Worst", na=False)].copy()
    if len(df_outer) == 0:
        return

    g = sns.relplot(
        data=df_outer,
        x="Rho",
        y="Percent Change",
        hue="Method",
        style="Metric",
        col="Name",
        row="Nominal Partition Type",
        kind="line",
        markers=True,
        dashes=False,
        facet_kws={"sharey": False},
    )
    g.figure.suptitle("Robust link recommendation: inner-loop baselines (outer metrics vs $\\rho$)", y=1.02)
    g.tight_layout()
    g.savefig(f"{out_dir}/experiment_8a_robust_link_recommendation_baselines.pdf", dpi=300, bbox_inches="tight")
    plt.close("all")


def main(args: argparse.Namespace) -> None:
    # experiment_1_link_recommendation_oracle(args)
    # experiment_2_link_recommendation_oracle(args)
    # experiment_3_worst_case_C_oracle(args, s_type='actual')
    # experiment_4_robust_link_recommendation_oracle(args)
    # experiment_5_fiedler_gradient_ascent(args)
    experiment_6_link_recommendation_baselines(args)
    experiment_7_fiedler_baselines(args)
    # experiment_8_robust_link_recommendation_baselines(args)

if __name__ == "__main__":
    args = parse_args()
    main(args)

