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

import cvxpy as cp

import seaborn as sns
import matplotlib.pyplot as plt
from scipy.linalg import solve_triangular
from scipy.linalg import cho_factor, cho_solve, eigh

from dataclasses import dataclass

Array = np.ndarray
Edge = Union[Tuple[int, int], Tuple[str, str]]

sns.set_theme(style="whitegrid")
sns.set_palette("deep")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default=['polblogs'], type=str, required=True)
    parser.add_argument('--group_type', type=str, choices=['spectral', 'random', 'label'], required=True)
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
    return parser.parse_args()

def load_weighted_undirected_graph(edge_file: str) -> nx.Graph:
    df = pd.read_csv(edge_file, header=None, sep=None, engine="python")
    if df.shape[1] < 2:
        raise ValueError("Edge file must have at least 2 columns: u v [w].")

    G = nx.Graph()
    if df.shape[1] == 2:
        for u, v in df.itertuples(index=False, name=None):
            G.add_edge(str(u), str(v), weight=1.0)
    else:
        for row in df.itertuples(index=False, name=None):
            u, v, w = row[0], row[1], row[2]
            G.add_edge(str(u), str(v), weight=float(w))

    if G.number_of_nodes() == 0:
        raise ValueError("Loaded graph is empty.")

    return G

def load_opinions(opinion_file: str) -> Dict[str, float]:
    """Load node -> opinion (scalar). Lines are node id and value, separated by tab or spaces."""
    df = pd.read_csv(opinion_file, header=None, sep=r"\s+", engine="python")
    if df.shape[1] < 2:
        raise ValueError("Opinion file must have two columns: node_id opinion_value.")

    out: Dict[str, float] = {}
    for node, val in df.iloc[:, 0:2].itertuples(index=False, name=None):
        out[str(node).strip()] = float(val)
    return out

def largest_connected_component(G: nx.Graph) -> nx.Graph:
    """Keep the largest connected component only."""
    if nx.is_connected(G):
        return G.copy()
    ccs = list(nx.connected_components(G))
    giant = max(ccs, key=len)
    return G.subgraph(giant).copy()

def spectral_partition_labels(G: nx.Graph, nodes: list[str]) -> np.ndarray:
    """
    Spectral bisection: ±1 labels from the sign of the Fiedler vector (unweighted Laplacian),
    in the order given by ``nodes``.
    """
    L = graph_to_laplacian(G, nodes)
    n = L.shape[0]
    if n <= 1:
        return np.ones(n, dtype=int)
    vals, vecs = spla.eigsh(L.astype(np.float64), k=2, which="SA")
    fiedler = vecs[:, 1]
    signs = np.sign(fiedler.astype(float))
    signs[signs == 0] = 1
    return signs.astype(int)

def graph_to_laplacian(G: nx.Graph, nodes: list[str]) -> sp.csr_matrix:
    """
    Returns L = D - W as a sparse matrix in the node order given by nodes.
    """
    W = nx.to_scipy_sparse_array(G, nodelist=nodes, weight="weight", format="csr", dtype=float)
    deg = np.asarray(W.sum(axis=1)).ravel()
    L = sp.diags(deg, format="csr") - W
    return L

def calculate_M_fast(L: sp.csr_matrix) -> sp.csr_matrix:
    # M = (I + L)**(-2) 
    # use eigenvalues and eigenvectors to compute M
    if isinstance(L, sp.csr_matrix):
        eigenvalues, eigenvectors = spla.eigsh(L.toarray(), k=min(L.shape[0], 10), which='SA')
    else:
        eigenvalues, eigenvectors = spla.eigsh(L, k=min(L.shape[0], 10), which='SA')
    M = eigenvectors @ np.diag(1 / (1 + eigenvalues)**2) @ eigenvectors.T
    return sp.csr_matrix(M)

def calculate_X_fast(L: sp.csr_matrix) -> sp.csr_matrix:
    # X = (I + L)**(-1) 
    # use eigenvalues and eigenvectors to compute X
    if isinstance(L, sp.csr_matrix):
        eigenvalues, eigenvectors = spla.eigsh(L.toarray(), k=min(L.shape[0], 10), which='SA')
    else:
        eigenvalues, eigenvectors = spla.eigsh(L, k=min(L.shape[0], 10), which='SA')
    X = eigenvectors @ np.diag(1 / (1 + eigenvalues)) @ eigenvectors.T
    return sp.csr_matrix(X)

def calculate_auxiliary_matrices(L: sp.csr_matrix, C: sp.csr_matrix) -> Tuple[sp.csr_matrix, sp.csr_matrix, sp.csr_matrix, sp.csr_matrix]:
    M = calculate_M_fast(L)
    X = calculate_X_fast(L)
    Z = M * C
    Z_tilde = X * C
    return M, X, Z, Z_tilde

def normalize_unit(x: np.ndarray) -> np.ndarray:
    nrm = np.linalg.norm(x)
    if nrm == 0:
        raise ValueError("Cannot normalize the zero vector.")
    return x / nrm

def build_opinion_vector(opinions: np.ndarray) -> np.ndarray:
    """Normalize observed opinions to ||s||_2 = 1."""
    s = opinions.astype(float).copy()
    return normalize_unit(s)

def group_masks(labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mask_a = (labels == +1)
    mask_b = (labels == -1)
    return mask_a, mask_b

def consensus(solve, s: np.ndarray) -> np.ndarray:
    return solve(s)

def compute_polarization(solve, s: np.ndarray, M: sp.csr_matrix, Z: sp.csr_matrix) -> float:
    return float(s.T @ M @ s)


def compute_disparity(s: np.ndarray, M: sp.csr_matrix, Z: sp.csr_matrix) -> float:
    return float(s.T @ M @ Z @ s)


def sparse_laplacian(G: nx.Graph, nodelist: Optional[Sequence[int]] = None) -> sp.csr_matrix:
    """Return the weighted Laplacian matrix as CSR."""
    if nodelist is None:
        nodelist = list(G.nodes())
    L = nx.laplacian_matrix(G, nodelist=nodelist, weight="weight")
    return L.tocsr().astype(float)

def corr_from_labels(labels: Array) -> Array:
    """Correlation matrix from labels."""
    labels = np.asarray(labels, dtype=float).reshape(-1)
    C = np.outer(labels, labels)
    return C

def top_eigenpair(A, tol=1e-6, max_iter=200):
    """
    Largest eigenpair of (approximately) symmetric A.

    Uses dense ``eigh`` for moderate n so the value is deterministic and not cut off
    by a spurious power-iteration stop when ``lam_old`` was initialized to 0 and
    ``|lambda_max| < tol``.
    """
    n = A.shape[0]
   
    v = np.ones(n, dtype=np.float64)
    v /= np.linalg.norm(v)
    lam_old: Optional[float] = None
    lam = 0.0
    for _ in range(max_iter):
        w = A @ v
        norm_w = float(np.linalg.norm(w))
        if norm_w == 0.0:
            return float(lam), v
        v = w / norm_w
        lam = float(v @ (A @ v))
        if lam_old is not None and abs(lam - lam_old) < tol:
            break
        lam_old = lam
    return lam, v


def project_psd(M):
    """
    Projection onto the PSD cone.
    """
    w, U = np.linalg.eigh(M)
    w_clipped = np.maximum(w, 0.0)
    return (U * w_clipped) @ U.T


def enforce_unit_diagonal(C):
    """
    Set diag(C) = 1.
    """
    C = C.copy()
    np.fill_diagonal(C, 1.0)
    return C


def project_spectral_ball(C: np.ndarray, Cbar: np.ndarray, rho: float) -> np.ndarray:
    """
    Project C onto the spectral ball {X : ||X - Cbar||_2 <= rho}
    by clipping eigenvalues of (C - Cbar).
    """
    M = C - Cbar
    w, U = np.linalg.eigh(M)
    w_clipped = np.clip(w, -rho, rho)
    M_proj = (U * w_clipped) @ U.T
    return Cbar + M_proj

def project_to_correlation(A, n_iters=20, eps=1e-10):
    """
    Approximate projection onto {C PSD, diag(C)=1}.
    """
    X = 0.5 * (A + A.T)
    for _ in range(n_iters):
        # PSD projection
        w, V = eigh(X)
        w = np.clip(w, eps, None)
        X = (V * w) @ V.T

        # Unit diagonal projection
        np.fill_diagonal(X, 1.0)

        X = 0.5 * (X + X.T)
    return X


def project_onto_U(C: np.ndarray, Cbar: np.ndarray, rho: float, n_rounds: int = 5) -> np.ndarray:
    """
    Approximate projection onto U = {C PSD, diag(C)=1, ||C-Cbar||_2 <= rho}
    using alternating projections.
    """
    X = C.copy()
    for _ in range(n_rounds):
        X = project_psd(X)
        X = enforce_unit_diagonal(X)
        X = project_spectral_ball(X, Cbar, rho)
    # final cleanup
    X = 0.5 * (X + X.T)
    X = project_psd(X)
    X = enforce_unit_diagonal(X)
    return X


def worst_case_C_for_fixed_L(
    L: np.ndarray,
    X: np.ndarray,
    Cbar: np.ndarray,
    rho: float,
    T_C: int = 20,
    step0: float = 1.0,
    tol: float = 1e-5,
    C0: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Approximately solve:
        max_{C in U} lambda_max( X ⊙ C )
    where X = (I + L)^(-1) is kept implicit/updated elsewhere.

    If ``C0`` is given (feasible correlation in U), subgradient ascent starts there; otherwise from ``Cbar``.
    """
    n = L.shape[0]

    C = Cbar.copy() if C0 is None else np.asarray(C0, dtype=np.float64).copy()
    prev_val = -np.inf

    for t in range(T_C):
        Z = X * C  # Hadamard product

        lam, v = top_eigenpair(Z)

        # Subgradient of lambda_max(X ⊙ C) w.r.t. C
        G = X * np.outer(v, v)

        # Step size schedule
        eta = step0 / np.sqrt(t + 1)

        # Ascent step
        C_hat = C + eta * G

        # Projection back to U
        C_new = project_onto_U(C_hat, Cbar, rho, n_rounds=5)

        # stopping criterion
        new_val, _ = top_eigenpair(X * C_new)
        if abs(new_val - prev_val) < tol:
            C = C_new
            break

        C = C_new
        prev_val = new_val

    return C


def sketch_solve(A: sp.csr_matrix, R: np.ndarray) -> np.ndarray:
    """Solve A U = R column-wise (e.g. CG). R is (n, q). Returns U with same shape."""
    n, q = R.shape
    if A.shape[0] != n:
        raise ValueError("A and R must agree in row dimension.")
    U = np.zeros((n, q), dtype=np.float64)
    for k in range(q):
        U[:, k], _ = spla.cg(A, R[:, k], atol=1e-8)
    return U


def sketch_U_sherman_morrison_two_rank(
    U: np.ndarray,
    q: int,
    w: float,
    a: np.ndarray,
    c: np.ndarray,
    denom_eps: float = 1e-14,
) -> np.ndarray:
    """
    Update sketch U when (I+L) becomes A' = A + w a a^T - w c c^T, with U ≈ A^{-1} R and
    X = A^{-1} ≈ (1/q) U U^T. Two Sherman–Morrison steps: A_1 = A + u u^T (u = sqrt(w) a),
    then A' = A_1 - z z^T (z = sqrt(w) c).
    """
    if w == 0.0:
        return U
    u_vec = np.sqrt(w) * np.asarray(a, dtype=np.float64).ravel()
    z_vec = np.sqrt(w) * np.asarray(c, dtype=np.float64).ravel()

    t_u = U.T @ u_vec
    Xu = (U @ t_u) / q
    d = float(u_vec @ Xu)
    denom1 = 1.0 + d
    if abs(denom1) < denom_eps:
        denom1 = math.copysign(denom_eps, denom1) if denom1 != 0.0 else denom_eps

    t_z = U.T @ z_vec
    Xz = (U @ t_z) / q
    X1_z = Xz - Xu * float(Xu @ z_vec) / denom1

    U1 = U - np.outer(Xu, t_u) / denom1

    d2 = 1.0 - float(z_vec @ X1_z)
    if abs(d2) < denom_eps:
        d2 = math.copysign(denom_eps, d2) if d2 != 0.0 else denom_eps

    g = U1.T @ z_vec
    return U1 + np.outer(X1_z, g) / d2


def sketch_solve_X(L: sp.csr_matrix, q: int, seed: int = 0, return_X: bool = False) -> np.ndarray:
    # L + I sparse matrix
    L_plus_I = L + sp.eye(L.shape[0], format="csr")
    rng = np.random.default_rng(seed)
    R = rng.standard_normal((L.shape[0], q))
    U = sketch_solve(L_plus_I, R)
    
    if return_X:
        X_approx = U @ U.T / q
        return U, X_approx
    else:
        return U, None


def robust_link_recommendation(G: nx.Graph, s: np.ndarray, Cbar: np.ndarray, rho: float, name: str, K: int = 10, T_L: int = 50, T_C: int = 20, eta: float = 1, batch_size: int = 100, seed: int = 0) -> None:
    L0 = sparse_laplacian(G)
    H = G.copy()

    # One inner solve at Cbar seeds the active set; later outer iterations only re-run link when k >= 1.
    df, L, X, _, _ = link_recommendation(H, s, Cbar, name, T_L=T_L, batch_size=batch_size, eta=eta, seed=seed, plot=False)
    df['k'] = 0
    active_set = [(Cbar.copy(), L, float(top_eigenpair(X * Cbar)[0]), df)]

    for k in range(K):
        if k > 0:
            C_inner = active_set[-1][0]
            df, L, X, _, _ = link_recommendation(
                H, s, C_inner, name, T_L=T_L, batch_size=batch_size, eta=eta, seed=seed, plot=False
            )
            df['k'] = k
        else:
            C_inner = Cbar

        worst_current = float(top_eigenpair(X * C_inner)[0])
        C_new = worst_case_C_for_fixed_L(L, X, Cbar, rho, T_C=T_C, C0=C_inner)
        worst_new = float(top_eigenpair(X * C_new)[0])

        # Only stop when the adversarial step does not improve λ_max beyond numerical noise.
        scale = max(1.0, abs(worst_current))
        if worst_new <= worst_current + 1e-12 + 1e-10 * scale:
            break

        active_set.append((C_new, L, worst_new, df))

    df.to_csv(f'figures/robust_link_recommendation_{name}.csv', index=False)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 4), squeeze=False)

    records = []

    for i, (C_, L_, worst_, df_) in enumerate(active_set):
        diff_L = L_ - L0
        diff_C = C_ - Cbar
        diff_norm_L = np.linalg.norm(diff_L, 'fro')
        diff_norm_C = np.linalg.norm(diff_C, 'fro')
        records.append({
            'Step': i,
            'Metric': 'Objective',
            'Value': worst_,
        })
        records.append({
            'Step': i,
            'Metric': '$\\|\\hat{{L}}_K - L_{{0}}\\|_F$',
            'Value': diff_norm_L,
        })
        records.append({
            'Step': i,
            'Metric': '$\\|\\hat{{C}}_K - C_{{bar}}\\|_F$',
            'Value': diff_norm_C,
        })

    df_records = pd.DataFrame(records)
    
    sns.lineplot(x='Step', y='Value', hue='Metric', data=df_records, ax=ax[0, 0], legend=True)
    fig.tight_layout()
    plt.savefig(f'figures/active_set_{name}.pdf', dpi=300, bbox_inches='tight')

def link_recommendation(G: nx.Graph, s: np.ndarray, C: np.ndarray, name: str, T_L: int = 100, batch_size: int = 100, eta: float = 1, seed: int = 0, plot: bool = False) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, nx.Graph]:
    nodelist = list(G.nodes())
    B = nx.incidence_matrix(G, nodelist=nodelist, oriented=True).toarray()
    edge_to_col = {}
    for j, e in enumerate(G.edges()):
        u_e, v_e = e
        edge_to_col[(u_e, v_e)] = j
        edge_to_col[(v_e, u_e)] = j
    L = sparse_laplacian(G)

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
        weight_change = min(eta_current, H[u_plus][v_plus]['weight'], H[u_minus][v_minus]['weight'])


        H[u_plus][v_plus]['weight'] += weight_change            
        H[v_plus][u_plus]['weight'] += weight_change
        H[u_minus][v_minus]['weight'] -= weight_change
        H[v_minus][u_minus]['weight'] -= weight_change

        if H[u_minus][v_minus]['weight'] == 0:
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

        surrogate_disparity = s.T @ Z_tilde @ s
        disparity = s.T @ Z @ s
        polarization = s.T @ M @ s

        if i == 0:
            initial_disparity = disparity
            initial_polarization = polarization
            initial_surrogate_disparity = surrogate_disparity

        records.append({
            'Step': i,
            'Metric': 'Surrogate Disparity',
            'Value': (surrogate_disparity - initial_surrogate_disparity) / initial_surrogate_disparity * 100,
        })
        records.append({
            'Step': i,
            'Metric': 'Disparity',
            'Value': (disparity - initial_disparity) / initial_disparity * 100,
        })
        records.append({
            'Step': i,
            'Metric': 'Polarization',
            'Value': (polarization - initial_polarization) / initial_polarization * 100,
        })

        print(f"Step {i}: Weight Change: {weight_change}, Surrogate Disparity: {surrogate_disparity}, Disparity: {disparity}, Polarization: {polarization}")

    df = pd.DataFrame(records)
    
    if plot:
        fig, ax = plt.subplots(figsize=(8, 8))
        sns.lineplot(x='Step', y='Value', hue='Metric', data=df, ax=ax)
        ax.set_title('Link Recommendation (Continuous)')
        ax.set_xlabel('Step')
        ax.set_ylabel('Percentage Change')
        ax.legend(loc='lower left')
        sns.despine(fig)
        fig.tight_layout()
        plt.savefig(f'figures/link_recommendation_{name}_continuous.pdf', dpi=300, bbox_inches='tight')
        plt.close(fig)

    return df, L, X, M, H


def cholesky_add_node(L, z, z_uu, jitter=1e-12):
    """
    Append one node to the Cholesky factor of Z_SS.

    Parameters
    ----------
    L : (m, m) ndarray or None
        Lower-triangular Cholesky factor such that Z_SS = L @ L.T
        for the current seed set S.
        If S is empty, pass None.
    z : (m,) ndarray
        Cross term Z_{S,u} for the new node u.
        If S is empty, pass an empty array.
    z_uu : float
        Diagonal entry Z_{u,u}.
    jitter : float
        Numerical stabilization for tiny negative Schur complements.

    Returns
    -------
    L_new : (m+1, m+1) ndarray
        Updated lower-triangular Cholesky factor.
    alpha : float
        New diagonal Schur complement before the square root:
            alpha = z_uu - z.T @ Z_SS^{-1} @ z
    """
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

def compute_delta_with_factor(Z, Q, s, S, ridge=1e-8, normalize: bool = False):
    """
    Compute the optimal intervention delta for a fixed seed set S,
    reusing a precomputed Cholesky factor of Z_SS.

    Parameters
    ----------
    Z : (n, n) ndarray
        PSD matrix in the objective.
    Q : (|S|, |S|) ndarray
        Lower-triangular Cholesky factor L with Z_SS = L @ L.T (same convention
        as ``cholesky_add_node`` / ``solve_with_cholesky``), not a ``cho_factor`` tuple.

    s : (n,) ndarray
        Opinion vector.
    S : list[int] or array-like
        Selected seed set.
    ridge : float
        Small regularization for numerical stability.

    Returns
    -------
    delta : (n,) ndarray
        Full intervention vector, supported on S.
    delta_S : (|S|,) ndarray
        Nonzero coordinates on S.
    """
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

def solve_with_cholesky(L, b):
    """
    Solve Z_SS x = b using the Cholesky factor L of Z_SS.
    """
    y = solve_triangular(L, b, lower=True, check_finite=False)
    x = solve_triangular(L.T, y, lower=False, check_finite=False)
    return x

def project_psd(A, eps=1e-10):
    """
    Symmetrize and project onto the PSD cone.
    """
    A = 0.5 * (A + A.T)
    w, V = eigh(A)
    w = np.maximum(w, eps)
    return (V * w) @ V.T


def opinion_seeding_fast(G: nx.Graph, s: np.ndarray, C: np.ndarray, name: str, b: int = 100, seed: int = 0, plot: bool = False) -> Tuple[pd.DataFrame, list[int], float]:
    L = sparse_laplacian(G)
    M = calculate_M_fast(L)

    Z = M * C
    records = []
    n = Z.shape[0]
    S = []
    remaining = list(range(n))
    Q = None

    objective_value = 0

    Z = 0.5 * (Z + Z.T)
    Z = Z + 1e-10 * np.eye(Z.shape[0])

    records.append({
        'Step': 0,
        'Metric': 'Objective Value',
        'Value': objective_value,
        'S': S,
    })

    for i in range(b):
        best_u = None
        best_gain = -np.inf
        best_z = None
        best_alpha = None

        S_idx = np.asarray(S, dtype=np.intp)

        for u in remaining:
            ui = int(u)
            if len(S) == 0:
                # No solve needed yet
                alpha = float(Z[ui, ui])
                q_u = Z[:, ui]
            else:
                Su = np.ix_(S_idx, np.array([ui], dtype=np.intp))
                z = Z[Su].reshape(-1)  # Z_{S,u}
                v = solve_with_cholesky(Q, z)  # Z_SS^{-1} Z_{S,u}
                alpha = float(Z[ui, ui] - np.dot(z, v))
                alpha = max(alpha, 1e-12)

                # q_S = Z_{:,u} - Z_{:,S} Z_SS^{-1} Z_{S,u}
                q_u = Z[:, ui] - Z[:, S_idx] @ v

            if alpha <= 1e-12:
                continue

            gain = (np.dot(s, q_u) ** 2) / alpha

            # print('Node: ', u, 'Gain: ', gain, 'Best Gain: ', best_gain)

            if np.isfinite(gain) and gain > best_gain:
                best_gain = gain
                best_u = ui
                best_z = (
                    None
                    if len(S) == 0
                    else Z[np.ix_(S_idx, np.array([ui], dtype=np.intp))]
                    .reshape(-1)
                    .astype(np.float64, copy=False)
                )
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

        S.append(bu)
        remaining.remove(bu)
        objective_value += best_gain
        delta, delta_S = compute_delta_with_factor(Z, Q, s, S)

        print(f"Step {i}: Objective Value: {objective_value}, S: {S}, Delta_S: {delta_S}")

        records.append({
            'Step': i,
            'Metric': 'Objective Value',
            'Value': objective_value,
            'S': list(S),
            'Delta': delta,
            'Delta_S': delta_S,
        })

        print(f"Step {i}: Objective Value: {objective_value}, S: {S}")

    df = pd.DataFrame(records)

    if plot:
        fig, ax = plt.subplots(figsize=(4, 4))
        sns.lineplot(x='Step', y='Value', hue='Metric', data=df, ax=ax)
        ax.set_title('Opinion Seeding')
        ax.set_xlabel('Step')
        ax.set_ylabel('Objective Value')
        ax.legend(loc='lower left')
        sns.despine(fig)
        fig.tight_layout()
        plt.savefig(f'figures/opinion_seeding_{name}.pdf', dpi=300, bbox_inches='tight')
        plt.close(fig)

    return df, S, objective_value

@dataclass
class ScenarioOracle:
    """
    Fast oracle for one scenario:
        F(S) = s^T P_S s,   P_S = Z_:S Z_SS^{-1} Z_S:
    and
        delta_S^* = - Z_SS^{-1} Z_Sbar s_bar.
    """
    Z: np.ndarray
    s: np.ndarray
    ridge: float = 1e-8

    def __post_init__(self):
        self.Z = project_psd(self.Z, eps=self.ridge)
        self.s = np.asarray(self.s, dtype=float).reshape(-1)
        ns = np.linalg.norm(self.s)
        if ns == 0:
            raise ValueError("s must be nonzero.")
        self.s = self.s / ns

        self._benefit_cache = {}
        self._delta_cache = {}

    def _factor(self, S):
        idx = np.asarray(tuple(sorted(S)), dtype=np.intp)
        ZSS = self.Z[np.ix_(idx, idx)]
        ZSS = 0.5 * (ZSS + ZSS.T) + self.ridge * np.eye(len(idx))
        return idx, cho_factor(ZSS, lower=True, check_finite=False)

    def benefit(self, S) -> float:
        """
        Exact fixed-set benefit F(S), computed stably.
        """
        key = tuple(sorted(S))
        if key in self._benefit_cache:
            return self._benefit_cache[key]

        if len(key) == 0:
            self._benefit_cache[key] = 0.0
            return 0.0

        idx, cfac = self._factor(key)
        # v = Z_{S,:} s
        v = self.Z[np.ix_(idx, np.arange(self.Z.shape[0]))] @ self.s
        x = cho_solve(cfac, v, check_finite=False)
        val = float(v @ x)
        val = max(val, 0.0)
        self._benefit_cache[key] = val
        return val

    def delta(self, S):
        """
        Optimal fixed-set intervention delta_S^* padded to length n.
        """
        key = tuple(sorted(S))
        if key in self._delta_cache:
            return self._delta_cache[key]

        n = self.Z.shape[0]
        delta = np.zeros(n, dtype=float)

        if len(key) == 0:
            self._delta_cache[key] = (delta, np.array([], dtype=float))
            return delta, np.array([], dtype=float)

        idx, cfac = self._factor(key)
        bar = np.setdiff1d(np.arange(n), idx, assume_unique=False)

        # delta_S^* = -Z_SS^{-1} Z_Sbar s_bar
        rhs = self.Z[np.ix_(idx, bar)] @ self.s[bar]
        delta_S = -cho_solve(cfac, rhs, check_finite=False)

        delta[idx] = delta_S
        self._delta_cache[key] = (delta, delta_S)
        return delta, delta_S


def saturate_fast_helper(oracles: List[ScenarioOracle], ground_set: List[int], tol: float = 1e-6, max_outer: int = 50) -> Tuple[list[int], float]:
    """
    Saturate over a finite set of scenarios, using fast benefit oracles.

    Returns
    -------
    best_S : set[int]
        Seed set for the discretized robust problem.
    best_c : float
        Largest feasible threshold found.
    """
    ground_set = list(ground_set)
    m = len(oracles)

    def truncated_sum(S, c):
        return sum(min(oracle.benefit(S), c) for oracle in oracles)

    def greedy_cover(c):
        """
        Approximately solve the submodular covering subproblem:
            find S such that sum_l min(F_l(S), c) >= m * c
        """
        S = set()
        current = truncated_sum(S, c)

        while current < m * c - tol:
            best_u = None
            best_gain = -np.inf

            for u in ground_set:
                if u in S:
                    continue
                cand = S | {u}
                gain = truncated_sum(cand, c) - current
                if gain > best_gain:
                    best_gain = gain
                    best_u = u

            if best_u is None or best_gain <= tol:
                return S, False

            S.add(best_u)
            current += best_gain

        return S, True

    # Safe upper bound: worst-case value on the full set
    full_S = set(ground_set)
    hi = min(oracle.benefit(full_S) for oracle in oracles)
    lo = 0.0

    best_S = set()
    best_c = 0.0

    for _ in range(max_outer):
        c = 0.5 * (lo + hi)
        S_c, feasible = greedy_cover(c)

        if feasible:
            lo = c
            best_S = S_c
            best_c = c
        else:
            hi = c

        if hi - lo <= tol:
            break

    return best_S, best_c


def spectral_extremes(C_bar, rho, k=3):
    w, V = np.linalg.eigh(0.5 * (C_bar + C_bar.T))
    idx = np.argsort(-w)

    scenarios = []
    for i in range(k):
        v = V[:, idx[i]]
        H = np.outer(v, v)
        np.fill_diagonal(H, 0.0)

        # push both directions
        for sign in [+1, -1]:
            C = C_bar + sign * rho * H
            C = project_to_correlation(C)
            scenarios.append(C)

    return scenarios

def random_low_rank(C_bar, rho, n_samples=5, rank=2):
    n = C_bar.shape[0]
    scenarios = []

    for _ in range(n_samples):
        U = np.random.randn(n, rank)
        H = U @ U.T
        H = 0.5 * (H + H.T)
        np.fill_diagonal(H, 0.0)

        H = H / np.linalg.norm(H, 2)

        C = C_bar + rho * H
        C = project_to_correlation(C)

        scenarios.append(C)

    return scenarios

def generate_scenarios(C_bar, rho):
    scenarios = []

    # anchors
    scenarios.append(C_bar)
    scenarios.append(np.eye(C_bar.shape[0]))

    # spectral extremes
    # scenarios += spectral_extremes(C_bar, rho, k=3)


    # random perturbations
    scenarios += random_low_rank(C_bar, rho, n_samples=5)

    import pdb; pdb.set_trace()

    return scenarios


def robust_opinion_seeding(G: nx.Graph, s: np.ndarray, C_bar: np.ndarray, rho: float, name: str, b: int = 100, seed: int = 0, plot: bool = False) -> Tuple[pd.DataFrame, list[int], float]:
    L = sparse_laplacian(G)
    M = calculate_M_fast(L)

    Cs = generate_scenarios(C_bar, rho)

    scenario_oracles = []
    for C_l in Cs:   # list of discrete scenarios C^(1), ..., C^(m)
        Z_l = M * C_l
        scenario_oracles.append(ScenarioOracle(Z_l, s, ridge=1e-8))

    # Run Saturate on the scenario family
    S_sat, c_star = saturate_fast_helper(scenario_oracles, ground_set=range(len(s)), tol=1e-6)

    # Compute the final interventions for each scenario if needed
    deltas = []
    delta_S_list = []
    for oracle in scenario_oracles:
        delta, delta_S = oracle.delta(S_sat)
        deltas.append(delta)
        delta_S_list.append(delta_S)
        
    print(deltas, delta_S_list, S_sat, c_star)

    return deltas, delta_S_list, S_sat, c_star

def main(args: argparse.Namespace) -> None:
    name = args.name
    group_type = args.group_type
    out_dir = args.out_dir
    rho = args.rho
    k = args.k
    eta = args.eta
    eps = args.eps
    T_L = args.T
    T_C = args.T
    K = args.K

    print(f'T_L = {T_L}, T_C = {T_C}, K = {K}')
    print(f'eps = {eps}')

    import pdb; pdb.set_trace()

    seed = args.seed
    b = args.b

    G = load_weighted_undirected_graph(f'data/{name}/edges.txt')
    G = largest_connected_component(G)
    batch_size = args.batch_size
    
    opinion_map = load_opinions(f'data/{name}/opinions.txt')
    nodes = sorted(
        (str(v) for v in G.nodes() if str(v) in opinion_map),
        key=lambda x: int(x) if x.isdigit() else x,
    )
    
    G = G.subgraph(nodes).copy()
    G = largest_connected_component(G)
    
    nodes = sorted((str(v) for v in G.nodes()), key=lambda x: int(x) if x.isdigit() else x)
    # set all edge weights to 1

    if group_type == 'spectral':
        labels = spectral_partition_labels(G, nodes)
    elif group_type == 'random':
        labels = np.random.choice([-1, 1], size=len(nodes))
    elif group_type == 'label':
        labels = [1 if opinion_map[n] > 0 else -1 for n in nodes]
        labels = np.array(labels)
    else:
        raise ValueError(f"Invalid group type: {group_type}")

    # relabel nodes of G 
    G = nx.relabel_nodes(G, {n: i for i, n in enumerate(nodes)})
    Cbar = corr_from_labels(labels, rho=0)

    for u, v in G.edges():
        G[u][v]['weight'] = 1.0

    s = build_opinion_vector(np.array([opinion_map[n] for n in nodes], dtype=float))

    robust_link_recommendation(G, s, Cbar, rho, name, K=K, T_L=T_L, T_C=T_C, eta=eta, batch_size=batch_size, seed=seed)
    
    
    # edge_list = list(G.edges())
    # n = len(nodes)
    # total_weight = 1.0

    # out = robust_link_recommendation_spectral_ball(
    #     n=n,
    #     edge_list=edge_list,
    #     total_weight=total_weight,
    #     Cbar=C,
    #     rho=rho,
    #     max_outer_iters=10,
    #     tol=1e-4,
    #     verbose=True,
    # )

    # print("\nFinal status:", out["status"])
    # print("Final objective t*:", out["t"])
    # print("Final edge weights w*:", out["w"])
    # print("Final correlation matrix C*:\n", out["C"])

if __name__ == "__main__":
    args = parse_args()
    main(args)
