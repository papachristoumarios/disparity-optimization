from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, List, Optional, Sequence, Tuple, Union

import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.linalg import cho_factor, cho_solve, eigh, solve_triangular
from tqdm import tqdm
import time

Array = np.ndarray
Edge = Union[Tuple[int, int], Tuple[str, str]]

def get_datasets(args: argparse.Namespace):
    datasets = [('reddit', ['spectral']), 
                ('twitter', ['spectral']), 
                ('polblogs', ['spectral'])]
    return datasets

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

def load_dataset(name, group_type, return_labels: bool = False):
    """Load dataset from name and group type."""
    G = load_weighted_undirected_graph(f'data/{name}/edges.txt')
    G = largest_connected_component(G)
    
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
        labels = spectral_partition_labels(G, nodes).astype(np.float64)
    elif group_type == 'random':
        labels = np.random.choice([-1, 1], size=len(nodes)).astype(np.float64)
    elif group_type == 'label':
        labels = [1 if opinion_map[n] > 0 else -1 for n in nodes]
        labels = np.array(labels).astype(np.float64)
    elif group_type == 'polarization':
        labels = np.ones(len(nodes)).astype(np.float64)
    else:
        raise ValueError(f"Invalid group type: {group_type}")

    # relabel nodes of G 
    G = nx.relabel_nodes(G, {n: i for i, n in enumerate(nodes)})
    
    Cbar = corr_from_labels(labels)

    for u, v in G.edges():
        G[u][v]['weight'] = 1.0

    s = build_opinion_vector(np.array([opinion_map[n] for n in nodes], dtype=float))

    if return_labels:
        return G, s, Cbar, labels
    else:
        return G, s, Cbar

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


def mean_M_cross_within_pairs(M: Array, Cbar: Array) -> Tuple[float, float]:
    """
    Average off-diagonal entries of ``M`` on within-group vs cross-group pairs,
    where groups follow the sign pattern of ``Cbar`` (e.g. ``Cbar = outer(l, l)`` with
    ``l in {-1, +1}`` gives ``C_ij = +1`` within a group and ``-1`` across groups).

    For ``polarization``-style ``Cbar`` (all +1 off-diagonals), every pair is treated
    as within-group and the cross-group mean is ``nan``.
    """
    M = np.asarray(M, dtype=np.float64)
    Cbar = np.asarray(Cbar, dtype=np.float64)
    n = M.shape[0]
    if n < 2:
        return float("nan"), float("nan")
    iu = np.triu_indices(n, k=1)
    cij = Cbar[iu]
    mij = M[iu]
    cross = mij[cij < -1e-9]
    within = mij[cij > 1e-9]
    mean_cross = float(np.mean(cross)) if cross.size else float("nan")
    mean_within = float(np.mean(within)) if within.size else float("nan")
    return mean_cross, mean_within


def edge_inter_intra_bucket(u: int, v: int, Cbar: Array) -> str:
    """Return ``'inter'`` or ``'intra'`` for edge ``(u, v)`` using the sign of ``Cbar[u, v]``."""
    c = float(Cbar[int(u), int(v)])
    if c < -1e-9:
        return "inter"
    return "intra"

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

def project_psd(A: np.ndarray, eps: Optional[float] = None) -> np.ndarray:
    """
    Projection onto the PSD cone.
    If ``eps`` is None, eigenvalues are clipped to >= 0.
    If ``eps`` is set, ``A`` is symmetrized first and eigenvalues clipped to >= ``eps``.
    """
    A = np.asarray(A, dtype=np.float64)
    if eps is not None:
        A = 0.5 * (A + A.T)
        w, V = eigh(A)
        w = np.maximum(w, eps)
        return (V * w) @ V.T
    w, U = np.linalg.eigh(A)
    w_clipped = np.maximum(w, 0.0)
    return (U * w_clipped) @ U.T


def enforce_unit_diagonal(C):
    C = C.copy()
    np.fill_diagonal(C, 1.0)
    return C

def project_box_constraint(C: np.ndarray, Cbar: np.ndarray, limit: float = 1.0) -> np.ndarray:
    # project non diagonal elements of C onto the box [-rho, rho]
    M = C - Cbar
    M = np.clip(M, -limit, limit)
    return Cbar + M


def project_to_correlation(A: np.ndarray, n_iters: int = 20, eps: float = 1e-10) -> np.ndarray:
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


def project_onto_U(C: np.ndarray, Cbar: np.ndarray, rho: float, n_rounds: int = 5, norm_constraint: str = 'classifier_error') -> np.ndarray:

    X = C.copy()
   
   
    return X

def worst_case_C_for_fixed_L(
    L: np.ndarray,
    X: np.ndarray,
    Cbar: np.ndarray,
    rho: float,
    C_prev: Optional[np.ndarray] = None,
    s: Optional[np.ndarray] = None,
    n_steps: int = 5,
) -> Tuple[np.ndarray, float]:
    n = L.shape[0]
    q = 4 * rho * (1 - rho)
    start_time = time.time()

    if C_prev is None:
        C = Cbar.copy()
    else:
        C = C_prev.copy()


    if s is not None:
        v = s.copy()
        lam = s.T @ (X * C) @ s
    else:
        lam, v = top_eigenpair(X * C)

        v = v.copy()

    C_init = C.copy()

    lam_prev = lam

    for step in range(n_steps):
        # Frank-Wolfe: move toward extreme point of box in gradient direction
        G = X * np.outer(v, v)       # gradient w.r.t. C
        C = C + q * np.sign(G)       # full step to box extreme point
        
        # Project back onto U
        C = project_box_constraint(C, Cbar, q)
        C = project_psd(C)
        C = project_box_constraint(C, Cbar, q)  # re-enforce after PSD
        C = enforce_unit_diagonal(C)

        # Recompute eigenpair at new C
        if s is not None:
            lam = s.T @ (X * C) @ s
        else:
            lam, v = top_eigenpair(X * C)


        if np.isclose(lam, lam_prev, rtol=1e-10):
            break

        lam_prev = lam

    eta_time = time.time() - start_time
    
    return C, eta_time


def sketch_solve_helper(A: Union[sp.csr_matrix, np.ndarray], R: np.ndarray) -> np.ndarray:
    n, q = R.shape
    if A.shape[0] != n:
        raise ValueError("A and R must agree in row dimension.")
    U = np.zeros((n, q), dtype=np.float64)
    for k in range(q):
        U[:, k], _ = spla.cg(A, R[:, k], atol=1e-8)
    return U


def sketch_U_sherman_morrison_two_rank(
    U: np.ndarray,
    R: np.ndarray,
    q: int,
    w: float,
    a: np.ndarray,
    c: np.ndarray,
    denom_eps: float = 1e-14,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    u_vec = np.sqrt(w) * np.asarray(a, dtype=np.float64).ravel()
    z_vec = np.sqrt(w) * np.asarray(c, dtype=np.float64).ravel()

    # --- Step 1: First Rank-1 Update (Adding u_vec) ---
    # Approximate X @ u_vec using (U @ R.T @ u) / q
    Xu = (U @ (R.T @ u_vec)) / q
    t_u = U.T @ u_vec

    denom1 = 1.0 + float(u_vec @ Xu)
    if abs(denom1) < denom_eps:
        denom1 = (
            math.copysign(denom_eps, denom1) if denom1 != 0.0 else denom_eps
        )

    # Compute intermediate U1 after the first rank-1 update
    U1 = U - np.outer(Xu, t_u) / denom1

    # --- Step 2: Second Rank-1 Update (Subtracting z_vec) ---
    # Approximate X @ z_vec on the original matrix
    Xz = (U @ (R.T @ z_vec)) / q

    # Apply Sherman-Morrison to find X1_z (the action of intermediate X1 on z_vec)
    X1_z = Xz - Xu * float(u_vec @ Xz) / denom1

    denom2 = 1.0 - float(z_vec @ X1_z)
    if abs(denom2) < denom_eps:
        denom2 = (
            math.copysign(denom_eps, denom2) if denom2 != 0.0 else denom_eps
        )

    # Compute final updated U matrix
    g = U1.T @ z_vec
    U = U1 + np.outer(X1_z, g) / denom2
    X = (U @ R.T) / q
    M = (U @ U.T) / q
    return U, R, X, M


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

def generate_correlation_matrix_scenario(Cbar: np.ndarray, mode: str, **kwargs) -> np.ndarray:
    if mode == 'classifier_error':
        if 'p' in kwargs:
            p = kwargs['p']
        else:
            raise ValueError("p must be provided")
        n = Cbar.shape[0]

        C = Cbar.copy()
        for i in range(n):
            for j in range(n):
                if i != j:
                    C[i, j] = (1 - 2*p)**2 * C[i, j]
    elif mode == 'differential_privacy':
        if 'epsilon' in kwargs:
            epsilon = kwargs['epsilon']
        else:
            raise ValueError("epsilon must be provided")
        p = 1 / (1 + np.exp(epsilon))
        return generate_correlation_matrix_scenario(Cbar, mode='classifier_error', p=p)
    else:
        raise ValueError(f"Invalid mode: {mode}")

    return C


def sketch_solve(A, q=None, seed=None):
    rng = np.random.default_rng(seed)
    n = A.shape[0]

    # Rademacher sketch matrix R
    R = rng.choice([-1.0, 1.0], size=(n, q))

    # Sketch for X = A^{-1}: solve A U_X = R
    U = sketch_solve_helper(A, R)         # shape (n, q)

    X = (U @ R.T) / q
    M = (U @ U.T) / q

    return U, R, X, M
