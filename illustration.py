from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import networkx as nx
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import seaborn as sns
import argparse
from utils import *

color_a, color_b = "#2c7fb8", "#e34a33"

# set scaling of font from sns
# sns.set(font_scale=1.5)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default=['polblogs'], type=str, required=True)
    parser.add_argument('--group_type', type=str, choices=['spectral', 'random', 'label'], required=True)
    parser.add_argument("--out-dir", type=str, default='figures')
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


# ----------------------------
# Linear-algebra helpers
# ----------------------------
def graph_to_laplacian(G: nx.Graph, nodes: list[str]) -> sp.csr_matrix:
    """
    Returns L = D - W as a sparse matrix in the node order given by nodes.
    """
    W = nx.to_scipy_sparse_array(G, nodelist=nodes, weight="weight", format="csr", dtype=float)
    deg = np.asarray(W.sum(axis=1)).ravel()
    L = sp.diags(deg, format="csr") - W
    return L


def fj_solver(L: sp.spmatrix):
    """
    Build a fast solver for (I + L)x = b.
    Uses sparse factorization once and reuses it.
    """
    n = L.shape[0]
    A = sp.eye(n, format="csc") + L.tocsc()
    solve = spla.factorized(A)  # returns a callable
    return solve


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


# ----------------------------
# Metric computations
# ----------------------------
def consensus(solve, s: np.ndarray) -> np.ndarray:
    return solve(s)


def compute_disparity(zA: np.ndarray, zB: np.ndarray) -> float:
    return float(np.linalg.norm(zA - zB, ord=2) ** 2)


def compute_polarization(solve, s: np.ndarray) -> float:
    z = consensus(solve, s)
    z_mean = z.mean()
    return float((z - z_mean) @ (z - z_mean))


# ----------------------------
# Plot 0: network “ridiculogram” (force-directed hairball)
# ----------------------------
def plot_network_ridiculogram(
    G: nx.Graph,
    nodes: list[str],
    labels: np.ndarray,
    outpath: str,
    polarization: float,
    disparity: float,
    name: str,
    seed: int = 1,
) -> None:
    """
    Classic spring-layout hairball: fine for dense graphs where a readable
    structural layout is intentionally sacrificed for a quick global view.
    """
    
    lab_by = {n: int(lab) for n, lab in zip(nodes, labels)}
    node_colors = [color_a if lab_by[v] == 1 else color_b for v in G.nodes()]

    n = G.number_of_nodes()
    m = G.number_of_edges()
    k = 2.0 / np.sqrt(max(n, 1))
    pos = nx.spring_layout(G, k=k, iterations=100, seed=seed)

    # x position, y position -> y position, x position
    pos = {v: (pos[v][1], pos[v][0]) for v in pos}

    fig, ax = plt.subplots(figsize=(5, 5), facecolor="white")
    edge_alpha = float(min(0.35, 900.0 / max(m, 1)))
    nx.draw_networkx_edges(
        G,
        pos,
        ax=ax,
        edge_color="#555555",
        width=0.35,
        alpha=edge_alpha,
        arrows=False,
    )
    node_size = max(12.0, min(45.0, 8000.0 / max(n, 1)))
    nx.draw_networkx_nodes(
        G,
        pos,
        ax=ax,
        nodelist=list(G.nodes()),
        node_color=node_colors,
        node_size=node_size,
        linewidths=0,
        alpha=0.95,
    )
    ax.set_aspect("equal")
    ax.axis("off")
    handles = [
        Patch(facecolor=color_a, edgecolor="none", alpha=0.95, label="Group A"),
        Patch(facecolor=color_b, edgecolor="none", alpha=0.95, label="Group B"),
    ]
    ax.legend(handles=handles, frameon=False, loc="upper left", fontsize=10)
    # ax.set_title(f"{name.capitalize()} Network ($n = {n}$, $m = {m}$)", fontsize=13, pad=8)

    # stats = (
    #     r"Polarization: " + f"{polarization:.1g}\n"
    #     + r"Disparity: " + f"{disparity:.1g}"
    # )
    # ax.text(
    #     0.98,
    #     0.02,
    #     stats,
    #     transform=ax.transAxes,
    #     fontsize=9,
    #     verticalalignment="bottom",
    #     horizontalalignment="right",
    #     bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="0.75", alpha=0.92),
    # )

    fig.tight_layout()
    fig.savefig(outpath, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ----------------------------
# Plot 1: counterfactual distributions
# ----------------------------
def plot_counterfactual_distributions(
    z_full: np.ndarray,
    zA: np.ndarray,
    zB: np.ndarray,
    labels: np.ndarray,
    outpath: str,
    name: str,
) -> None:
    """
    Vertical bar chart of z_A - z_B with line overlay z, nodes ordered by
    spectral group then by full consensus for a tidy layout; bar facecolor
    matches group membership.
    """
    # Primary: group (+1 before -1); secondary: full consensus (within group)
    order = np.lexsort((z_full, -labels))
    z_full_s = z_full[order]
    zA_s = zA[order]
    zB_s = zB[order]
    diff = zA_s - zB_s

    mask_a, mask_b = group_masks(labels)
    mask_a_s = mask_a[order]

    n = len(diff)
    x = np.arange(n, dtype=float)
    color_a, color_b = "#2c7fb8", "#e34a33"
    bar_colors = np.where(mask_a_s, color_a, color_b)

    records = []

    for i in range(n):
        # records.append({
        #     "Node ID": x[i],
        #     "Variable Type" : "Difference in consensus",
        #     "Group": "Group A" if mask_a_s[i] else "Group B",
        #     "Value": diff[i],
        # })

        records.append({
            "Node ID": x[i],
            "Variable Type" : "Full consensus",
            "Group": "Group A" if mask_a_s[i] else "Group B",
            "Value": z_full_s[i],
        })

        records.append({
            "Node ID": x[i],
            "Variable Type" : "Counterfactual consensus",
            "Group": "Group A" if mask_a_s[i] else "Group B",
            "Value": zA_s[i] if mask_a_s[i] else zB_s[i],
        })

    df = pd.DataFrame(records)
    sns.set_palette([color_a, color_b])

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.lineplot(x="Node ID", y="Value", hue="Group", style='Variable Type', data=df, alpha=0.9, linewidth=1.5, ax=ax)


    ax.set_xlabel("")
    ax.set_xticks([])

    n_A = mask_a_s.sum()
    ax.axvline(x=n_A, color="black", linewidth=2)
    ax.axhline(y=0, color="black", linewidth=0.5)

    ax.text(n_A // 2, ax.get_ylim()[1], "Group A", ha="center", va="bottom", fontsize=10)
    ax.text(n_A + (n - n_A) // 2, ax.get_ylim()[1], "Group B", ha="center", va="bottom", fontsize=10)


    sns.despine(fig)

    fig.suptitle(f"{name.capitalize()} Network: Difference in consensus between groups", y=1.02, fontsize=14)
    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")

    sns.despine(fig)
    plt.close(fig)

def calculate_M_fast(L: sp.csr_matrix) -> sp.csr_matrix:
    # M = (I + L)**(-2) 
    # use eigenvalues and eigenvectors to compute M
    eigenvalues, eigenvectors = spla.eigsh(L.toarray(), k=L.shape[0], which='SA')
    M = eigenvectors @ np.diag(1 / (1 + eigenvalues)**2) @ eigenvectors.T
    return sp.csr_matrix(M)

def sigmoid(x: float) -> float:
    return 1 / (1 + np.exp(-x))

def illustration_ratio(out_dir: str) -> None:
    rho_range = np.array([0.2, 0.3, 0.4])

    datasets = [
        ('polblogs', 'spectral'),
        ('twitter', 'spectral'),
        ('reddit', 'spectral')
    ]

    records = []

    for name, group_type in datasets:
        G, s, Cbar = load_dataset(name, group_type)
        L = sparse_laplacian(G)
        M = calculate_M_fast(L)
        M = M.toarray()
        eigenvalues, eigenvectors = spla.eigsh(L.toarray(), k=L.shape[0], which='SA')
        lambda_2 = eigenvalues[1]
        lambda_n = eigenvalues[-1]

        for rho in rho_range:

            C = generate_correlation_matrix_scenario(Cbar, mode='classifier_error', p=rho)
            Z = M * C
            disparity = s.T @ Z @ s
            polarization = s.T @ M @ s
            ratio = polarization / disparity

            lower_bound = (1 - 2 * rho)**2 * (1 + lambda_2)**2
            upper_bound = (1 + lambda_n)**2 * (1 - 4 * rho * (1 - rho) * (1 - 1 / (1 + lambda_n)**2))

            records.append({
                'Name': f'{name} ($\\lambda_2 = {lambda_2:.2f}$, $\\lambda_n = {lambda_n:.2f}$)',
                'Group Type': group_type,
                'Rho': rho,
                'Bound': 'Lower',
                'Value': lower_bound,
            })
            records.append({
                'Name': f'{name} ($\\lambda_2 = {lambda_2:.2f}$, $\\lambda_n = {lambda_n:.2f}$)',
                'Group Type': group_type,
                'Rho': rho,
                'Bound': 'Upper',
                'Value': upper_bound,
            })
            records.append({
                'Name': f'{name} ($\\lambda_2 = {lambda_2:.2f}$, $\\lambda_n = {lambda_n:.2f}$)',
                'Group Type': group_type,
                'Rho': rho,
                'Bound': 'Actual',
                'Value': ratio,
            })

    df = pd.DataFrame(records)
    fig, ax = plt.subplots(figsize=(5, 5))
    sns.set_palette([color_a, color_b])
    df_1 = df[df['Bound'] != 'Actual']
    df_2 = df[df['Bound'] == 'Actual']
    sns.scatterplot(x="Rho", y="Value", data=df_2, ax=ax, color='black', s=100, style='Name', markers=True, marker='x')

    sns.lineplot(x="Rho", y="Value", hue='Bound', style='Name', markers=True, marker='x', data=df_1, ax=ax, color='black', legend=False)
    ax.set_xlabel("Classifier error probability ($\\rho$)", fontsize=14)
    ax.set_ylabel("$R(\\rho)$", fontsize=14)
    ax.set_yscale('log')
    ax.legend(loc='upper left', fontsize=10)
    sns.despine(fig)
    fig.tight_layout()
    fig.savefig(str(Path(out_dir) / f"0_illustration_ratio.pdf"), dpi=300, bbox_inches="tight")
    plt.close(fig)

# ----------------------------
# Main
# ----------------------------
def main(name: str, group_type: str, out_dir: str) -> None:

    G, s, Cbar = load_dataset(name, group_type)

    # Load data
    G = load_weighted_undirected_graph(f'data/{name}/edges.txt')
    G = largest_connected_component(G)

    opinion_map = load_opinions(f'data/{name}/opinions.txt')

    # Nodes that appear both in the giant component and opinion file
    nodes = sorted(
        (str(v) for v in G.nodes() if str(v) in opinion_map),
        key=lambda x: int(x) if x.isdigit() else x,
    )
    if len(nodes) == 0:
        raise ValueError("No nodes with opinions were found in the graph.")

    G = G.subgraph(nodes).copy()
    G = largest_connected_component(G)
    nodes = sorted((str(v) for v in G.nodes()), key=lambda x: int(x) if x.isdigit() else x)

    if group_type == 'spectral':
        labels = spectral_partition_labels(G, nodes)
    elif group_type == 'random':
        labels = np.random.choice([-1, 1], size=len(nodes))
    elif group_type == 'label':
        # if label > 0, then group A, otherwise group B
        labels = [1 if opinion_map[n] > 0 else -1 for n in nodes]
        labels = np.array(labels)
    else:
        raise ValueError(f"Invalid group type: {group_type}")


    # Build Laplacian and solver
    L = graph_to_laplacian(G, nodes)
    solve = fj_solver(L)

    opinions = np.array([opinion_map[n] for n in nodes], dtype=float)
    s = build_opinion_vector(opinions)

    # Group masks and counterfactual opinions
    mask_a, mask_b = group_masks(labels)
    sA = s * mask_a.astype(float)
    sB = s * mask_b.astype(float)

    # Equilibria
    z_full = consensus(solve, s)
    zA = consensus(solve, sA)
    zB = consensus(solve, sB)

    # Metrics
    polarization = compute_polarization(solve, s)
    conditional_disparity = compute_disparity(zA, zB)

    # Plots
    plot_network_ridiculogram(
        G=G,
        nodes=nodes,
        labels=labels,
        outpath=str(Path(out_dir) / f"0_network_ridiculogram_{name}.pdf"),
        polarization=polarization,
        disparity=conditional_disparity,
        name=name,
        seed=0,
    )

    plot_counterfactual_distributions(
        z_full=z_full,
        zA=zA,
        zB=zB,
        labels=labels,
        outpath=str(Path(out_dir) / f"1_counterfactual_distributions_{name}.pdf"),
        name=name,
    )

    rho_range = [0, 0.1, 0.2, 0.3, 0.4, 0.5]

    M = calculate_M_fast(L)
    M = M.toarray()

    records = []

    for rho in rho_range:
        print('Rho = ', rho)
        C = np.zeros((len(nodes), len(nodes)))
        for i in range(len(nodes)):
            for j in range(len(nodes)):
                if i == j:
                    C[i, j] = 1
                else:
                    C[i, j] = (1 - 2 * rho)**2 * labels[i] * labels[j]

        Z = M * C

        noisy_disparity = s.T @ Z @ s
        records.append({
            "Classifier error probability ($\\rho$)": rho,
            "Disparity": noisy_disparity,
        })

        if rho == 1:
            conditional_disparity = noisy_disparity

        
    fig, ax = plt.subplots(figsize=(5, 5))
    df = pd.DataFrame(records)
    sns.set_palette([color_a, color_b])
    sns.barplot(x="Classifier error probability ($\\rho$)", y="Disparity", data=df, ax=ax, color='black')
    ax.set_xlabel("Classifier error probability ($\\rho$)", fontsize=14)
    ax.set_ylabel("Disparity", fontsize=14)
    ax.axhline(y=polarization, color=color_a, linestyle='--', label=f'Polarization = {polarization:.1g}')
    ax.axhline(y=conditional_disparity, color=color_b, linestyle='--', label=f'Conditional Disparity = {conditional_disparity:.1g}')
    ax.legend(loc='upper left', fontsize=10)
    sns.despine(fig)
    # fig.suptitle(f"{name.capitalize()} Network: Noisy disparity", y=1.02, fontsize=14)
    fig.tight_layout()
    fig.savefig(str(Path(out_dir) / f"2_noisy_disparity_{name}.pdf"), dpi=300, bbox_inches="tight")
    plt.close(fig)


    illustration_ratio(out_dir)


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    main(name=args.name, group_type=args.group_type, out_dir=args.out_dir)