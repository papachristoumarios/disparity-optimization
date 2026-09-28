"""Controlled sweep on a two-block stochastic block model (SBM).

Sweep 1 (mixing): balanced blocks, vary p_out / p_in from near-disconnected
communities to an Erdos-Renyi graph. Reports polarization, disparity for
several classifier errors rho, their ratio, and the Proposition 1 bound.

Sweep 2 (imbalance): fixed mixing ratio, vary the size of group A from 50% to
90% of the users. Reports the same metrics together with group-size and
group-volume imbalance statistics.

Outputs (all under figures/): sbm_sweep_mixing.csv, sbm_sweep_imbalance.csv,
experiment_10_sbm_sweep.pdf and experiment_10_sbm_sweep_main.pdf
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns

from utils import configure_plot_style


def two_block_sbm(n_a: int, n_b: int, p_in: float, p_out: float, seed: int):
    rng = np.random.default_rng(seed)
    n = n_a + n_b
    labels = np.r_[np.ones(n_a), -np.ones(n_b)]
    iu, ju = np.triu_indices(n, k=1)
    same = labels[iu] == labels[ju]
    probs = np.where(same, p_in, p_out)
    keep = rng.random(len(iu)) < probs
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(zip(iu[keep], ju[keep]))
    # keep the largest connected component so that lambda_2 > 0
    cc = max(nx.connected_components(G), key=len)
    G = G.subgraph(cc).copy()
    nodes = sorted(G.nodes())
    return nx.relabel_nodes(G, {v: i for i, v in enumerate(nodes)}), labels[nodes]


def metrics(G: nx.Graph, labels: np.ndarray, rhos) -> dict:
    n = G.number_of_nodes()
    L = nx.laplacian_matrix(G, nodelist=range(n)).toarray().astype(float)
    evals, evecs = np.linalg.eigh(L)
    M = evecs @ np.diag(1.0 / (1.0 + evals) ** 2) @ evecs.T
    lam2 = evals[1]

    # private opinions aligned with the nominal groups, centred and normalised
    s = labels - labels.mean()
    s = s / np.linalg.norm(s)
    y = s * labels

    pol = float(s @ M @ s)
    yMy = float(y @ M @ y)
    diag_term = float(np.sum(np.diag(M) * y ** 2))

    A = nx.to_numpy_array(G, nodelist=range(n))
    deg = A.sum(1)
    m = deg.sum() / 2
    cross = A[labels[:, None] != labels[None, :]].sum() / 2
    mask_a = labels == 1
    vol_a, vol_b = deg[mask_a].sum(), deg[~mask_a].sum()
    modularity = nx.algorithms.community.modularity(
        G, [set(np.where(mask_a)[0]), set(np.where(~mask_a)[0])])

    out = {
        "n": n,
        "m": m,
        "lambda_2": lam2,
        "polarization": pol,
        "cross_edge_fraction": cross / m,
        "modularity": modularity,
        "size_imbalance": abs(mask_a.sum() - (~mask_a).sum()) / n,
        "volume_imbalance": abs(vol_a - vol_b) / (2 * m),
        "trM_over_n": np.trace(M) / n,
    }
    for rho in rhos:
        gamma = (1 - 2 * rho) ** 2
        # C(rho) = gamma r r^T + (1 - gamma) I  =>  g = gamma y^T M y + (1-gamma) sum_i M_ii y_i^2
        g = gamma * yMy + (1 - gamma) * diag_term
        out[f"disparity_rho={rho}"] = g
        out[f"ratio_rho={rho}"] = g / pol
        out[f"bound_rho={rho}"] = gamma * (1 + lam2) ** 2
    return out


def run(args):
    rhos = [0.0, 0.05, 0.1, 0.2]
    out_dir = Path(args.out_dir)
    fig_dir = Path(args.fig_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    fig_dir.mkdir(exist_ok=True, parents=True)

    n = args.n
    p_in = args.avg_within_degree / (n / 2)

    # ---- Sweep 1: mixing ratio, balanced groups
    ratios = np.logspace(-2, 0, args.num_points)
    rows = []
    for q in ratios:
        for seed in range(args.seeds):
            G, lab = two_block_sbm(n // 2, n // 2, p_in, q * p_in, seed)
            rows.append({"mixing_ratio": q, "seed": seed, **metrics(G, lab, rhos)})
    df_mix = pd.DataFrame(rows)
    df_mix.to_csv(out_dir / "sbm_sweep_mixing.csv", index=False)

    # ---- Sweep 2: group-size imbalance, fixed mixing ratio
    fracs = np.linspace(0.5, 0.9, args.num_points)
    rows = []
    for f in fracs:
        n_a = int(round(f * n))
        for seed in range(args.seeds):
            G, lab = two_block_sbm(n_a, n - n_a, p_in, args.imbalance_mixing * p_in, seed)
            rows.append({"frac_a": f, "seed": seed, **metrics(G, lab, rhos)})
    df_imb = pd.DataFrame(rows)
    df_imb.to_csv(out_dir / "sbm_sweep_imbalance.csv", index=False)

    plot(df_mix, df_imb, rhos, fig_dir / "experiment_10_sbm_sweep.pdf")
    summarize(df_mix, df_imb, rhos)


def _long(df, xcol, rhos):
    recs = []
    for _, r in df.iterrows():
        recs.append({xcol: r[xcol], "seed": r["seed"], "Metric": "Polarization $\\mathcal{P}$", "Value": r["polarization"]})
        for rho in rhos:
            recs.append({xcol: r[xcol], "seed": r["seed"], "Metric": f"Disparity ($\\rho={rho}$)", "Value": r[f"disparity_rho={rho}"]})
    return pd.DataFrame(recs)


def plot(df_mix, df_imb, rhos, path):
    configure_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    palette = ["black"] + list(sns.color_palette("magma", len(rhos) + 1))[:len(rhos)]

    # (i) metrics vs mixing ratio
    ax = axes[0]
    sns.lineplot(data=_long(df_mix, "mixing_ratio", rhos), x="mixing_ratio", y="Value",
                 hue="Metric", palette=palette, marker="o", ax=ax, errorbar="sd")
    ax.lines[0].set_linestyle("--")
    ax.set_xscale("log")
    ax.set_xlabel("Mixing ratio $p_{\\mathrm{out}} / p_{\\mathrm{in}}$")
    ax.set_ylabel("Value")
    ax.set_title("Balanced groups: metrics vs. connectivity")
    ax.legend(fontsize=10, title=None)

    # (ii) ratio and bound vs mixing ratio
    ax = axes[1]
    for k, rho in enumerate(rhos[1:], start=1):
        g = df_mix.groupby("mixing_ratio")
        ax.plot(g[f"ratio_rho={rho}"].mean().index, g[f"ratio_rho={rho}"].mean().values,
                marker="o", color=palette[k], label=f"$R$ ($\\rho={rho}$)")
        ax.plot(g[f"bound_rho={rho}"].mean().index, g[f"bound_rho={rho}"].mean().values,
                linestyle="--", color=palette[k], label=f"Bound ($\\rho={rho}$)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Mixing ratio $p_{\\mathrm{out}} / p_{\\mathrm{in}}$")
    ax.set_ylabel("Disparity / polarization")
    ax.set_title("Ratio $R$ and Proposition 1 bound")
    ax.legend(fontsize=9, ncol=2)

    # (iii) metrics vs group-size imbalance
    ax = axes[2]
    sns.lineplot(data=_long(df_imb, "frac_a", rhos), x="frac_a", y="Value",
                 hue="Metric", palette=palette, marker="o", ax=ax, errorbar="sd", legend=False)
    ax.lines[0].set_linestyle("--")
    ax2 = ax.twinx()
    g = df_imb.groupby("frac_a")
    ax2.plot(g["size_imbalance"].mean().index, g["size_imbalance"].mean().values, color="gray", linestyle=":", label="Size imbalance")
    ax2.plot(g["volume_imbalance"].mean().index, g["volume_imbalance"].mean().values, color="gray", linestyle="-.", label="Volume imbalance")
    ax2.set_ylabel("Imbalance", color="gray")
    ax2.legend(fontsize=9, loc="upper left", bbox_to_anchor=(0.02, 0.62))
    ax.set_xlabel("Fraction of users in group $A$")
    ax.set_ylabel("Value")
    ax.set_title("Fixed connectivity: metrics vs. imbalance")

    sns.despine(fig, right=False)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # stand-alone version of panel (i) for the main text
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    sns.lineplot(data=_long(df_mix, "mixing_ratio", rhos), x="mixing_ratio", y="Value",
                 hue="Metric", palette=palette, marker="o", ax=ax, errorbar="sd")
    ax.lines[0].set_linestyle("--")
    ax.set_xscale("log")
    ax.set_xlabel("Mixing ratio $p_{\\mathrm{out}} / p_{\\mathrm{in}}$")
    ax.set_ylabel("Value")
    ax.legend(fontsize=10, title=None)
    sns.despine(fig)
    fig.tight_layout()
    fig.savefig(str(path).replace(".pdf", "_main.pdf"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def summarize(df_mix, df_imb, rhos):
    pd.set_option("display.width", 200)
    cols = ["lambda_2", "polarization", "cross_edge_fraction", "modularity"] + \
           [f"disparity_rho={r}" for r in rhos] + [f"ratio_rho={r}" for r in rhos[1:]] + [f"bound_rho={r}" for r in rhos[1:]]
    print("=== Sweep 1: mixing ratio (balanced) ===")
    print(df_mix.groupby("mixing_ratio")[cols].mean().round(4).to_string())
    cols = ["lambda_2", "polarization", "size_imbalance", "volume_imbalance", "modularity"] + \
           [f"disparity_rho={r}" for r in rhos] + [f"ratio_rho={r}" for r in rhos[1:]]
    print("\n=== Sweep 2: group-size imbalance (fixed mixing) ===")
    print(df_imb.groupby("frac_a")[cols].mean().round(4).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--avg_within_degree", type=float, default=10.0)
    ap.add_argument("--imbalance_mixing", type=float, default=0.1)
    ap.add_argument("--num_points", type=int, default=9)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out-dir", dest="out_dir", default="figures")
    ap.add_argument("--fig-dir", dest="fig_dir", default="figures")
    run(ap.parse_args())
