"""Disparity versus polarization and group-structure statistics on real networks.

For each network we take the nominal group structure (ground-truth labels where
available, spectral bisection otherwise), align the private opinions with the
groups (centred and normalised group indicator), and report polarization,
disparity for several classifier errors rho, the structural disparity
h(L, C(rho)) = lambda_max(M * C(rho)), and group-structure statistics that a
platform could compute without any opinion dynamics.

Outputs (all under figures/): real_network_comparison.csv, real_network_correlations.csv,
experiment_11_real_networks.pdf and experiment_11_real_networks.tex
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse.linalg as spla
import seaborn as sns
from scipy.stats import spearmanr

from utils import configure_plot_style, load_dataset

# (directory, group type, display name, family)
NETWORKS = [
    ("polblogs", "label", "Polblogs", "Political blogs"),
    ("reddit", "spectral", "Reddit", "Discussion (spectral)"),
    ("twitter", "spectral", "Twitter", "Discussion (spectral)"),
    ("twitch-DE", "label", "Twitch-DE", "Twitch (mature)"),
    ("twitch-ENGB", "label", "Twitch-ENGB", "Twitch (mature)"),
    ("twitch-ES", "label", "Twitch-ES", "Twitch (mature)"),
    ("twitch-FR", "label", "Twitch-FR", "Twitch (mature)"),
    ("twitch-PTBR", "label", "Twitch-PTBR", "Twitch (mature)"),
    ("twitch-RU", "label", "Twitch-RU", "Twitch (mature)"),
    ("fb-Caltech36", "label", "FB-Caltech", "Facebook (gender)"),
    ("fb-Swarthmore42", "label", "FB-Swarthmore", "Facebook (gender)"),
]
RHOS = [0.0, 0.1, 0.2]


def analyse(name: str, group_type: str) -> dict:
    G, _, _, labels = load_dataset(name, group_type, return_labels=True)
    n = G.number_of_nodes()
    labels = np.asarray(labels, dtype=float)
    L = nx.laplacian_matrix(G, nodelist=range(n), weight="weight").toarray().astype(float)
    evals, evecs = np.linalg.eigh(L)
    M = (evecs * (1.0 / (1.0 + evals) ** 2)) @ evecs.T
    lam2, lamn = evals[1], evals[-1]

    # opinions aligned with the nominal groups (centred, unit norm)
    s = labels - labels.mean()
    s /= np.linalg.norm(s)
    y = s * labels
    pol = float(s @ M @ s)
    yMy = float(y @ M @ y)
    dM = np.diag(M)
    diag_term = float(np.sum(dM * y ** 2))
    # disagreement of Musco et al.: z^T L z at the FJ equilibrium
    z = (evecs * (1.0 / (1.0 + evals))) @ (evecs.T @ s)
    disagreement = float(z @ L @ z)

    A = nx.to_numpy_array(G, nodelist=range(n), weight="weight")
    deg = A.sum(1)
    m = deg.sum() / 2
    diff = labels[:, None] != labels[None, :]
    cross = A[diff].sum() / 2
    mask_a = labels == 1
    vol_a, vol_b = deg[mask_a].sum(), deg[~mask_a].sum()
    comms = [set(np.where(mask_a)[0]), set(np.where(~mask_a)[0])]
    nx.set_node_attributes(G, {i: int(labels[i]) for i in range(n)}, "grp")

    out = {
        "n": n, "m": int(m), "lambda_2": lam2, "lambda_n": lamn,
        "polarization": pol, "disagreement": disagreement,
        "cross_edge_fraction": cross / m,
        "ei_index": (cross - (m - cross)) / m,
        "modularity": nx.algorithms.community.modularity(G, comms),
        "assortativity": nx.attribute_assortativity_coefficient(G, "grp"),
        "size_imbalance": abs(mask_a.sum() - (~mask_a).sum()) / n,
        "volume_imbalance": abs(vol_a - vol_b) / (2 * m),
        "trM_over_n": float(np.trace(M)) / n,
    }
    for rho in RHOS:
        gamma = (1 - 2 * rho) ** 2
        g = gamma * yMy + (1 - gamma) * diag_term
        out[f"disparity_rho={rho}"] = g
        out[f"ratio_rho={rho}"] = g / pol
        out[f"bound_rho={rho}"] = gamma * (1 + lam2) ** 2
        # structural disparity: lambda_max(gamma * D_r M D_r + (1-gamma) diag(M))
        Mr = gamma * (labels[:, None] * M * labels[None, :]) + (1 - gamma) * np.diag(dM)
        out[f"structural_rho={rho}"] = float(spla.eigsh(Mr, k=1, which="LA")[0][0])
    return out


def run(args):
    out_dir, fig_dir = Path(args.out_dir), Path(args.fig_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    fig_dir.mkdir(exist_ok=True, parents=True)

    rows = []
    if args.from_csv:
        df = pd.read_csv(out_dir / "real_network_comparison.csv")
    for d, gt, disp, fam in ([] if args.from_csv else NETWORKS):
        print(f"[{disp}] ...", flush=True)
        r = analyse(d, gt)
        rows.append({"name": disp, "dataset": d, "family": fam, "labels": "ground truth" if gt == "label" else "spectral", **r})
        print(f"  n={r['n']} m={r['m']} lam2={r['lambda_2']:.3f} pol={r['polarization']:.4f} "
              f"g(0.1)={r['disparity_rho=0.1']:.3f} R(0.1)={r['ratio_rho=0.1']:.2f} "
              f"size_imb={r['size_imbalance']:.2f} vol_imb={r['volume_imbalance']:.2f} Q={r['modularity']:.3f}", flush=True)
    if not args.from_csv:
        df = pd.DataFrame(rows)
        df.to_csv(out_dir / "real_network_comparison.csv", index=False)

    stats = ["polarization", "disagreement", "modularity", "assortativity", "cross_edge_fraction",
             "size_imbalance", "volume_imbalance", "lambda_2"]
    targets = ["disparity_rho=0.1", "disparity_rho=0.2", "ratio_rho=0.1", "ratio_rho=0.2"]
    corr = pd.DataFrame({t: {s_: spearmanr(df[s_], df[t]).statistic for s_ in stats} for t in targets})
    corr.to_csv(out_dir / "real_network_correlations.csv")
    print("\nSpearman correlations:\n", corr.round(2).to_string())

    write_table(df, corr, fig_dir / "experiment_11_real_networks.tex")
    plot(df, fig_dir / "experiment_11_real_networks.pdf")


def write_table(df, corr, path):
    cols = [("n", "$n$", "{:d}"), ("m", "$m$", "{:d}"), ("lambda_2", "$\\lambda_2$", "{:.2f}"),
            ("size_imbalance", "Size imb.", "{:.2f}"), ("volume_imbalance", "Vol. imb.", "{:.2f}"),
            ("modularity", "$Q$", "{:.2f}"), ("assortativity", "Assort.", "{:.2f}"),
            ("cross_edge_fraction", "Cross-edge frac.", "{:.2f}"),
            ("polarization", "$\\mathcal P$", "{:.3f}"), ("disagreement", "Disagr.", "{:.3f}"),
            ("disparity_rho=0.1", "$g(\\rho{=}0.1)$", "{:.3f}"),
            ("ratio_rho=0.1", "$R(\\rho{=}0.1)$", "{:.1f}")]
    lines = ["\\begin{tabular}{ll" + "r" * len(cols) + "}", "\\toprule",
             "Network & Labels & " + " & ".join(c[1] for c in cols) + " \\\\", "\\midrule"]
    for _, r in df.iterrows():
        lines.append(f"{r['name']} & {r['labels']} & " + " & ".join(c[2].format(r[c[0]]) for c in cols) + " \\\\")
    lines += ["\\midrule", "\\multicolumn{2}{l}{Spearman $\\rho_s$ with $g(\\rho{=}0.1)$} & " +
              " & ".join("" if c[0] not in corr.index else f"{corr.loc[c[0], 'disparity_rho=0.1']:.2f}" for c in cols) + " \\\\",
              "\\multicolumn{2}{l}{Spearman $\\rho_s$ with $R(\\rho{=}0.1)$} & " +
              " & ".join("" if c[0] not in corr.index else f"{corr.loc[c[0], 'ratio_rho=0.1']:.2f}" for c in cols) + " \\\\",
              "\\bottomrule", "\\end{tabular}"]
    path.write_text("\n".join(lines))


def plot(df, path):
    configure_plot_style()
    from matplotlib.lines import Line2D
    # the figure is 20in wide and is scaled to the text width (~6.5in) in the paper, so
    # fonts and markers are set ~3x larger than their intended printed size
    FS = 24          # one font size for annotations, legends and colorbar
    MS = 420         # base marker area
    fig, axes = plt.subplots(1, 2, figsize=(20, 7.5))
    fams = list(dict.fromkeys(df["family"]))
    markers = dict(zip(fams, ["o", "s", "^", "D", "P", "X"]))
    # explicit label offsets in points per network, (left panel, right panel); labels that are
    # pushed far from their marker get a thin gray connector line
    off = {"FB-Swarthmore": ((0, -45), (-22, 0)),
           "FB-Caltech":    ((-15, -35), (22, 0)),
           "Twitch-FR":     ((-40, 40), (-10, 40)),
           "Twitch-PTBR":   ((0, -60), (24, -12)),
           "Twitch-ES":     ((22, 6), (-60, 30)),
           "Twitch-DE":     ((0, 40), (0, 45)),
           "Twitch-RU":     ((22, -8), (-45, 24)),
           "Twitch-ENGB":   ((20, 40), (55, -18)),
           "Polblogs":      ((22, -30), (-40, -50)),
           "Twitter":       ((0, 30), (0, 24)),
           "Reddit":        ((-24, 0), (-8, 24))}
    def annotate(ax, r, x, y, k):
        dx, dy = off.get(r["name"], ((16, 16), (16, 16)))[k]
        ha = "left" if dx > 0 else "right" if dx < 0 else "center"
        va = "bottom" if dy > 0 else "top" if dy < 0 else "center"
        far = (dx ** 2 + dy ** 2) ** 0.5 > 26
        ax.annotate(r["name"], (x, y), fontsize=FS, xytext=(dx, dy), textcoords="offset points", ha=ha, va=va,
                    arrowprops=dict(arrowstyle="-", color="gray", lw=1.0, shrinkA=2, shrinkB=12) if far else None)

    # ---- left: disparity vs polarization
    ax = axes[0]
    sc = None
    for fam in fams:
        sub = df[df["family"] == fam]
        sc = ax.scatter(sub["polarization"], sub["disparity_rho=0.1"], c=sub["lambda_2"], cmap="magma",
                        vmin=df["lambda_2"].min(), vmax=df["lambda_2"].max(),
                        s=MS + 900 * sub["size_imbalance"], marker=markers[fam], edgecolor="black", linewidth=1.2)
    for _, r in df.iterrows():
        annotate(ax, r, r["polarization"], r["disparity_rho=0.1"], 0)
    ax.set_xscale("log")
    ax.set_xlim(df["polarization"].min() * 0.5, df["polarization"].max() * 1.8)
    ax.set_ylim(0.02, 0.80)
    ax.set_xlabel("Polarization $\\mathcal{P}(s, L)$")
    ax.set_ylabel("Disparity $g(s, L, C(0.1))$")
    handles = [Line2D([], [], marker=markers[f], linestyle="", markersize=13, markerfacecolor="lightgray",
                      markeredgecolor="black", label=f) for f in fams]
    ax.legend(handles=handles, fontsize=FS - 5, loc="lower right", frameon=True, framealpha=1.0, borderaxespad=0.8, handletextpad=0.4)
    cb = fig.colorbar(sc, ax=ax, pad=0.02)
    cb.set_label("$\\lambda_2$", fontsize=FS + 4)
    cb.ax.tick_params(labelsize=FS)

    # ---- right: ratio vs lambda_2 with the Proposition 1 bound
    ax = axes[1]
    xs = np.linspace(df["lambda_2"].min() * 0.9, df["lambda_2"].max() * 1.1, 200)
    cols = sns.color_palette("magma", 4)[1:3]
    for rho, col in zip([0.1, 0.2], cols):
        for fam in fams:
            sub = df[df["family"] == fam]
            ax.scatter(sub["lambda_2"], sub[f"ratio_rho={rho}"], color=col, marker=markers[fam],
                       edgecolor="black", linewidth=1.2, s=MS)
        ax.plot(xs, (1 - 2 * rho) ** 2 * (1 + xs) ** 2, linestyle="--", color=col, linewidth=2.5)
    for _, r in df.iterrows():
        annotate(ax, r, r["lambda_2"], r["ratio_rho=0.1"], 1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(df["lambda_2"].min() * 0.5, df["lambda_2"].max() * 1.3)
    ax.set_xlabel("Algebraic connectivity $\\lambda_2$")
    ax.set_ylabel("Ratio $R$ (disparity / polarization)")
    handles = [Line2D([], [], marker="o", linestyle="--", color=col, markerfacecolor=col, markeredgecolor="black",
                      markersize=16, label=f"$R$ and bound, $\\rho={rho}$") for rho, col in zip([0.1, 0.2], cols)]
    ax.legend(handles=handles, fontsize=FS - 5, loc="lower right", frameon=True, framealpha=1.0, borderaxespad=0.8, handletextpad=0.4)

    for ax in axes:
        ax.tick_params(labelsize=FS)
        ax.xaxis.label.set_size(FS + 4)
        ax.yaxis.label.set_size(FS + 4)
    sns.despine(fig)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", dest="out_dir", default="figures")
    ap.add_argument("--fig-dir", dest="fig_dir", default="figures")
    ap.add_argument("--from-csv", dest="from_csv", action="store_true", help="re-plot from the saved CSV")
    run(ap.parse_args())
