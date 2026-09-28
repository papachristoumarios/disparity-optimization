"""Convert Facebook100 .mat files into the repository's edges.txt / opinions.txt format.

For each school the nominal groups are the two reported genders (local_info column 1:
1 -> +1, 2 -> -1). Users with no reported gender (0) are dropped, together with their
edges. Node ids are the original 0-based row indices of the adjacency matrix.

Run from the repository root:
    python data/facebook100/preprocess.py
which writes data/fb-<School>/edges.txt and data/fb-<School>/opinions.txt.

Source: Traud, Mucha, and Porter (2012), "Social structure of Facebook networks",
Physica A 391(16). The .mat files in this directory are the original release.
"""
import os

import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse as sp

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.dirname(HERE)
SCHOOLS = ["Caltech36", "Swarthmore42"]

for school in SCHOOLS:
    mat = scipy.io.loadmat(os.path.join(HERE, f"{school}.mat"))
    A = sp.triu(sp.csr_matrix(mat["A"]), k=1).tocoo()
    gender = mat["local_info"][:, 1]
    keep = gender != 0

    out = os.path.join(DATA, f"fb-{school}")
    os.makedirs(out, exist_ok=True)

    edges = pd.DataFrame({"u": A.row, "v": A.col})
    edges = edges[keep[edges.u] & keep[edges.v]]
    edges.to_csv(os.path.join(out, "edges.txt"), sep="\t", header=False, index=False)

    ids = np.where(keep)[0]
    labels = np.where(gender[ids] == 1, 1, -1)
    pd.DataFrame({"n": ids, "l": labels}).to_csv(
        os.path.join(out, "opinions.txt"), sep="\t", header=False, index=False)

    print(f"{school}: {keep.sum()} users, {len(edges)} edges, "
          f"group sizes {int((labels == 1).sum())} / {int((labels == -1).sum())}")
