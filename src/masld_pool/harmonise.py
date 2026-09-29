"""Build the cross-cohort matrices from the per-cohort files.

counts   integer counts of the eight cohorts on the genes they all share (15,800). For
         GSE183229 the counts are recovered exactly from its deposited log2(normalised + 1)
         matrix (io.recover_counts). Used for per-cohort DESeq2, the severity strata and the PCA.
logexpr  log2 CPM on the same 15,800 genes (library size over those genes), and its
         per-cohort z-scored copy used by the classifier.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import io
from .config import COHORTS, RECOVERED_COUNT_COHORTS


def detect_scale(mat):
    """Scale is read from the data, never from a file name."""
    v = mat.values
    frac = float((v % 1 != 0).mean())
    if np.nanmax(v) < 30 and frac > 0.3:
        return "log-expression"
    sums = mat.sum(axis=0)
    if float(sums.std() / sums.mean()) < 0.01:
        return "normalised"
    return "counts"


@dataclass
class Harmonised:
    meta: pd.DataFrame          # one row per primary sample, in matrix order
    counts: pd.DataFrame        # genes x samples, eight cohorts
    logexpr: pd.DataFrame       # genes x samples, log2 CPM
    z: pd.DataFrame             # per-cohort z-scored logexpr
    scale: dict                 # scale detected on the deposited file
    genes_shared_counts: int


def build(log=print):
    order = io.sample_order()
    metas, mats, scale = [], {}, {}
    for acc in COHORTS:
        m = io.metadata(acc)
        m = m[m.analysis_set == "primary"]
        gsm = list(order[order.series_accession == acc].gsm)
        raw = io.counts_raw(acc)
        scale[acc] = detect_scale(raw[gsm])
        d = io.counts_by_ensembl(acc)[gsm]
        assert detect_scale(d) == "counts", acc
        mats[acc] = d
        metas.append(m.set_index("gsm").loc[d.columns].rename_axis("gsm").reset_index())
        log("  %-10s %4d samples  %6d genes  deposited scale: %-14s%s"
            % (acc, d.shape[1], d.shape[0], scale[acc],
               " -> integer counts recovered exactly" if acc in RECOVERED_COUNT_COHORTS else ""))
    assert {a for a, s in scale.items() if s != "counts"} == RECOVERED_COUNT_COHORTS
    meta = pd.concat(metas, ignore_index=True)

    shared = sorted(set.intersection(*[set(mats[a].index) for a in COHORTS]))
    assert shared == sorted(io.reference_list("gene_universe_counts.tsv")), \
        "count-gene intersection differs from the reference universe"
    counts = pd.concat([mats[a].loc[shared] for a in COHORTS], axis=1)

    parts = []
    for a in COHORTS:
        d = mats[a].loc[shared]
        lib = d.sum(axis=0).replace(0, np.nan)
        parts.append(np.log2(d.div(lib, axis=1) * 1e6 + 1))
    logexpr = pd.concat(parts, axis=1)
    z = logexpr.copy()
    for a in COHORTS:
        cols = list(mats[a].columns)
        blk = z[cols]
        z[cols] = blk.sub(blk.mean(axis=1), axis=0).div(blk.std(axis=1).replace(0, np.nan), axis=0)
    log("  count matrix %d genes x %d samples | log2 CPM matrix %d genes x %d samples"
        % (counts.shape[0], counts.shape[1], logexpr.shape[0], logexpr.shape[1]))
    return Harmonised(meta, counts, logexpr, z, scale, len(shared))
