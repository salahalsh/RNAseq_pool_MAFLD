"""Build the two cross-cohort matrices from the per-cohort files.

counts   deposited counts of the seven count-bearing cohorts, on the genes they all share
         (15,896). Used for per-cohort DESeq2, the severity strata and the PCA.
logexpr  log2 CPM for the count cohorts plus GSE183229's deposited log-scale values, on the
         15,799-gene universe, and its per-cohort z-scored copy used by the classifier.

The 15,799-gene universe is the intersection over all nine screened-in cohorts, including
GSE334651, which is held out of the primary analysis and not distributed here; it is read
from data/reference so that library sizes, and therefore log2 CPM, are reproduced exactly.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import io
from .config import COHORTS, LOG_SCALE_COHORTS


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
    counts: pd.DataFrame        # genes x samples, seven count cohorts
    logexpr: pd.DataFrame       # genes x samples, eight cohorts
    z: pd.DataFrame             # per-cohort z-scored logexpr
    scale: dict
    genes_shared_counts: int


def build(log=print):
    order = io.sample_order()
    metas, mats, scale = [], {}, {}
    for acc in COHORTS:
        m = io.metadata(acc)
        m = m[m.analysis_set == "primary"]
        d = io.counts_by_ensembl(acc)
        d = d[[g for g in order[order.series_accession == acc].gsm]]
        scale[acc] = detect_scale(d)
        mats[acc] = d
        metas.append(m.set_index("gsm").loc[d.columns].rename_axis("gsm").reset_index())
        log("  %-10s %4d samples  %6d genes  scale: %s"
            % (acc, d.shape[1], d.shape[0], scale[acc]))
    assert {a for a, s in scale.items() if s != "counts"} == LOG_SCALE_COHORTS
    meta = pd.concat(metas, ignore_index=True)

    count_cohorts = [a for a in COHORTS if scale[a] == "counts"]
    shared = sorted(set.intersection(*[set(mats[a].index) for a in count_cohorts]))
    ref = io.reference_list("gene_universe_counts.tsv")
    assert shared == sorted(ref), "count-gene intersection differs from the reference universe"
    counts = pd.concat([mats[a].loc[shared] for a in count_cohorts], axis=1)

    universe = sorted(io.reference_list("gene_universe_logexpr.tsv"))
    parts = []
    for a in COHORTS:
        d = mats[a].loc[universe]
        if scale[a] == "counts":
            lib = d.sum(axis=0).replace(0, np.nan)
            d = np.log2(d.div(lib, axis=1) * 1e6 + 1)
        parts.append(d)
    logexpr = pd.concat(parts, axis=1)
    z = logexpr.copy()
    for a in COHORTS:
        cols = list(mats[a].columns)
        blk = z[cols]
        z[cols] = blk.sub(blk.mean(axis=1), axis=0).div(blk.std(axis=1).replace(0, np.nan), axis=0)
    log("  count matrix %d genes x %d samples | log-expression matrix %d genes x %d samples"
        % (counts.shape[0], counts.shape[1], logexpr.shape[0], logexpr.shape[1]))
    return Harmonised(meta, counts, logexpr, z, scale, len(shared))
