"""Per-cohort differential expression, case versus control, no covariates.

Count cohorts: DESeq2 as implemented in PyDESeq2 (Wald test, Cook's refitting, no LFC
shrinkage, so the standard error stays usable for inverse-variance pooling).
Log-scale cohort (GSE183229): gene-wise linear model with limma-style empirical-Bayes
moderation of the residual variance.
"""
import warnings

import numpy as np
import pandas as pd
from scipy import stats

from .config import MIN_COUNT
from .stats import bh

warnings.filterwarnings("ignore")


def squeeze_var(s2, df, trim=0.1):
    """limma squeezeVar: moment-match an F distribution to the gene variances."""
    from scipy.optimize import brentq
    from scipy.special import digamma, polygamma
    s2 = np.asarray(s2, float)
    ok = np.isfinite(s2) & (s2 > 0)
    z = np.log(s2[ok])
    lo, hi = np.quantile(z, [trim, 1 - trim])
    zt = z[(z >= lo) & (z <= hi)]
    ez = zt.mean() + digamma(df / 2) - np.log(df / 2)
    vz = zt.var(ddof=1) - polygamma(1, df / 2)
    if vz <= 0:
        return float(np.exp(ez)), np.inf
    try:
        d0 = 2 * brentq(lambda d: polygamma(1, d / 2) - vz, 1e-6, 1e6)
    except Exception:
        d0 = 4.0
    return float(np.exp(ez + digamma(d0 / 2) - np.log(d0 / 2))), float(d0)


def moderated_lm(mat, y, min_expr=1.0):
    keep = (mat > min_expr).sum(axis=1) >= max(3, int(0.2 * mat.shape[1]))
    M = mat.loc[keep]
    g0, g1 = M.loc[:, y == 0].values, M.loc[:, y == 1].values
    n0, n1 = g0.shape[1], g1.shape[1]
    df = n0 + n1 - 2
    s2 = ((g0.var(1, ddof=1) * (n0 - 1)) + (g1.var(1, ddof=1) * (n1 - 1))) / df
    s0_2, d0 = squeeze_var(s2, df)
    s2_post = (d0 * s0_2 + df * s2) / (d0 + df) if np.isfinite(d0) else np.full_like(s2, s0_2)
    se = np.sqrt(s2_post * (1.0 / n0 + 1.0 / n1))
    lfc = g1.mean(1) - g0.mean(1)
    t = lfc / se
    p = 2 * stats.t.sf(np.abs(t), df + (d0 if np.isfinite(d0) else 0))
    return pd.DataFrame({"baseMean": M.mean(1).values, "log2FC": lfc, "lfcSE": se, "stat": t,
                         "pvalue": p, "padj": bh(p)}, index=M.index)


def deseq2(counts, groups, labels=("control", "case")):
    """DESeq2 Wald test of labels[1] against labels[0]; `groups` aligned to counts columns."""
    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.ds import DeseqStats
    X = counts.T.round().astype(int)
    md = pd.DataFrame({"grp": pd.Categorical(list(groups), categories=list(labels))},
                      index=counts.columns)
    dds = DeseqDataSet(counts=X, metadata=md, design="~grp", refit_cooks=True, quiet=True)
    dds.deseq2()
    st = DeseqStats(dds, contrast=["grp", labels[1], labels[0]], quiet=True)
    st.summary()
    r = st.results_df.copy()
    r["padj"] = bh(r["pvalue"].values)
    return r.rename(columns={"log2FoldChange": "log2FC"})


def count_filter(c, n_small):
    return c.loc[(c >= MIN_COUNT).sum(axis=1) >= n_small]


def per_cohort(H, acc):
    m = H.meta[H.meta.series_accession == acc]
    y = (m.group_class == "case").astype(int).values
    n0, n1 = int((y == 0).sum()), int((y == 1).sum())
    if H.scale[acc] == "counts":
        c = count_filter(H.counts[list(m.gsm)], min(n0, n1))
        r = deseq2(c, m.group_class.values)
        method = "DESeq2 (PyDESeq2), Wald, ~group_class"
    else:
        r = moderated_lm(H.logexpr[list(m.gsm)], y)
        method = "moderated linear model on log-expression (limma-style eBayes)"
    r.index.name = "ensembl_gene_id"
    r["n_control"], r["n_case"], r["method"] = n0, n1, method
    return r[["baseMean", "log2FC", "lfcSE", "stat", "pvalue", "padj",
              "n_control", "n_case", "method"]]
