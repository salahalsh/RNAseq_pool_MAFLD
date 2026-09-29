"""Per-cohort differential expression, case versus control, no covariates.

All eight cohorts: DESeq2 as implemented in PyDESeq2 (Wald test, Cook's refitting, no LFC
shrinkage, so the standard error stays usable for inverse-variance pooling). GSE183229 enters
on the integer counts recovered from its deposited matrix (io.recover_counts).
"""
import warnings

import pandas as pd

from .config import MIN_COUNT
from .stats import bh

warnings.filterwarnings("ignore")


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
    c = count_filter(H.counts[list(m.gsm)], min(n0, n1))
    r = deseq2(c, m.group_class.values)
    method = "DESeq2 (PyDESeq2), Wald, ~group_class"
    r.index.name = "ensembl_gene_id"
    r["n_control"], r["n_case"], r["method"] = n0, n1, method
    return r[["baseMean", "log2FC", "lfcSE", "stat", "pvalue", "padj",
              "n_control", "n_case", "method"]]
