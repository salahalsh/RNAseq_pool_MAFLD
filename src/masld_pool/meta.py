"""Random-effects meta-analysis and its sensitivity analyses."""
import numpy as np
import pandas as pd

from .config import COHORTS, FDR, K_MIN, LFC_MIN, LFC_STRICT
from .stats import EstimatorRefit, bh, dersimonian_laird


def effect_matrices(de):
    genes = sorted(set().union(*[set(r.index) for r in de.values()]))
    Y = pd.DataFrame({a: de[a].log2FC.reindex(genes) for a in COHORTS}, index=genes)
    S = pd.DataFrame({a: de[a].lfcSE.reindex(genes) for a in COHORTS}, index=genes)
    return Y, S


def signature_rule(M, lfc=LFC_MIN):
    return M[(M.padj < FDR) & (M.pooled_log2FC.abs() >= lfc) & (M.k_cohorts >= K_MIN)]


def pool(Y, S):
    M = dersimonian_laird(Y, S)
    M["padj"] = bh(M.pvalue.values)          # BH over every pooled gene, not the k >= 4 subset
    M.index.name = "ensembl_gene_id"
    return M.sort_values("pvalue")


def run(de):
    Y, S = effect_matrices(de)
    M = pool(Y, S)
    sig = signature_rule(M)
    return {"Y": Y, "S": S, "M": M, "sig": sig, "strict": signature_rule(M, LFC_STRICT)}


def leave_one_cohort_out(Y, S, sig):
    base = set(sig.index)
    rows = []
    for a in COHORTS:
        s2 = signature_rule(pool(Y.drop(columns=a), S.drop(columns=a)))
        both = base & set(s2.index)
        rows.append({"cohort_left_out": a, "signature_size": len(s2),
                     "jaccard_vs_full": len(both) / max(len(base | set(s2.index)), 1),
                     "fraction_retained": len(both) / max(len(base), 1)})
    return pd.DataFrame(rows)


def single_cohort_replication(de, M):
    rows = []
    for a in COHORTS:
        r = de[a]
        top = [g for g in r[r.padj < 0.05].sort_values("pvalue").head(500).index if g in M.index]
        pooled = M.loc[top]
        kept = int(((pooled.padj < FDR) & (pooled.pooled_log2FC.abs() >= LFC_MIN)).sum())
        same = int((np.sign(pooled.pooled_log2FC) == np.sign(r.loc[top, "log2FC"])).sum())
        rows.append({"cohort": a, "top_genes_tested": len(top), "replicated": kept,
                     "replication_rate": kept / max(len(top), 1),
                     "same_direction_rate": same / max(len(top), 1)})
    return pd.DataFrame(rows)


def estimator_sensitivity(Y, S):
    R = EstimatorRefit(Y, S).fit()
    sigs = {k: set(signature_rule(v.assign(k_cohorts=v.k_cohorts)).index) for k, v in R.items()}
    return R, sigs


def expression_floor(de):
    """25th percentile of each gene's median DESeq2 baseMean across the eight per-cohort tables."""
    B = pd.DataFrame({a: de[a].baseMean for a in COHORTS}).median(axis=1)
    return B, float(B.quantile(0.25))


def hksj(Y, S, M):
    """Hartung-Knapp-Sidik-Jonkman sensitivity: same inputs and DL tau^2, pooled variance
    rescaled by the weighted residual mean square and referred to t with k - 1 df."""
    from scipy import stats
    y, v = Y.loc[M.index].values, np.square(S.loc[M.index].values)
    ok = np.isfinite(y) & np.isfinite(v) & (v > 0)
    k = ok.sum(1)
    mu, tau2 = M.pooled_log2FC.values, M.tau2.values
    w = np.where(ok, 1.0 / (np.where(ok, v, 1.0) + tau2[:, None]), 0.0)
    qres = (w * (np.where(ok, y, 0.0) - mu[:, None]) ** 2).sum(1) / np.maximum(k - 1, 1)
    se = np.sqrt(qres / w.sum(1))
    with np.errstate(divide="ignore", invalid="ignore"):
        p = 2 * stats.t.sf(np.abs(mu / se), np.maximum(k - 1, 1))
    p[k < 2] = np.nan
    return pd.DataFrame({"pooled_log2FC": mu, "p_hksj": p, "padj_hksj": bh(p), "k_cohorts": k},
                        index=M.index)
