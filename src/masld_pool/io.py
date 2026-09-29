"""Readers for the shipped per-cohort files."""
import numpy as np
import pandas as pd

from .config import COHORT_DIR, RECOVERED_COUNT_COHORTS, REF_DIR


def metadata(acc):
    return pd.read_csv(COHORT_DIR / acc / f"{acc}_metadata.csv", low_memory=False)


def counts_raw(acc):
    """Deposited matrix: gene_id_as_deposited, ensembl_gene_id, then one column per GSM."""
    return pd.read_csv(COHORT_DIR / acc / f"{acc}_counts.csv.gz", low_memory=False)


def recover_counts(logmat, tol=0.02):
    """Raw integer counts from a deposited log2(DESeq2-normalised count + 1) matrix.

    After back-transformation each sample's smallest positive value is the normalised value of
    a raw count of one, i.e. 1 / size factor, so rescaling by it returns the raw counts. Two
    checks, both of which must hold: every recovered value is an integer (to `tol`), and the
    DESeq2 median-of-ratios size factors computed on the recovered counts equal the inferred
    ones."""
    norm = np.power(2.0, logmat.astype(float)) - 1.0
    out, inferred = {}, {}
    for g in norm.columns:
        v = norm[g]
        inv = 1.0 / v[v > 1e-9].min()
        raw = v * inv
        frac = float((np.abs(raw - np.round(raw)) < tol).mean())
        assert frac == 1.0, "%s: only %.4f of values are integer after rescaling" % (g, frac)
        out[g], inferred[g] = np.round(raw).astype(np.int64), inv
    C = pd.DataFrame(out, index=logmat.index)
    pos = C[(C > 0).all(axis=1)]
    lg = np.log(pos)
    mor = np.exp(lg.sub(lg.mean(axis=1), axis=0).median())
    rel = float((mor / pd.Series(inferred) - 1).abs().max())
    assert rel < 1e-3, "median-of-ratios size factors differ from the inferred ones (%.3g)" % rel
    return C


def counts_by_ensembl(acc):
    """Deposited values on the harmonised gene namespace.

    Identifiers that could not be mapped to Ensembl are dropped; several deposited
    identifiers mapping to one Ensembl gene are summed, as in the paper. For GSE183229 the
    counts are recovered from the deposited log-scale matrix first, row by row as deposited."""
    d = counts_raw(acc)
    if acc in RECOVERED_COUNT_COHORTS:
        gsm = [c for c in d.columns if c not in ("gene_id_as_deposited", "ensembl_gene_id")]
        d[gsm] = recover_counts(d[gsm])
    d = d[d.ensembl_gene_id.notna() & (d.ensembl_gene_id.astype(str) != "")]
    d = d.drop(columns=["gene_id_as_deposited"])
    return d.groupby("ensembl_gene_id").sum(numeric_only=True)


def de_reference(acc):
    """The per-cohort differential-expression table exactly as it entered the meta-analysis."""
    return pd.read_csv(COHORT_DIR / acc / f"{acc}_DE.csv.gz", index_col="ensembl_gene_id")


def reference_list(name):
    return list(pd.read_csv(REF_DIR / name, sep="\t").ensembl_gene_id)


def symbols():
    s = pd.read_csv(REF_DIR / "gene_symbols_hgnc.tsv", sep="\t").dropna()
    return dict(zip(s.ensembl_gene_id, s.symbol))


def sample_order():
    return pd.read_csv(REF_DIR / "sample_order.tsv", sep="\t")
