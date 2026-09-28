"""Readers for the shipped per-cohort files."""
import pandas as pd

from .config import COHORT_DIR, REF_DIR


def metadata(acc):
    return pd.read_csv(COHORT_DIR / acc / f"{acc}_metadata.csv", low_memory=False)


def counts_raw(acc):
    """Deposited matrix: gene_id_as_deposited, ensembl_gene_id, then one column per GSM."""
    return pd.read_csv(COHORT_DIR / acc / f"{acc}_counts.csv.gz", low_memory=False)


def counts_by_ensembl(acc):
    """Deposited values on the harmonised gene namespace.

    Identifiers that could not be mapped to Ensembl are dropped; several deposited
    identifiers mapping to one Ensembl gene are summed, as in the paper."""
    d = counts_raw(acc)
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
