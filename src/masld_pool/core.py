"""Gene prioritisation by agreement across three evidence lines.

Lines: (1) membership of the pooled signature, above the expression floor; (2) rank within
the top 200 SHAP features of the final classifier; (3) STRING degree >= 5 in the network of
the top 500 pooled genes (a dated snapshot of the STRING query, data/reference), above the
expression floor. Tier 1 = all three lines, Tier 2 = two, Tier 3 = one. The validation panel
is the four Tier 1 genes plus the eight Tier 2 genes with the smallest pooled adjusted p.
"""
import pandas as pd

from .config import HUB_MIN_DEGREE, REF_DIR, SHAP_TOP


def prioritise(M, sig, shap_imp, B, floor):
    passes = lambda g: bool(B.get(g, float("nan")) >= floor)  # noqa: E731
    core_pass = {g for g in sig.index if passes(g)}
    shap_top = set(shap_imp.head(SHAP_TOP).ensembl_gene_id)
    hub = pd.read_csv(REF_DIR / "string_top500_hub_degree.tsv", sep="\t")
    hubs = hub[(hub.degree >= HUB_MIN_DEGREE) & hub.ensembl_gene_id.map(lambda g: passes(g))]
    hub_ens = set(hubs.ensembl_gene_id.dropna())
    sig_all = set(sig.index)
    univ = sorted(core_pass | (shap_top & (sig_all | hub_ens)) | (hub_ens & (sig_all | shap_top)))
    rows = []
    for g in univ:
        n = int(g in core_pass) + int(g in shap_top) + int(g in hub_ens)
        rows.append({"ensembl_gene_id": g, "n_lines": n, "tier": {3: 1, 2: 2, 1: 3}[n],
                     "in_signature": g in sig_all, "signature_line": g in core_pass,
                     "shap_line": g in shap_top, "network_line": g in hub_ens,
                     "pooled_log2FC": M.pooled_log2FC.get(g), "pooled_padj": M.padj.get(g),
                     "k_cohorts": M.k_cohorts.get(g)})
    C = pd.DataFrame(rows).sort_values(["n_lines", "pooled_padj"], ascending=[False, True])
    panel = C[C.n_lines >= 2].head(12)
    return C, panel, len(core_pass)
