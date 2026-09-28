"""The two competing explanations for between-cohort heterogeneity: obesity in the
comparator (tested inside GSE126848) and disease severity (fibrosis strata)."""
import re

import numpy as np
import pandas as pd

from . import io
from .de import count_filter, deseq2


def _sig(r):
    return set(r[(r.padj < 0.05) & (r.log2FC.abs() >= 0.5)].index)


def obesity(log=print):
    """Three contrasts inside one cohort, so library preparation and processing are fixed."""
    m = io.metadata("GSE126848")
    raw = io.counts_raw("GSE126848").set_index("gene_id_as_deposited").drop(columns="ensembl_gene_id")
    grp = m.disease_label.astype(str).str.strip().str.lower()
    G = {"healthy": list(m.gsm[grp == "healthy"]), "obese": list(m.gsm[grp == "obese"]),
         "masld": list(m.gsm[grp.isin(["nafld", "nash"])])}
    res = {}
    for name, case, ctrl in (("A_MASLD_vs_lean", "masld", "healthy"),
                             ("B_MASLD_vs_obese", "masld", "obese"),
                             ("C_obese_vs_lean", "obese", "healthy")):
        cols = G[ctrl] + G[case]
        sub = count_filter(raw[cols], min(len(G[ctrl]), len(G[case])))
        res[name] = deseq2(sub, ["ctrl"] * len(G[ctrl]) + ["case"] * len(G[case]), ("ctrl", "case"))
        log("    %-18s %2d vs %2d  tested %5d  DE (BH<0.05, |log2FC|>=0.5) %5d"
            % (name, len(G[case]), len(G[ctrl]), len(res[name]), len(_sig(res[name]))))
    A, B, C = res["A_MASLD_vs_lean"], res["B_MASLD_vs_obese"], res["C_obese_vs_lean"]
    sA, sB, sC = _sig(A), _sig(B), _sig(C)
    testable = [g for g in sA if g in B.index]
    kept = [g for g in testable if g in sB]
    r = float(np.corrcoef(A.loc[testable, "log2FC"], B.loc[testable, "log2FC"])[0, 1])
    return {"n_A": len(sA), "n_B": len(sB), "n_C": len(sC), "testable_in_B": len(testable),
            "untestable_in_B": len(sA) - len(testable), "retained": len(kept),
            "retained_frac": len(kept) / len(testable), "r_A_B_testable": r,
            "groups": {k: len(v) for k, v in G.items()}}


def _num(v):
    m = re.search(r"\d+(?:\.\d+)?", str(v))
    return float(m.group(0)) if m else np.nan


LADDER = {"GSE135251": [("F0-1", [0, 1]), ("F2", [2]), ("F3-4", [3, 4])],
          "GSE162694": [("F0-1", [0, 1]), ("F2", [2]), ("F3-4", [3, 4])],
          "GSE281797": [("F0", [0]), ("F1", [1]), ("F2-3", [2, 3])],
          "GSE185051": [("F0-1", [0, 1]), ("F2-3", [2, 3])]}


def severity(H, sig, log=print):
    """Each fibrosis stratum against the same cohort's own controls."""
    M = H.meta.copy()
    M["fib"] = M.fibrosis_stage.map(_num)
    rows = []
    for acc, strata in LADDER.items():
        sub = M[M.series_accession == acc]
        ctrl = [g for g in sub[sub.group_class == "control"].gsm if g in H.counts.columns]
        for name, stages in strata:
            cases = [g for g in sub[(sub.group_class == "case") & sub.fib.isin(stages)].gsm
                     if g in H.counts.columns]
            if len(cases) < 5 or len(ctrl) < 4:
                continue
            cols = ctrl + cases
            r = deseq2(count_filter(H.counts[cols], min(len(ctrl), len(cases))),
                       ["ctrl"] * len(ctrl) + ["case"] * len(cases), ("ctrl", "case"))
            ms = r.index.intersection(sig.index)
            rows.append({"cohort": acc, "stratum": name, "n_control": len(ctrl), "n_case": len(cases),
                         "DE_genes": len(_sig(r)),
                         "median_abs_log2FC_signature": float(r.loc[ms, "log2FC"].abs().median())})
            log("    %-10s %-5s n=%-3d DE %5d  median |log2FC| on signature genes %.2f"
                % (acc, name, len(cases), rows[-1]["DE_genes"], rows[-1]["median_abs_log2FC_signature"]))
    return pd.DataFrame(rows)
