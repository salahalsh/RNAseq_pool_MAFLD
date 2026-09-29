"""Target-based repurposing: re-run the gate sequence offline on the dated query snapshot.

The drug-target edges, network neighbours and drug annotations come from live databases
(Open Targets, DGIdb, ChEMBL, Pharos, BindingDB, openFDA, STRING v12.0, Reactome,
KEGG) queried on 29 September 2026. Those queries are not repeatable to the record, so their
results are shipped as a snapshot and only the gates, which are deterministic, are re-run.
The signature the gates look genes up in is the one recomputed in this run.
"""
import contextlib
import gzip
import io as _io
import os
import runpy
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import RESULTS, SNAPSHOT_DIR

HERE = Path(__file__).resolve().parent


def neighbour_directions(M):
    """Each network neighbour's own direction and tier, read from the pooled results."""
    snap = pd.read_csv(SNAPSHOT_DIR / "neighbour_direction.tsv", sep="\t")
    r = M.reindex(snap.ensembl_gene_id)
    tier = np.where(r.padj.isna().values, "NOT_TESTED",
                    np.where(r.padj.values < 0.05, "SIGNIFICANT", "MEASURED_NS"))
    direction = np.where(r.pooled_log2FC.values > 0, "UP", "DOWN")
    same = (tier == snap.neighbour_evidence_tier.values)
    tested = tier != "NOT_TESTED"
    same &= ~tested | (direction == snap.neighbour_direction.values)
    return int(same.sum()), len(snap)


def run(sig, symbols, log=print):
    work = RESULTS / "repurposing"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    for f in SNAPSHOT_DIR.iterdir():
        if f.name.startswith("expected_"):
            continue
        if f.suffix == ".gz":
            with gzip.open(f, "rb") as fi, open(work / f.stem, "wb") as fo:
                shutil.copyfileobj(fi, fo)
        else:
            shutil.copyfile(f, work / f.name)
    s = sig.copy()
    s["symbol"] = s.index.map(symbols).fillna("")
    s.to_csv(work / "meta_signature_annotated.tsv", sep="\t")

    os.environ["MASLD_REPURPOSING_WORKDIR"] = str(work)
    buf = _io.StringIO()
    with contextlib.redirect_stdout(buf):
        runpy.run_path(str(HERE / "gate_and_rank.py"), run_name="__main__")
    (work / "gate_log.txt").write_text(buf.getvalue(), encoding="utf-8")
    att = pd.read_csv(work / "attrition.tsv", sep="\t")
    cand = pd.read_csv(work / "section_b_candidates.tsv", sep="\t")
    neg = pd.read_csv(work / "negative_control_wrong_direction.tsv", sep="\t")
    cur = pd.read_csv(SNAPSHOT_DIR / "mechanism_concordance_curated.tsv", sep="\t")
    cand = cand.merge(cur[["drug_name_canonical", "target", "concordance"]],
                      on=["drug_name_canonical", "target"], how="left")
    return att, cand, neg


def expected():
    """Counts of the reference gate run shipped with the snapshot."""
    cand = pd.read_csv(SNAPSHOT_DIR / "expected_section_b_candidates.tsv", sep="\t")
    neg = pd.read_csv(SNAPSHOT_DIR / "expected_negative_control_wrong_direction.tsv", sep="\t")
    return {"rows": len(cand), "drugs": cand.drug_name_canonical.nunique(), "negative_control": len(neg)}


def same_attrition(att):
    """True when every stage of both sections has the same drug, gene and pair counts as the
    reference run."""
    ref = pd.read_csv(SNAPSHOT_DIR / "expected_attrition.tsv", sep="\t")
    cols = [c for c in ref.columns if c in att.columns]
    a = att[cols].sort_values(["section", "stage"]).reset_index(drop=True)
    b = ref[cols].sort_values(["section", "stage"]).reset_index(drop=True)
    return bool(a.equals(b))
