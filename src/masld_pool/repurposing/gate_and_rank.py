"""
Gate, rank and emit the two candidate sections of the target-based repurposing arm.
Runs offline on the dated query snapshot; vendored unchanged from the analysis apart from paths.

O9E -- Task 7: gate, rank and emit the two candidate sections for the target-based
repurposing arm.

Fix round 1 (2026-09-11): pair identity is CANONICAL (case-insensitive,
whitespace-trimmed drug_name; drug_name.strip().upper()) end to end for gating,
consensus and ranking. An earlier version of this script used raw case-sensitive
drug_name as the pair key because it reproduced the controller's original reference
numbers (807/300/215) -- those numbers were themselves computed with the same
case-sensitivity bug (Pharos reports drug names lowercase; ChEMBL/DGIdb/OpenTargets
report them uppercase or mixed, so the same real drug at the same real target was
being scored as two separate single-platform pairs instead of one multi-platform
pair). The controller re-derived the correct base numbers as 776 pair-level MATCHes,
316 pairs at >=2 platforms, 229 canonical drugs, pre-gates. The final candidate table
has exactly ONE row per (canonical drug, target) pair; where case variants merge,
platforms/action_types/n_edges are unioned, and the raw strings that were merged are
recorded in `merged_raw_names` (display name preferring the variant with the most
platform support, tie-broken toward title case then uppercase).

The annotation join (approval, oral, hepatic, cardiac, mechanism) is also by canonical
name, which is the only key that resolves cleanly against drug_annotations.tsv (0
within-canon-group conflicts across 137 case-collapse groups, verified before writing
this script).

Gate order (fixed, attrition recorded at every stage):
  1. raw pair-level candidates
  2. PK pseudo-target re-assertion (0 removed -- check, not filter)
  3. approved only (approval_status == APPROVED; sub-buckets logged)
  4. oral (oral_verdict == ORAL)
  5. hepatic safety (hepatic_flag != DOCUMENTED; NO_US_LABEL retained, flagged)
  6. cardiac gate (cardiac_flag != DOCUMENTED; knowledge-based)
  7. direction rule (pair-level MATCH; WRONG DIRECTION -> negative control)
  8. consensus (label only; single-platform retained)
  9. mechanism status (fix round 1: LABEL only, never a filter -- REFERENCED or
     MECHANISM UNSUPPORTED both survive to the final table; this is the honest
     replacement for a fabricated mechanism, not grounds to drop the candidate)
"""

import csv
import os
import sys

PILOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PILOT_DIR)
import _o9lib as lib

# Working directory prepared by masld_pool.repurposing.run(): it holds the dated query
# snapshot (data/repurposing_snapshot) plus this run's recomputed signature table, and
# receives the gate outputs.
OUT = os.environ["MASLD_REPURPOSING_WORKDIR"]
META_SIG_DIR = OUT

log_lines = []


def log(*parts):
    line = " ".join(str(p) for p in parts)
    print(line)
    log_lines.append(line)


def read_tsv(path):
    with open(path, encoding="utf-8", newline="") as f:
        r = csv.DictReader(f, delimiter="\t")
        return list(r)


def write_tsv(path, rows, columns):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        for row in rows:
            out = {}
            for c in columns:
                v = row.get(c, "")
                if v is None:
                    v = ""
                out[c] = v
            w.writerow(out)


# ---------------------------------------------------------------------------
# Step 1: load reference tables
# ---------------------------------------------------------------------------

log("=== O9E: loading reference tables ===")

neighbour_direction = {}
for row in read_tsv(os.path.join(OUT, "neighbour_direction.tsv")):
    neighbour_direction[row["neighbour_symbol"]] = {
        "log2FC": row["neighbour_log2FC"],
        "padj": row["neighbour_padj"],
        "direction": row["neighbour_direction"],
        "tier": row["neighbour_evidence_tier"],
    }
log("neighbour_direction.tsv: %d rows" % len(neighbour_direction))

meta_sig_master = {}
meta_sig_path = os.path.join(META_SIG_DIR, "meta_signature_annotated.tsv")
for row in read_tsv(meta_sig_path):
    meta_sig_master[row["symbol"]] = {
        "log2FC": row["pooled_log2FC"],
        "padj": row["padj"],
    }
log("meta_signature_annotated.tsv (signature master): %d symbols" % len(meta_sig_master))

gene_sets_rows = read_tsv(os.path.join(OUT, "gene_sets.tsv"))
gene_sets_fallback = {}  # symbol -> {gene_set: (log2FC, padj)}, priority order applied at lookup
GENE_SET_PRIORITY = ["signature", "core", "early", "panel"]
for row in gene_sets_rows:
    gene_sets_fallback.setdefault(row["symbol"], {})[row["gene_set"]] = {
        "log2FC": row["pooled_log2FC"],
        "padj": row["padj"],
    }


def gene_a_stats(symbol):
    if symbol in meta_sig_master:
        return meta_sig_master[symbol]["log2FC"], meta_sig_master[symbol]["padj"], "signature_master"
    fallback = gene_sets_fallback.get(symbol, {})
    for gs in GENE_SET_PRIORITY:
        if gs in fallback:
            return fallback[gs]["log2FC"], fallback[gs]["padj"], gs
    return "", "", "NOT_FOUND"


lowcount_screen = {}
for row in read_tsv(os.path.join(META_SIG_DIR, "lowcount_screen.tsv")):
    lowcount_screen[row["symbol"]] = {
        "verdict": row["verdict"],
        "flagged_by_O3_5": row["flagged_by_O3_5"],
    }
log("lowcount_screen.tsv: %d symbols" % len(lowcount_screen))

drug_annot_rows = read_tsv(os.path.join(OUT, "drug_annotations.tsv"))
drug_annot_by_canon = {}
for row in drug_annot_rows:
    canon = row["drug_name_canonical"]
    if canon in drug_annot_by_canon:
        # verified upstream: all rows sharing a canon key carry identical annotation
        # data (0 diffs across 137 groups) -- keep the first, deterministic.
        continue
    drug_annot_by_canon[canon] = row
log("drug_annotations.tsv: %d rows, %d canonical drugs" % (len(drug_annot_rows), len(drug_annot_by_canon)))

mechanism_refs_rows = read_tsv(os.path.join(OUT, "mechanism_refs.tsv"))
mrefs_by_drug = {}
for row in mechanism_refs_rows:
    mrefs_by_drug.setdefault(row["drug_chembl_id"], []).append(row)
log("mechanism_refs.tsv: %d rows, %d distinct drug_chembl_id" % (len(mechanism_refs_rows), len(mrefs_by_drug)))

# Sanity check verified before writing this script: mechanism_status == REFERENCED
# in drug_annotations.tsv iff drug_chembl_id has >=1 row in mechanism_refs.tsv, with
# zero exceptions in either direction (667 REFERENCED ids, 331 UNSUPPORTED ids, 0
# UNSUPPORTED id present in mechanism_refs, 0 REFERENCED id absent from it).
referenced_ids = set(r["drug_chembl_id"] for r in drug_annot_rows if r["mechanism_status"] == "REFERENCED")
unsupported_ids = set(r["drug_chembl_id"] for r in drug_annot_rows if r["mechanism_status"] == "MECHANISM UNSUPPORTED")
mref_ids = set(mrefs_by_drug.keys())
assert not (unsupported_ids & mref_ids), "MECHANISM UNSUPPORTED drug has mechanism_refs rows -- upstream contract broken"
assert not (referenced_ids - mref_ids), "REFERENCED drug missing from mechanism_refs -- upstream contract broken"
log("mechanism_status <-> mechanism_refs.tsv cross-check: consistent (0 exceptions)")


# ---------------------------------------------------------------------------
# Step 2: build pair-level structures for both sections
# ---------------------------------------------------------------------------

def parse_evidence_moa(evidence_detail):
    """Extract the mechanism_of_action text a ChEMBL-platform edge embedded in its
    evidence_detail field, formatted 'mechanism_of_action=TEXT;refs=...'. Returns ''
    if the field does not carry this key (non-ChEMBL platforms)."""
    if not evidence_detail:
        return ""
    if not evidence_detail.startswith("mechanism_of_action="):
        return ""
    rest = evidence_detail[len("mechanism_of_action="):]
    idx = rest.find(";refs=")
    if idx == -1:
        return rest
    return rest[:idx]


def canon_name(drug_name):
    """Canonical drug identity: case-insensitive, whitespace-trimmed. Only strings
    identical up to case and leading/trailing whitespace are ever merged -- no fuzzy
    matching."""
    return drug_name.strip().upper()


def new_pair_entry(section):
    return {
        "section": section,
        "action_types": set(),
        "platforms": set(),
        "gene_sets": set(),
        "seed_symbols": set(),
        "expansion_sources": set(),
        "chembl_moa_texts": set(),
        "n_edges": 0,
        "gene_direction": None,
        "raw_name_platforms": {},  # raw drug_name string -> set(platforms) seen under that exact string
    }


pairs = {}  # (section, canonical_drug_name, target_symbol) -> entry

log("\n=== building pair-level structures (canonical drug identity) ===")

edges_direct = read_tsv(os.path.join(OUT, "edges_direct.tsv"))
n_direct_nominating = 0
for row in edges_direct:
    if row["edge_role"] != "nominating":
        continue
    n_direct_nominating += 1
    raw_name = row["drug_name"]
    key = ("A", canon_name(raw_name), row["gene_symbol"])
    e = pairs.setdefault(key, new_pair_entry("A"))
    e["platforms"].add(row["platform"])
    e["raw_name_platforms"].setdefault(raw_name, set()).add(row["platform"])
    e["gene_sets"].add(row["gene_set"])
    e["n_edges"] += 1
    if row["action_type"].strip():
        e["action_types"].add(row["action_type"])
    if e["gene_direction"] is None:
        e["gene_direction"] = row["gene_direction"]
    elif e["gene_direction"] != row["gene_direction"]:
        log("WARNING: inconsistent gene_direction for", key, e["gene_direction"], "vs", row["gene_direction"])
    if row["platform"] == "ChEMBL":
        moa = parse_evidence_moa(row["evidence_detail"])
        if moa:
            e["chembl_moa_texts"].add(moa)

log("edges_direct.tsv nominating rows:", n_direct_nominating, "-> Section A raw canonical pairs:",
    sum(1 for k in pairs if k[0] == "A"))

edges_expanded = read_tsv(os.path.join(OUT, "edges_expanded.tsv"))
n_expanded_nominating = 0
for row in edges_expanded:
    if row["edge_role"] != "nominating":
        continue
    n_expanded_nominating += 1
    sym = row["neighbour_symbol"]
    raw_name = row["drug_name"]
    key = ("B", canon_name(raw_name), sym)
    e = pairs.setdefault(key, new_pair_entry("B"))
    e["platforms"].add(row["platform"])
    e["raw_name_platforms"].setdefault(raw_name, set()).add(row["platform"])
    e["n_edges"] += 1
    e["seed_symbols"].add(row["seed_symbol"])
    for src in row["expansion_sources"].split(";"):
        if src.strip():
            e["expansion_sources"].add(src.strip())
    if row["action_type"].strip():
        e["action_types"].add(row["action_type"])
    nd = neighbour_direction.get(sym, {})
    nd_dir = nd.get("direction", "")
    if e["gene_direction"] is None:
        e["gene_direction"] = nd_dir
    if row["platform"] == "ChEMBL":
        moa = parse_evidence_moa(row["evidence_detail"])
        if moa:
            e["chembl_moa_texts"].add(moa)

log("edges_expanded.tsv nominating rows:", n_expanded_nominating, "-> Section B raw canonical pairs:",
    sum(1 for k in pairs if k[0] == "B"))

# Merge log: every canonical pair whose raw drug_name had more than one distinct
# case/whitespace variant across the underlying edges.
merge_log_rows = []
for (section, canon, target), e in pairs.items():
    if len(e["raw_name_platforms"]) > 1:
        merge_log_rows.append({
            "section": section, "drug_name_canonical": canon, "target": target,
            "merged_raw_names": ";".join(sorted(e["raw_name_platforms"].keys())),
        })
log("case-variant merges (canonical pairs whose raw drug_name had >1 distinct string form): %d"
    % len(merge_log_rows))
for m in merge_log_rows:
    log("  MERGED [%s] %s @ %s <- %s" % (m["section"], m["drug_name_canonical"], m["target"], m["merged_raw_names"]))


def pick_display_name(raw_name_platforms):
    """Prefer the raw string backed by the most platforms; tie-break toward title
    case, then uppercase, then lexical order for full determinism."""
    def sort_key(item):
        name, plats = item
        is_title = name == name.title() and name != name.upper() and name != name.lower()
        is_upper = name == name.upper()
        rank = 0 if is_title else (1 if is_upper else 2)
        return (-len(plats), rank, name)
    return sorted(raw_name_platforms.items(), key=sort_key)[0][0]

# Stage 2 check: PK pseudo-target re-assertion over every raw pair's target symbol,
# both sections.
pk_hits = [k for k in pairs if lib.is_pk_pseudotarget(k[2])]
log("PK pseudo-target re-assertion: %d raw pairs carry a PK pseudo-target (expect 0)" % len(pk_hits))
assert not pk_hits, "PK pseudo-targets present in raw pairs: %s" % pk_hits


# ---------------------------------------------------------------------------
# Step 3: per-pair annotation, direction verdict, target stats
# ---------------------------------------------------------------------------

def resolve_mechanism(drug_chembl_id, chembl_moa_texts):
    """Attach the accepted references for a candidate, preferring a target-specific
    isolation via the drug's own ChEMBL-platform evidence_detail text, falling back
    to the full drug-level mechanism_refs set when isolation is not possible."""
    rows = mrefs_by_drug.get(drug_chembl_id, [])
    if not rows:
        return "UNSUPPORTED", "", []
    if len(chembl_moa_texts) == 1:
        moa_text = next(iter(chembl_moa_texts))
        matched = [r for r in rows if r["mechanism_of_action"] == moa_text]
        if matched:
            distinct_targets = set(r["target_chembl_id"] for r in matched)
            if len(distinct_targets) == 1:
                return "TARGET_SPECIFIC", moa_text, matched
    distinct_moas = sorted(set(r["mechanism_of_action"] for r in rows))
    moa_out = distinct_moas[0] if len(distinct_moas) == 1 else ";".join(distinct_moas)
    return "DRUG_LEVEL_FALLBACK", moa_out, rows


def format_refs(ref_rows):
    seen = set()
    out = []
    for r in ref_rows:
        rt = r["ref_type"].strip()
        rid = r["ref_id"].strip()
        rt_cap = {"pubmed": "PubMed", "dailymed": "DailyMed", "fda": "FDA",
                  "fda label": "FDA", "clinicaltrials": "ClinicalTrials"}.get(rt.lower(), rt)
        item = "%s:%s" % (rt_cap, rid)
        if item not in seen:
            seen.add(item)
            out.append(item)
    return ";".join(sorted(out))


def extract_pmids(ref_rows):
    pmids = sorted(set(r["ref_id"].strip() for r in ref_rows if r["ref_type"].strip().lower() == "pubmed"))
    return ";".join(pmids)


records_by_section = {"A": [], "B": []}

for (section, canon, target), e in pairs.items():
    drug_name = pick_display_name(e["raw_name_platforms"])
    annot = drug_annot_by_canon.get(canon)

    verdict = lib.direction_verdict(e["gene_direction"], e["action_types"])

    if section == "A":
        log2fc, padj, stat_source = gene_a_stats(target)
        tier = ""
    else:
        nd = neighbour_direction.get(target, {})
        log2fc, padj, tier = nd.get("log2FC", ""), nd.get("padj", ""), nd.get("tier", "")
        stat_source = "neighbour_direction"

    lc = lowcount_screen.get(target)

    rec = {
        "section": section,
        "drug_name": drug_name,
        "drug_name_canonical": canon,
        "gene_symbol": target,
        "target": target,
        "gene_direction": e["gene_direction"],
        "target_direction": e["gene_direction"],
        "target_log2FC": log2fc,
        "target_padj": padj,
        "target_stat_source": stat_source,
        "neighbour_evidence_tier": tier,
        "action_type": ";".join(sorted(e["action_types"])),
        "platforms": ";".join(sorted(e["platforms"])),
        "platforms_n": len(e["platforms"]),
        "direction_verdict": verdict,
        "n_edges": e["n_edges"],
        "gene_sets": ";".join(sorted(e["gene_sets"])),
        "seed_symbol": ";".join(sorted(e["seed_symbols"])),
        "expansion_sources": ";".join(sorted(e["expansion_sources"])),
        "lowcount_verdict": lc["verdict"] if lc else "NOT_IN_SIG75",
        "lowcount_flagged_by_O3_5": lc["flagged_by_O3_5"] if lc else "",
        "merged_raw_names": ";".join(sorted(e["raw_name_platforms"].keys())),
        "n_raw_name_variants": len(e["raw_name_platforms"]),
        "_chembl_moa_texts": e["chembl_moa_texts"],
    }

    if annot is None:
        rec["approval_bucket"] = "NO_ANNOTATION"
        rec["approval_status"] = ""
        rec["drug_chembl_id"] = ""
        rec["oral_verdict"] = ""
        rec["hepatic_flag"] = ""
        rec["cardiac_flag"] = ""
        rec["mechanism_status"] = ""
        rec["hepatotoxicity_p"] = "NOT_SCORED"
        rec["cardiotoxicity_p"] = "NOT_SCORED"
    else:
        rec["approval_bucket"] = annot["approval_status"] if annot["approval_status"] != "APPROVED" else "APPROVED"
        rec["approval_status"] = annot["approval_status"]
        rec["drug_chembl_id"] = annot["drug_chembl_id"]
        rec["oral_verdict"] = annot["oral_verdict"]
        rec["hepatic_flag"] = annot["hepatic_flag"]
        rec["cardiac_flag"] = annot["cardiac_flag"]
        rec["mechanism_status"] = annot["mechanism_status"]
        rec["hepatotoxicity_p"] = annot["hepatotoxicity_p"]
        rec["cardiotoxicity_p"] = annot["cardiotoxicity_p"]

    records_by_section[section].append(rec)

log("\nSection A raw pair records: %d" % len(records_by_section["A"]))
log("Section B raw pair records: %d" % len(records_by_section["B"]))


# ---------------------------------------------------------------------------
# Step 4: raw-pair direction-verdict reproduction check (informational only,
# computed over ALL raw pairs before any gate is applied)
# ---------------------------------------------------------------------------

def verdict_counts(recs):
    c = {"MATCH": 0, "WRONG DIRECTION": 0, "UNKNOWN": 0}
    for r in recs:
        c[r["direction_verdict"]] += 1
    return c

log("\n=== canonical raw pair-level direction-verdict reproduction check (fix round 1) ===")
all_recs = records_by_section["A"] + records_by_section["B"]
vc_all = verdict_counts(all_recs)
vc_a = verdict_counts(records_by_section["A"])
vc_b = verdict_counts(records_by_section["B"])
log("overall:", vc_all, " (corrected controller reference: 776 MATCH, pre-gates, canonical keying)")
log("Section A:", vc_a)
log("Section B:", vc_b)

match_multi = [r for r in all_recs if r["direction_verdict"] == "MATCH" and r["platforms_n"] >= 2]
log(">=2-platform MATCH pairs: %d, distinct drugs: %d (corrected controller reference: 316 pairs / 229 drugs)"
    % (len(match_multi), len(set(r["drug_name_canonical"] for r in match_multi))))


# ---------------------------------------------------------------------------
# Step 5: the nine-stage gate, applied per section
# ---------------------------------------------------------------------------

attrition_rows = []


def log_attrition(section, stage, recs):
    n_pairs = len(recs)
    n_drugs = len(set(r["drug_name_canonical"] for r in recs))
    n_genes = len(set(r["gene_symbol"] for r in recs))
    n_edges = sum(r["n_edges"] for r in recs)
    attrition_rows.append({
        "section": section, "stage": stage,
        "n_pairs": n_pairs, "n_drugs": n_drugs, "n_genes": n_genes, "n_edges": n_edges,
    })
    log("  [%s] %s: n_pairs=%d n_drugs=%d n_genes=%d n_edges=%d" % (section, stage, n_pairs, n_drugs, n_genes, n_edges))


negative_control_rows = []
final_candidates = {"A": [], "B": []}

for section in ("A", "B"):
    log("\n=== Section %s gate ===" % section)
    recs = records_by_section[section]

    # Stage 1: raw pairs
    log_attrition(section, "1_raw_pairs", recs)

    # Stage 2: PK pseudo-target re-assertion (0 removed; already asserted above)
    log_attrition(section, "2_pk_pseudotarget_check", recs)

    # Stage 3: approved only, with sub-buckets logged
    approved = [r for r in recs if r["approval_status"] == "APPROVED"]
    for bucket in ("NOT_APPROVED", "APPROVED_NO_LABEL", "MAX_PHASE_UNKNOWN", "NO_ANNOTATION"):
        removed = [r for r in recs if r["approval_bucket"] == bucket]
        attrition_rows.append({
            "section": section, "stage": "3_removed_" + bucket,
            "n_pairs": len(removed),
            "n_drugs": len(set(r["drug_name_canonical"] for r in removed)),
            "n_genes": len(set(r["gene_symbol"] for r in removed)),
            "n_edges": sum(r["n_edges"] for r in removed),
        })
        log("  [%s] 3_removed_%s: n_pairs=%d n_drugs=%d" % (section, bucket, len(removed),
            len(set(r["drug_name_canonical"] for r in removed))))
    log_attrition(section, "3_approved_only", approved)

    # Stage 4: oral
    oral = [r for r in approved if r["oral_verdict"] == "ORAL"]
    log_attrition(section, "4_oral", oral)

    # Stage 5: hepatic safety
    hepatic_ok = [r for r in oral if r["hepatic_flag"] != "DOCUMENTED"]
    for r in hepatic_ok:
        r["hepatic_note"] = "RETAINED_NO_US_LABEL" if r["hepatic_flag"] == "NO_US_LABEL" else ""
    log_attrition(section, "5_hepatic_safety", hepatic_ok)

    # Stage 6: cardiac gate
    cardiac_ok = [r for r in hepatic_ok if r["cardiac_flag"] != "DOCUMENTED"]
    for r in cardiac_ok:
        r["cardiac_note"] = "RETAINED_NO_US_LABEL" if r["cardiac_flag"] == "NO_US_LABEL" else ""
    log_attrition(section, "6_cardiac_gate", cardiac_ok)

    # Stage 7: direction rule (pair-level, already computed)
    direction_match = [r for r in cardiac_ok if r["direction_verdict"] == "MATCH"]
    wrong_direction = [r for r in cardiac_ok if r["direction_verdict"] == "WRONG DIRECTION"]
    unknown_direction = [r for r in cardiac_ok if r["direction_verdict"] == "UNKNOWN"]
    for r in wrong_direction:
        negative_control_rows.append(dict(r, reason="WRONG DIRECTION: gene_direction=%s vs action_types=%s"
                                           % (r["gene_direction"], r["action_type"])))
    attrition_rows.append({
        "section": section, "stage": "7_removed_WRONG_DIRECTION",
        "n_pairs": len(wrong_direction),
        "n_drugs": len(set(r["drug_name_canonical"] for r in wrong_direction)),
        "n_genes": len(set(r["gene_symbol"] for r in wrong_direction)),
        "n_edges": sum(r["n_edges"] for r in wrong_direction),
    })
    attrition_rows.append({
        "section": section, "stage": "7_removed_UNKNOWN",
        "n_pairs": len(unknown_direction),
        "n_drugs": len(set(r["drug_name_canonical"] for r in unknown_direction)),
        "n_genes": len(set(r["gene_symbol"] for r in unknown_direction)),
        "n_edges": sum(r["n_edges"] for r in unknown_direction),
    })
    log_attrition(section, "7_direction_match", direction_match)

    # Stage 8: consensus (labelling only -- no removal)
    for r in direction_match:
        r["consensus_label"] = lib.consensus_label(r["platforms"].split(";"))
    log_attrition(section, "8_consensus_labelled", direction_match)
    n_consensus = sum(1 for r in direction_match if r["consensus_label"] == "CONSENSUS")
    n_single = len(direction_match) - n_consensus
    log("  [%s] consensus split: CONSENSUS=%d, single-platform(retained)=%d" % (section, n_consensus, n_single))

    # Stage 9 (fix round 1): mechanism status is a LABEL, never a filter. Every pair
    # that reaches stage 9 is retained, carrying mechanism_status ==
    # REFERENCED or MECHANISM UNSUPPORTED. Nothing is removed here by construction.
    # Fix round 3, Finding 2: the attrition row is named "..._label_only_no_removal",
    # not "removed_...", because a "removed" row that is structurally always zero
    # reads in a funnel figure as a filter that ran and caught nothing, which
    # misrepresents what stage 9 actually does (it never filters).
    referenced = [r for r in direction_match if r["mechanism_status"] == "REFERENCED"]
    unsupported = [r for r in direction_match if r["mechanism_status"] != "REFERENCED"]
    attrition_rows.append({
        "section": section, "stage": "9_mechanism_label_only_no_removal",
        "n_pairs": 0, "n_drugs": 0, "n_genes": 0, "n_edges": 0,
    })
    log("  [%s] 9_mechanism_label_only_no_removal: n_pairs=0 (label only, not a filter -- %d REFERENCED / %d "
        "MECHANISM UNSUPPORTED both retained)" % (section, len(referenced), len(unsupported)))
    log_attrition(section, "9_mechanism_labelled", direction_match)

    final_candidates[section] = direction_match


# ---------------------------------------------------------------------------
# Step 6: mechanism reference resolution for surviving candidates
# ---------------------------------------------------------------------------

n_target_specific = {"A": 0, "B": 0}
n_drug_fallback = {"A": 0, "B": 0}

for section in ("A", "B"):
    for r in final_candidates[section]:
        scope, moa, ref_rows = resolve_mechanism(r["drug_chembl_id"], r["_chembl_moa_texts"])
        r["ref_scope"] = scope
        r["mechanism_of_action"] = moa
        r["accepted_references"] = format_refs(ref_rows)
        r["ref_pmids"] = extract_pmids(ref_rows)
        if scope == "TARGET_SPECIFIC":
            n_target_specific[section] += 1
        elif scope == "DRUG_LEVEL_FALLBACK":
            n_drug_fallback[section] += 1

log("\nmechanism reference resolution: Section A target_specific=%d drug_level_fallback=%d"
    % (n_target_specific["A"], n_drug_fallback["A"]))
log("mechanism reference resolution: Section B target_specific=%d drug_level_fallback=%d"
    % (n_target_specific["B"], n_drug_fallback["B"]))


# ---------------------------------------------------------------------------
# Step 7 (fix round 1 + fix round 2 + fix round 3): pair identity is canonical
# (case-collapsed) by construction from the merge step above -- one row per
# (canonical drug, target) pair after case folding. Fix round 2 (USER DECISION) adds
# a second merge pass on top of that: salt-form/free-base siblings of the same active
# ingredient at the same target are merged into one row per ingredient, using
# lib.SALT_SUFFIXES / lib.strip_base_name from _o9lib.py (fix round 3, Finding 4:
# moved there from a local copy so this module and o9d_drug_annotate.py share one
# tested definition, including the "alfa"/"beta" exclusion -- Task 6 Finding 2, the
# THROMBIN ALFA case). Two rows merge only when their base name AND target are both
# identical -- never fuzzy-matched.
# ---------------------------------------------------------------------------

# Fix round 3 (2026-09-11): SALT_SUFFIXES and strip_base_name moved into _o9lib.py so this
# script and o9d_drug_annotate.py share one tested copy (lib.strip_base_name below) instead
# of two independent copies of a correctness-critical exclusion list that could drift apart.


def merge_salt_group(rows):
    """Merge >=2 candidate rows that share a base name and target into one row per
    active ingredient. Platforms/action_types/n_edges are unioned. mechanism_status
    is REFERENCED if ANY sibling was REFERENCED (the mechanism belongs to the
    pharmacophore, not the counterion); accepted_references/ref_pmids/mechanism_of_action
    are unioned across the REFERENCED siblings only. All other annotation-derived
    fields (approval_status, oral_verdict, hepatic_flag, cardiac_flag, hepatotoxicity_p,
    cardiotoxicity_p and their notes) are taken from the "primary" sibling -- the one
    with the most platform support, used only to pick which sibling's annotation row
    backs the merged record -- since every sibling already passed the same stage 3-6
    gates individually. Fix round 3, Finding 3: the DISPLAYED drug_name is always the
    base/ingredient name (lib.strip_base_name of the group's shared base), never the
    popularity-contest winner among salt forms -- picking display by platform count
    would silently print a specific salt's name in a future run where that salt happens
    to have more platform support than the free base, which would be inconsistent with
    drug_name_canonical (also the base name) on the same row."""
    name_platforms = {r["drug_name_canonical"]: set(r["platforms"].split(";")) for r in rows}
    primary_name = pick_display_name(name_platforms)
    primary = next(r for r in rows if r["drug_name_canonical"] == primary_name)
    base_name = lib.strip_base_name(primary_name)

    all_platforms = set()
    all_action_types = set()
    all_gene_sets = set()
    all_seed_symbols = set()
    all_expansion_sources = set()
    all_name_forms = set()
    all_chembl_ids = []
    total_n_edges = 0
    for r in rows:
        all_platforms.update(p for p in r["platforms"].split(";") if p)
        all_action_types.update(a for a in r["action_type"].split(";") if a)
        all_gene_sets.update(g for g in r["gene_sets"].split(";") if g)
        all_seed_symbols.update(s for s in r["seed_symbol"].split(";") if s)
        all_expansion_sources.update(x for x in r["expansion_sources"].split(";") if x)
        all_name_forms.update(n for n in r["merged_raw_names"].split(";") if n)
        all_name_forms.add(r["drug_name_canonical"])
        if r["drug_chembl_id"]:
            all_chembl_ids.append(r["drug_chembl_id"])
        total_n_edges += r["n_edges"]

    referenced_rows = [r for r in rows if r["mechanism_status"] == "REFERENCED"]
    if referenced_rows:
        mechanism_status = "REFERENCED"
        refs, pmids, moas, scopes = set(), set(), set(), set()
        for r in referenced_rows:
            if r["accepted_references"]:
                refs.update(r["accepted_references"].split(";"))
            if r["ref_pmids"]:
                pmids.update(r["ref_pmids"].split(";"))
            if r["mechanism_of_action"]:
                moas.add(r["mechanism_of_action"])
            scopes.add(r["ref_scope"])
        accepted_references = ";".join(sorted(refs))
        ref_pmids = ";".join(sorted(pmids))
        mechanism_of_action = ";".join(sorted(moas))
        ref_scope = "TARGET_SPECIFIC" if scopes == {"TARGET_SPECIFIC"} else "DRUG_LEVEL_FALLBACK"
    else:
        mechanism_status = "MECHANISM UNSUPPORTED"
        accepted_references = ""
        ref_pmids = ""
        mechanism_of_action = ""
        ref_scope = "UNSUPPORTED"

    merged = dict(primary)
    merged["drug_name"] = base_name
    merged["drug_name_canonical"] = base_name
    merged["drug_chembl_id"] = ";".join(sorted(set(all_chembl_ids)))
    merged["platforms"] = ";".join(sorted(all_platforms))
    merged["platforms_n"] = len(all_platforms)
    merged["action_type"] = ";".join(sorted(all_action_types))
    merged["n_edges"] = total_n_edges
    merged["gene_sets"] = ";".join(sorted(all_gene_sets))
    merged["seed_symbol"] = ";".join(sorted(all_seed_symbols))
    merged["expansion_sources"] = ";".join(sorted(all_expansion_sources))
    merged["merged_raw_names"] = ";".join(sorted(all_name_forms))
    merged["n_raw_name_variants"] = len(all_name_forms)
    merged["mechanism_status"] = mechanism_status
    merged["accepted_references"] = accepted_references
    merged["ref_pmids"] = ref_pmids
    merged["mechanism_of_action"] = mechanism_of_action
    merged["ref_scope"] = ref_scope
    merged["consensus_label"] = lib.consensus_label(sorted(all_platforms))
    return merged


salt_merge_log = {"A": [], "B": []}
for section in ("A", "B"):
    groups = {}
    for r in final_candidates[section]:
        base = lib.strip_base_name(r["drug_name_canonical"])
        key = (base, r["gene_symbol"])
        groups.setdefault(key, []).append(r)
    merged_list = []
    for (base, target), rows in groups.items():
        if len(rows) == 1:
            merged_list.append(rows[0])
            continue
        merged = merge_salt_group(rows)
        merged_list.append(merged)
        sibling_desc = ";".join(sorted("%s(%s)" % (r["drug_name_canonical"], r["mechanism_status"]) for r in rows))
        salt_merge_log[section].append({
            "section": section, "base_name": base, "target": target,
            "n_siblings": len(rows), "siblings": sibling_desc,
            "merged_mechanism_status": merged["mechanism_status"],
            "merged_drug_name": merged["drug_name"],
            "merged_platforms_n": merged["platforms_n"],
            "merged_n_edges": merged["n_edges"],
        })
        log("  SALT-MERGED [%s] %s @ %s <- %s -> merged mechanism_status=%s (display=%s)"
            % (section, base, target, sibling_desc, merged["mechanism_status"], merged["drug_name"]))
    final_candidates[section] = merged_list

log("\nsalt-form merges applied: Section A=%d, Section B=%d" % (len(salt_merge_log["A"]), len(salt_merge_log["B"])))

# Re-verify: after both merge passes, no (drug_name_canonical, gene_symbol) duplicate
# of any kind should remain in either section.
for section in ("A", "B"):
    seen = set()
    dupes = []
    for r in final_candidates[section]:
        k = (r["drug_name_canonical"], r["gene_symbol"])
        if k in seen:
            dupes.append(k)
        seen.add(k)
    assert not dupes, "Section %s: remaining (drug,target) duplicates after salt merge: %s" % (section, dupes)
    log("Section %s: 0 remaining (drug,target) duplicates after salt merge (checked %d rows)"
        % (section, len(final_candidates[section])))

# Fix round 3, Finding 1 (Critical): record the salt-merge collapse as its own attrition
# stage, immediately after 9_mechanism_labelled. Without this row, attrition.tsv's last
# row for Section B would still show 23/20 while section_b_candidates.tsv shows 17/14 --
# a reviewer-visible mismatch between the funnel figure and the candidate table it is
# supposed to summarise. This is an extension of the existing identity-consolidation
# work (fix rounds 1-2), not a new numbered gate -- nothing is filtered out here, rows
# are only merged, so n_drugs and n_pairs can both drop while nothing was "removed".
for section in ("A", "B"):
    log_attrition(section, "10_salt_form_merge", final_candidates[section])


# ---------------------------------------------------------------------------
# Step 8: ranking within each section
# ---------------------------------------------------------------------------

def hepatotoxicity_sort_key(r):
    v = r["hepatotoxicity_p"]
    if v == "NOT_SCORED" or v is None or v == "":
        return (1, 0.0)
    try:
        return (0, float(v))
    except ValueError:
        return (1, 0.0)


for section in ("A", "B"):
    recs = final_candidates[section]
    genes_per_drug = {}
    for r in recs:
        genes_per_drug.setdefault(r["drug_name_canonical"], set()).add(r["gene_symbol"])
    for r in recs:
        r["n_genes_engaged"] = len(genes_per_drug[r["drug_name_canonical"]])

    def abs_log2fc(r):
        try:
            return abs(float(r["target_log2FC"]))
        except (TypeError, ValueError):
            return 0.0

    recs.sort(key=lambda r: (
        -r["platforms_n"],
        -r["n_genes_engaged"],
        -abs_log2fc(r),
        hepatotoxicity_sort_key(r),
    ))
    for i, r in enumerate(recs, start=1):
        r["rank"] = i


# ---------------------------------------------------------------------------
# Step 9: write outputs
# ---------------------------------------------------------------------------

CANDIDATE_COLUMNS = [
    "rank", "section", "drug_name", "drug_name_canonical", "drug_chembl_id",
    "gene_symbol", "target", "gene_direction", "target_direction",
    "target_log2FC", "target_padj", "target_stat_source", "neighbour_evidence_tier",
    "action_type", "platforms", "platforms_n", "consensus_label",
    "approval_status", "oral_verdict", "hepatic_flag", "hepatic_note",
    "cardiac_flag", "cardiac_note", "mechanism_status", "mechanism_of_action",
    "ref_scope", "accepted_references", "ref_pmids",
    "hepatotoxicity_p", "cardiotoxicity_p",
    "n_genes_engaged", "n_edges", "merged_raw_names", "n_raw_name_variants",
    "lowcount_verdict", "lowcount_flagged_by_O3_5",
    "gene_sets",  # Section A
    "seed_symbol", "neighbour_symbol", "expansion_sources",  # Section B
]

for section, fname in (("A", "section_a_candidates.tsv"), ("B", "section_b_candidates.tsv")):
    recs = final_candidates[section]
    if section == "B":
        for r in recs:
            r["neighbour_symbol"] = r["gene_symbol"]
    write_tsv(os.path.join(OUT, fname), recs, CANDIDATE_COLUMNS)
    log("wrote %s: %d candidates" % (fname, len(recs)))

NEG_CONTROL_COLUMNS = CANDIDATE_COLUMNS + ["reason"]
for r in negative_control_rows:
    if r["section"] == "B":
        r["neighbour_symbol"] = r["gene_symbol"]
write_tsv(os.path.join(OUT, "negative_control_wrong_direction.tsv"), negative_control_rows, NEG_CONTROL_COLUMNS)
log("wrote negative_control_wrong_direction.tsv: %d rows (WRONG DIRECTION only)" % len(negative_control_rows))

SALT_MERGE_LOG_COLUMNS = ["section", "base_name", "target", "n_siblings", "siblings",
                          "merged_mechanism_status", "merged_drug_name",
                          "merged_platforms_n", "merged_n_edges"]
all_salt_merge_rows = salt_merge_log["A"] + salt_merge_log["B"]
write_tsv(os.path.join(OUT, "salt_merge_log.tsv"), all_salt_merge_rows, SALT_MERGE_LOG_COLUMNS)
log("wrote salt_merge_log.tsv: %d rows (fix round 3, Finding 1)" % len(all_salt_merge_rows))

ATTRITION_COLUMNS = ["section", "stage", "n_pairs", "n_drugs", "n_genes", "n_edges"]
write_tsv(os.path.join(OUT, "attrition.tsv"), attrition_rows, ATTRITION_COLUMNS)
log("wrote attrition.tsv: %d rows" % len(attrition_rows))

log("\n=== DONE ===")
for section in ("A", "B"):
    recs = final_candidates[section]
    n_ref = sum(1 for r in recs if r["mechanism_status"] == "REFERENCED")
    n_uns = sum(1 for r in recs if r["mechanism_status"] != "REFERENCED")
    log("Section %s final candidates: %d (REFERENCED=%d, MECHANISM UNSUPPORTED=%d)"
        % (section, len(recs), n_ref, n_uns))
