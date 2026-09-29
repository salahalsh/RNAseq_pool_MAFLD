"""Reproduce the analysis of the paper from the per-cohort files in data/.

Every stage prints its results to the terminal and checks them against the values reported
in the paper (PASS / DIFF). No figures are drawn.

    python run_all.py                    # full run; the classifier over ten seeds (about 15 min)
    python run_all.py --quick            # classifier on the pre-set seed only (about 5 min);
                                         # checked against the seed-1 sensitivity values
    python run_all.py --skip-classifier  # skip the classifier, SHAP and prioritised core
    python run_all.py --save             # also write the result tables to results/
    python run_all.py --use-recomputed-de  # propagate this run's DESeq2 output downstream
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from masld_pool import core, de, harmonise, io, meta, models, repurposing, strata  # noqa: E402
from masld_pool.checks import Checker  # noqa: E402
from masld_pool.config import COHORTS, RESULTS, SEED, SEEDS  # noqa: E402

# ---------------------------------------------------------------------- published values
# classifier, SHAP and core: the primary summary over ten seeds (decision R2-09), and the
# pre-set seed alone, which the paper reports as a sensitivity (used by --quick)
EXPECT_SEEDS = {
    "en_auroc": 0.890, "gb_auroc": 0.820, "en_slope": 1.148, "gb_slope": 0.322,
    "en_internal_auroc": 0.854, "en_internal_slope": 0.724,
    "gb_internal_auroc": 0.905, "gb_internal_slope": 0.333,
    "top100_in_signature": 19, "lgbm_cohort_acc": 0.954, "rf_cohort_acc": 0.793,
    "tiers": (3, 43, 32), "tier1": ["ENO3", "FBXO2", "OAT"],
    "panel": ["CTNNA3", "CYP3A4", "DEFB1", "ENO3", "FBXO2", "KRTCAP3", "MACROH2A2", "MGST3",
              "OAT", "PGAP4", "POMGNT1", "VIL1"],
    "panel_not_signature": ["MGST3", "POMGNT1"],
}
EXPECT_SEED1 = {
    "en_auroc": 0.890, "gb_auroc": 0.819, "en_slope": 1.283, "gb_slope": 0.329,
    "en_internal_auroc": 0.856, "en_internal_slope": 0.784,
    "gb_internal_auroc": 0.902, "gb_internal_slope": 0.296,
    "top100_in_signature": 34, "lgbm_cohort_acc": 0.957, "rf_cohort_acc": 0.817,
    "tiers": (8, 58, 22),
    "tier1": ["CTNNA3", "CYP2C19", "ENO3", "FBXO2", "ME1", "OAT", "OLFM2", "SLCO1A2"],
    "panel": ["CTNNA3", "CYP2C19", "CYP3A4", "DEFB1", "ENO3", "FBXO2", "KRTCAP3", "ME1", "OAT",
              "OLFM2", "SLCO1A2", "VIL1"],
    "panel_not_signature": [],
}


def section(title):
    print("\n" + "=" * 88 + "\n" + title + "\n" + "=" * 88)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="classifier on the pre-set seed only")
    ap.add_argument("--skip-classifier", action="store_true",
                    help="skip the classifier, SHAP, negative control and prioritised core")
    ap.add_argument("--save", action="store_true", help="write result tables to results/")
    ap.add_argument("--use-recomputed-de", action="store_true",
                    help="carry this run's DESeq2 tables forward instead of the shipped ones")
    args = ap.parse_args()
    t0 = time.time()
    chk = Checker()
    sym = io.symbols()
    saved = {}

    # ------------------------------------------------------------------ 1
    section("1. Harmonisation: eight cohorts on one gene namespace")
    H = harmonise.build()
    n_ctrl = int((H.meta.group_class == "control").sum())
    n_case = int((H.meta.group_class == "case").sum())
    chk.eq("primary samples", len(H.meta), 605)
    chk.eq("controls / cases", (n_ctrl, n_case), (111, 494))
    chk.eq("genes shared by the eight cohorts", H.genes_shared_counts, 15800)
    chk.eq("genes in the log2 CPM matrix", H.logexpr.shape[0], 15800)

    # ------------------------------------------------------------------ 2
    section("2. Per-cohort differential expression (case vs control, no covariates)")
    print("  Each cohort is re-analysed from its counts and compared with the DE table that entered\n"
          "  the meta-analysis (shipped in data/cohorts). Downstream stages use the shipped tables\n"
          "  unless --use-recomputed-de is given; see README, 'Numerical reproducibility'.")
    table1 = {"GSE135251": (14239, 6728), "GSE162694": (15516, 7321), "GSE281797": (14022, 2),
              "GSE185051": (15256, 9515), "GSE126848": (12734, 6601), "GSE268360": (13148, 9),
              "GSE260666": (14231, 61), "GSE183229": (14094, 7)}
    DE, DE_new, rows = {}, {}, []
    for acc in COHORTS:
        r = de.per_cohort(H, acc)
        ref = io.de_reference(acc)
        DE_new[acc], DE[acc] = r, ref
        common = r.index.intersection(ref.index)
        d_se = (r.loc[common, "lfcSE"] - ref.loc[common, "lfcSE"]).abs()
        d_lfc = (r.loc[common, "log2FC"] - ref.loc[common, "log2FC"]).abs()
        n_new, n_ref = int((r.padj < 0.05).sum()), int((ref.padj < 0.05).sum())
        rows.append({"cohort": acc, "genes_tested": len(r), "same_genes": bool(len(common) == len(r) == len(ref)),
                     "DE_FDR05_recomputed": n_new, "DE_FDR05_shipped": n_ref,
                     "median_abs_diff_SE": float(d_se.median()),
                     "genes_SE_diff_gt_1e-3": int((d_se > 1e-3).sum()),
                     "max_abs_diff_log2FC": float(d_lfc.max())})
        print("  %-10s tested %5d  DE (FDR<0.05) recomputed %5d / shipped %5d | median |dSE| %.0e, "
              "genes with |dSE| > 1e-3: %d"
              % (acc, len(r), n_new, n_ref, d_se.median(), (d_se > 1e-3).sum()))
        chk.eq("%s genes tested / DE (shipped table)" % acc, (len(ref), n_ref), table1[acc])
        chk.eq("%s recomputed: same genes, DE count within 1" % acc,
               bool(len(common) == len(r) == len(ref) and abs(n_new - n_ref) <= 1), True)
    saved["de_recomputation_vs_shipped"] = pd.DataFrame(rows)
    if args.use_recomputed_de:
        DE = DE_new
        print("  --use-recomputed-de: downstream stages use the recomputed tables")

    # ------------------------------------------------------------------ 3
    section("3. Random-effects meta-analysis (DerSimonian-Laird) and the 73-gene signature")
    R = meta.run(DE)
    M, sig = R["M"], R["sig"]
    up, down = int((sig.pooled_log2FC > 0).sum()), int((sig.pooled_log2FC < 0).sum())
    print("  pooled genes %d | with k >= 4 cohorts %d | signature %d (%d up, %d down) | strict |log2FC|>=1: %d"
          % (len(M), int((M.k_cohorts >= 4).sum()), len(sig), up, down, len(R["strict"])))
    top = sig.head(10).copy()
    top.insert(0, "symbol", top.index.map(sym))
    print(top[["symbol", "pooled_log2FC", "ci_low", "ci_high", "padj", "I2", "k_cohorts"]]
          .round(4).to_string())
    chk.eq("genes pooled (BH universe)", len(M), 15741)
    chk.eq("genes with k >= 4", int((M.k_cohorts >= 4).sum()), 14364)
    chk.eq("signature size (up, down)", (len(sig), up, down), (73, 36, 37))
    chk.eq("signature genes with |log2FC| >= 1", int((sig.pooled_log2FC.abs() >= 1).sum()), 14)
    chk.close("median I2 in the signature (%)", sig.I2.median(), 67.2, 0.05)
    chk.eq("signature genes with I2 > 75 %", int((sig.I2 > 75).sum()), 25)
    cyp = M.loc["ENSG00000160868"]
    chk.close("CYP3A4 pooled log2FC", cyp.pooled_log2FC, -0.628, 0.0005)
    chk.close("CYP3A4 adjusted p (x1e7)", cyp.padj * 1e7, 4.50, 0.005)
    hk = meta.hksj(R["Y"], R["S"], M)
    hk_sig = hk.loc[sig.index]
    hk_rule = (hk.padj_hksj < 0.05) & (hk.pooled_log2FC.abs() >= 0.5) & (hk.k_cohorts >= 4)
    print("  Hartung-Knapp-Sidik-Jonkman: genes passing the rule %d | smallest adjusted p %.3f | "
          "signature genes with nominal p < 0.05: %d, < 0.001: %d"
          % (int(hk_rule.sum()), np.nanmin(hk.padj_hksj), int((hk_sig.p_hksj < 0.05).sum()),
             int((hk_sig.p_hksj < 0.001).sum())))
    chk.eq("HKSJ: genes at BH < 0.05 under the signature rule", int(hk_rule.sum()), 0)
    chk.close("HKSJ: smallest adjusted p", np.nanmin(hk.padj_hksj), 0.132, 0.0005)
    chk.eq("HKSJ: signature genes with nominal p < 0.05 / < 0.001",
           (int((hk_sig.p_hksj < 0.05).sum()), int((hk_sig.p_hksj < 0.001).sum())), (72, 9))
    saved["meta_all_genes"] = M
    saved["signature"] = sig.assign(symbol=sig.index.map(sym))
    saved["hksj"] = hk

    # ------------------------------------------------------------------ 4
    section("4. Robustness: between-study variance estimator, leave-one-cohort-out, replication")
    E, sigs = meta.estimator_sensitivity(R["Y"], R["S"])
    for k, v in E.items():
        print("  %-5s median tau2 %.4f | median pooled SE %.4f | signature %d genes | of the 73: %d"
              % (k, v.tau2.median(), v.pooled_SE.median(), len(sigs[k]), len(sigs[k] & set(sig.index))))
    invariant = sigs["DL"] & sigs["PM"] & sigs["REML"]
    wid = {k: float((E[k].ci_width / E["DL"].ci_width).replace([np.inf, -np.inf], np.nan).dropna().median() - 1)
           for k in ("PM", "REML")}
    print("  called by all three estimators: %d | median per-gene CI widening: PM %.2f %%, REML %.2f %%"
          % (len(invariant), 100 * wid["PM"], 100 * wid["REML"]))
    chk.eq("signature size under PM / REML", (len(sigs["PM"]), len(sigs["REML"])), (73, 75))
    chk.eq("genes called by all three estimators", len(invariant), 57)
    chk.close("median tau2, DL", E["DL"].tau2.median(), 0.1124, 0.00005)
    chk.close("median tau2, PM", E["PM"].tau2.median(), 0.1434, 0.00005)
    chk.close("median per-gene CI widening, PM (%)", 100 * wid["PM"], 5.65, 0.005)
    chk.close("median per-gene CI widening, REML (%)", 100 * wid["REML"], 5.15, 0.005)

    loo = meta.leave_one_cohort_out(R["Y"], R["S"], sig)
    print(loo.round(3).to_string(index=False))
    chk.close("LOO minimum fraction retained", loo.fraction_retained.min(), 0.493, 0.0005)
    chk.close("LOO maximum fraction retained", loo.fraction_retained.max(), 0.904, 0.0005)
    chk.eq("cohorts whose removal leaves Jaccard < 0.5", int((loo.jaccard_vs_full < 0.5).sum()), 4)
    chk.eq("cohorts whose removal drops > half the signature", int((loo.fraction_retained < 0.5).sum()), 1)

    rep = meta.single_cohort_replication(DE, M)
    print(rep.round(3).to_string(index=False))
    chk.close("median replication rate, eight cohorts (%)", 100 * rep.replication_rate.median(), 1.7, 0.05)
    chk.close("maximum replication rate (%)", 100 * rep.replication_rate.max(), 14.3, 0.05)
    saved["loo"], saved["replication"] = loo, rep

    # ------------------------------------------------------------------ 5
    section("5. Batch structure: PCA variance partition (cohort vs disease)")
    V, n8, g8 = models.pca_primary(H)
    print("  eight cohorts, %d samples, %d genes" % (n8, g8))
    print((V.set_index("PC") * 100).round(2).to_string())
    chk.eq("PCA samples / genes", (n8, g8), (605, 15800))
    chk.close("PC1 variance explained (%)", 100 * V.variance_explained[0], 34.66, 0.005)
    chk.close("PC1 R2 cohort (%)", 100 * V.r2_cohort[0], 93.68, 0.005)
    chk.close("PC1 R2 disease (%)", 100 * V.r2_disease[0], 2.95, 0.005)
    chk.close("PC2 R2 cohort (%)", 100 * V.r2_cohort[1], 95.19, 0.005)
    chk.close("PC3 R2 cohort (%)", 100 * V.r2_cohort[2], 43.60, 0.005)
    chk.eq("cohort R2 exceeds disease R2 more than tenfold on PC1-PC5",
           bool((V.r2_cohort > 10 * V.r2_disease).all()), True)

    # ------------------------------------------------------------------ 6
    section("6. Obesity in the comparator (three contrasts inside GSE126848)")
    ob = strata.obesity()
    print("  %d of %d lean-control DE genes are testable in the obesity-controlled contrast "
          "(%d fail its count filter); %d (%.0f %%) remain significant; r = %.3f"
          % (ob["testable_in_B"], ob["n_A"], ob["untestable_in_B"], ob["retained"],
             100 * ob["retained_frac"], ob["r_A_B_testable"]))
    chk.eq("DE genes A / B / C", (ob["n_A"], ob["n_B"], ob["n_C"]), (3950, 3530, 155))
    chk.eq("testable in B / retained", (ob["testable_in_B"], ob["retained"]), (3936, 2880))
    chk.close("effect-size correlation r", ob["r_A_B_testable"], 0.948, 0.0005)

    section("7. Disease severity (fibrosis strata against each cohort's own controls)")
    sev = strata.severity(H, sig)
    ends = sev.groupby("cohort").median_abs_log2FC_signature.agg(["first", "last"]).round(2)
    for acc, (a, b) in {"GSE135251": (0.91, 1.23), "GSE162694": (0.49, 0.97),
                        "GSE281797": (0.23, 0.48), "GSE185051": (1.03, 1.04)}.items():
        chk.eq("%s lowest -> highest stratum" % acc, (float(ends.loc[acc, "first"]), float(ends.loc[acc, "last"])), (a, b))
    saved["severity"] = sev

    # ------------------------------------------------------------------ 8
    B, floor = meta.expression_floor(DE)
    print("\n  expression floor (25th percentile of median DESeq2 baseMean, eight cohorts) %.2f" % floor)
    chk.close("expression floor", floor, 55.32, 0.005)
    chk.eq("signature genes above the floor", int((B.reindex(sig.index) >= floor).sum()), 46)

    if not args.skip_classifier:
        seeds = (SEED,) if args.quick else SEEDS
        X = EXPECT_SEED1 if args.quick else EXPECT_SEEDS
        tag = "seed %d" % SEED if args.quick else "median over seeds %d-%d" % (seeds[0], seeds[-1])
        section("8. Leave-one-cohort-out classifier (features re-selected inside every fold; %s)" % tag)
        clf = models.Classifier(H, DE)
        print("  matrix %d genes x %d samples" % (clf.Z.shape[0], clf.Z.shape[1]))
        L, I, head = clf.over_seeds(seeds)
        print(head[["AUROC_median", "AUROC_min", "AUROC_max", "calibration_slope_median",
                    "calibration_slope_min", "calibration_slope_max", "brier_median",
                    "internal_AUROC", "internal_calibration_slope"]].round(3).to_string())
        h = head
        chk.close("median LOCO AUROC, elastic net", h.loc["elastic_net", "AUROC_median"], X["en_auroc"], 0.0005)
        chk.close("median LOCO AUROC, gradient boosting", h.loc["grad_boosting", "AUROC_median"], X["gb_auroc"], 0.0005)
        chk.close("median calibration slope, elastic net",
                  h.loc["elastic_net", "calibration_slope_median"], X["en_slope"], 0.0005)
        chk.close("median calibration slope, gradient boosting",
                  h.loc["grad_boosting", "calibration_slope_median"], X["gb_slope"], 0.0005)
        chk.close("internal CV AUROC, elastic net", h.loc["elastic_net", "internal_AUROC"],
                  X["en_internal_auroc"], 0.0005)
        chk.close("internal calibration slope, elastic net",
                  h.loc["elastic_net", "internal_calibration_slope"], X["en_internal_slope"], 0.0005)
        chk.close("internal CV AUROC, gradient boosting", h.loc["grad_boosting", "internal_AUROC"],
                  X["gb_internal_auroc"], 0.0005)
        chk.close("internal calibration slope, gradient boosting",
                  h.loc["grad_boosting", "internal_calibration_slope"], X["gb_internal_slope"], 0.0005)

        section("9. SHAP importance (elastic net, %s) and the cohort-label negative control" % tag)
        feats_all = clf.features(COHORTS)
        imp = clf.shap_importance(feats_all, sym, seeds)
        n_top = int(imp.head(100).ensembl_gene_id.isin(sig.index).sum())
        print("  top-100 SHAP features in the signature: %d" % n_top)
        print("  top 15 features: %s" % ", ".join(imp.head(15).symbol))
        chk.eq("top-100 SHAP features that are signature genes", n_top, X["top100_in_signature"])
        nc = clf.cohort_negative_control(list(imp.ensembl_gene_id))
        print("  cohort-label accuracy: LightGBM %.3f | random forest %.3f | majority baseline %.3f | chance %.3f"
              % (nc["lightgbm"], nc["random_forest"], nc["majority_baseline"], nc["uniform_chance"]))
        chk.close("cohort-label accuracy, LightGBM", nc["lightgbm"], X["lgbm_cohort_acc"], 0.0005)
        chk.close("cohort-label accuracy, random forest", nc["random_forest"], X["rf_cohort_acc"], 0.0005)
        chk.close("majority-class baseline", nc["majority_baseline"], 0.357, 0.0005)

        section("10. Prioritised core and the twelve-gene validation panel (%s)" % tag)
        C, panel, n_core = core.prioritise(M, sig, imp, B, floor)
        tiers = C.tier.value_counts().sort_index()
        panel = panel.assign(symbol=panel.ensembl_gene_id.map(sym))
        print("  genes per tier: %s" % tiers.to_dict())
        print(panel[["symbol", "tier", "in_signature", "pooled_log2FC", "pooled_padj", "k_cohorts"]]
              .round(4).to_string(index=False))
        chk.eq("genes in tiers 1 / 2 / 3", tuple(int(tiers.get(t, 0)) for t in (1, 2, 3)), X["tiers"])
        chk.eq("Tier 1 genes", sorted(panel[panel.tier == 1].symbol), X["tier1"])
        chk.eq("validation panel", sorted(panel.symbol), X["panel"])
        chk.eq("panel genes that are not signature genes",
               sorted(panel[~panel.in_signature].symbol), X["panel_not_signature"])
        saved.update({"loco_by_seed": L, "internal_cv_by_seed": I, "classifier_headline": head,
                      "shap": imp, "core": C, "panel": panel})

    # ------------------------------------------------------------------ 11
    section("11. Target-based repurposing: gates re-run on the dated query snapshot")
    same, total = repurposing.neighbour_directions(M)
    print("  network neighbours whose direction and tier recompute identically: %d of %d" % (same, total))
    chk.eq("neighbour directions reproduced", same, total)
    att, cand, neg = repurposing.run(sig, sym)
    a = att.set_index(["section", "stage"])
    for sec, lab in (("A", "direct-target"), ("B", "network-expanded")):
        s = a.loc[sec]
        print("  %-17s pairs: raw %5d -> approved %4d -> oral %4d -> hepatic %4d -> cardiac %4d -> direction %3d"
              % (lab, s.loc["1_raw_pairs", "n_pairs"], s.loc["3_approved_only", "n_pairs"],
                 s.loc["4_oral", "n_pairs"], s.loc["5_hepatic_safety", "n_pairs"],
                 s.loc["6_cardiac_gate", "n_pairs"], s.loc["7_direction_match", "n_pairs"]))
    drugs = sorted(cand.drug_name_canonical.unique())
    conc = cand.drop_duplicates("drug_name_canonical").concordance.value_counts().to_dict()
    print("  candidates: %d rows, %d drugs: %s" % (len(cand), len(drugs), ", ".join(drugs)))
    print("  mechanism at the nominating target (curated): %s" % conc)
    chk.eq("attrition, every stage and section, equals the reference run", repurposing.same_attrition(att), True)
    chk.eq("direct-target arm: raw pairs -> candidates",
           (int(a.loc[("A", "1_raw_pairs"), "n_pairs"]), int(a.loc[("A", "7_direction_match"), "n_pairs"])), (122, 0))
    chk.eq("network arm: raw pairs, cardiac, direction-matched",
           tuple(int(a.loc[("B", s), "n_pairs"]) for s in ("1_raw_pairs", "6_cardiac_gate", "7_direction_match")),
           (4617, 248, 8))
    chk.eq("final candidate rows / drugs", (len(cand), len(drugs)), (8, 5))
    chk.eq("drug-target pairs failing the direction gate", len(neg), 13)
    chk.eq("mechanism at target: concordant / discordant / not established",
           (conc.get("CONCORDANT", 0), conc.get("DISCORDANT", 0), conc.get("NOT ESTABLISHED", 0)), (1, 3, 1))
    saved.update({"repurposing_attrition": att, "repurposing_candidates": cand})

    # ------------------------------------------------------------------ done
    section("Summary")
    ok = chk.summary()
    if args.save:
        RESULTS.mkdir(exist_ok=True)
        for name, df in saved.items():
            df.to_csv(RESULTS / f"{name}.tsv", sep="\t")
        print("tables written to %s" % RESULTS)
    print("elapsed %.1f min" % ((time.time() - t0) / 60))
    return 0 if ok else 1


if __name__ == "__main__":
    np.seterr(all="ignore")
    sys.exit(main())
