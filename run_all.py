"""Reproduce the analysis of the paper from the per-cohort files in data/.

Every stage prints its results to the terminal and checks them against the values reported
in the paper (PASS / DIFF). No figures are drawn.

    python run_all.py                    # full run (about 10-20 minutes on a laptop)
    python run_all.py --skip-classifier  # skip the leave-one-cohort-out models (fast)
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
from masld_pool.config import COHORTS, RESULTS, SEED  # noqa: E402


def section(title):
    print("\n" + "=" * 88 + "\n" + title + "\n" + "=" * 88)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
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
    chk.eq("genes shared by the seven count cohorts", H.genes_shared_counts, 15896)
    chk.eq("genes in the log-expression matrix", H.logexpr.shape[0], 15799)

    # ------------------------------------------------------------------ 2
    section("2. Per-cohort differential expression (case vs control, no covariates)")
    print("  Each cohort is re-analysed from its counts and compared with the DE table that entered\n"
          "  the meta-analysis (shipped in data/cohorts). Downstream stages use the shipped tables\n"
          "  unless --use-recomputed-de is given; see README, 'Numerical reproducibility'.")
    table1 = {"GSE135251": (14242, 6729), "GSE162694": (15597, 7325), "GSE281797": (14027, 2),
              "GSE185051": (15292, 9540), "GSE126848": (12734, 6601), "GSE268360": (13149, 9),
              "GSE260666": (14232, 61), "GSE183229": (15568, 0)}
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
    section("3. Random-effects meta-analysis (DerSimonian-Laird) and the 75-gene signature")
    R = meta.run(DE)
    M, sig = R["M"], R["sig"]
    up, down = int((sig.pooled_log2FC > 0).sum()), int((sig.pooled_log2FC < 0).sum())
    print("  pooled genes %d | with k >= 4 cohorts %d | signature %d (%d up, %d down) | strict |log2FC|>=1: %d"
          % (len(M), int((M.k_cohorts >= 4).sum()), len(sig), up, down, len(R["strict"])))
    top = sig.head(10).copy()
    top.insert(0, "symbol", top.index.map(sym))
    print(top[["symbol", "pooled_log2FC", "ci_low", "ci_high", "padj", "I2", "k_cohorts"]]
          .round(4).to_string())
    chk.eq("genes pooled (BH universe)", len(M), 15867)
    chk.eq("genes with k >= 4", int((M.k_cohorts >= 4).sum()), 14748)
    chk.eq("signature size (up, down)", (len(sig), up, down), (75, 38, 37))
    chk.eq("signature genes with |log2FC| >= 1", int((sig.pooled_log2FC.abs() >= 1).sum()), 18)
    chk.close("median I2 in the signature (%)", sig.I2.median(), 62, 0.5)
    chk.eq("signature genes with I2 > 75 %", int((sig.I2 > 75).sum()), 24)
    cyp = M.loc["ENSG00000160868"]
    chk.close("CYP3A4 pooled log2FC", cyp.pooled_log2FC, -0.64, 0.005)
    chk.close("CYP3A4 adjusted p (x1e7)", cyp.padj * 1e7, 2.0, 0.05)
    saved["meta_all_genes"] = M
    saved["signature"] = sig.assign(symbol=sig.index.map(sym))

    # ------------------------------------------------------------------ 4
    section("4. Robustness: between-study variance estimator, leave-one-cohort-out, replication")
    E, sigs = meta.estimator_sensitivity(R["Y"], R["S"])
    for k, v in E.items():
        print("  %-5s median tau2 %.4f | median pooled SE %.4f | signature %d genes | of the 75: %d"
              % (k, v.tau2.median(), v.pooled_SE.median(), len(sigs[k]), len(sigs[k] & set(sig.index))))
    invariant = sigs["DL"] & sigs["PM"] & sigs["REML"]
    wid = {k: float((E[k].ci_width / E["DL"].ci_width).median() - 1) for k in ("PM", "REML")}
    print("  called by all three estimators: %d | median per-gene CI widening: PM %.1f %%, REML %.1f %%"
          % (len(invariant), 100 * wid["PM"], 100 * wid["REML"]))
    chk.eq("signature size under PM / REML", (len(sigs["PM"]), len(sigs["REML"])), (85, 83))
    chk.eq("genes called by all three estimators", len(invariant), 66)
    chk.close("median tau2, DL", E["DL"].tau2.median(), 0.114, 0.0005)
    chk.close("median tau2, PM", E["PM"].tau2.median(), 0.144, 0.0005)
    chk.close("median per-gene CI widening, PM (%)", 100 * wid["PM"], 6.1, 0.05)
    chk.close("median per-gene CI widening, REML (%)", 100 * wid["REML"], 5.5, 0.05)

    loo = meta.leave_one_cohort_out(R["Y"], R["S"], sig)
    print(loo.round(3).to_string(index=False))
    chk.close("LOO minimum fraction retained", loo.fraction_retained.min(), 0.40, 0.005)
    chk.close("LOO maximum fraction retained", loo.fraction_retained.max(), 0.85, 0.005)
    chk.eq("cohorts whose removal leaves Jaccard < 0.5", int((loo.jaccard_vs_full < 0.5).sum()), 4)
    chk.eq("cohorts whose removal drops > half the signature", int((loo.fraction_retained < 0.5).sum()), 2)

    rep = meta.single_cohort_replication(DE, M)
    print(rep.round(3).to_string(index=False))
    with_genes = rep[rep.top_genes_tested > 0]
    chk.close("median replication rate, 7 cohorts with testable genes (%)",
              100 * with_genes.replication_rate.median(), 0.6, 0.05)
    chk.close("maximum replication rate (%)", 100 * with_genes.replication_rate.max(), 11.1, 0.05)
    saved["loo"], saved["replication"] = loo, rep

    # ------------------------------------------------------------------ 5
    section("5. Batch structure: PCA variance partition (cohort vs disease)")
    (V7, n7, g7), (V8, n8, g8) = models.pca_primary(H)
    print("  primary: seven count-bearing cohorts, %d samples, %d genes" % (n7, g7))
    print((V7.set_index("PC") * 100).round(2).to_string())
    print("  sensitivity: all eight cohorts (GSE183229 on its deposited log scale), %d samples" % n8)
    print((V8.set_index("PC") * 100).round(2).to_string())
    chk.close("PC1 variance explained, 589 samples (%)", 100 * V7.variance_explained[0], 35.4, 0.05)
    chk.close("PC1 R2 cohort (%)", 100 * V7.r2_cohort[0], 93.9, 0.05)
    chk.close("PC1 R2 disease (%)", 100 * V7.r2_disease[0], 3.0, 0.05)
    chk.close("PC2 R2 cohort (%)", 100 * V7.r2_cohort[1], 95.4, 0.05)
    chk.close("PC3 R2 cohort (%)", 100 * V7.r2_cohort[2], 43.4, 0.05)
    chk.close("eight-cohort PC1 R2 cohort (%)", 100 * V8.r2_cohort[0], 96.4, 0.05)
    chk.close("eight-cohort PC1 R2 disease (%)", 100 * V8.r2_disease[0], 4.4, 0.05)

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
    for acc, (a, b) in {"GSE135251": (0.94, 1.27), "GSE162694": (0.49, 1.00),
                        "GSE281797": (0.22, 0.48), "GSE185051": (1.07, 1.03)}.items():
        chk.eq("%s lowest -> highest stratum" % acc, (ends.loc[acc, "first"], ends.loc[acc, "last"]), (a, b))
    saved["severity"] = sev

    # ------------------------------------------------------------------ 8
    B, floor = meta.expression_floor(DE, pd.read_csv(
        io.REF_DIR / "GSE334651_baseMean_heldout.tsv", sep="\t", index_col=0).baseMean)
    _, floor8 = meta.expression_floor(DE)
    print("\n  expression floor (25th percentile of median baseMean) %.1f as published "
          "(the held-out GSE334651 entered this median); %.1f from the eight primary cohorts alone"
          % (floor, floor8))
    chk.close("expression floor", floor, 35.3, 0.05)
    chk.eq("signature genes above the floor", int((B.reindex(sig.index) >= floor).sum()), 44)

    if not args.skip_classifier:
        section("8. Leave-one-cohort-out classifier (features re-selected inside every fold)")
        clf = models.Classifier(H, DE)
        print("  matrix %d genes x %d samples | seed %d" % (clf.Z.shape[0], clf.Z.shape[1], SEED))
        loco = clf.loco()
        med = loco.groupby("model")[["AUROC", "calibration_slope", "brier"]].median()
        print(med.round(3).to_string())
        internal, feats_all = clf.internal_cv()
        print("  internal five-fold CV (optimistic baseline):")
        print(internal.round(3).to_string(index=False))
        chk.close("median LOCO AUROC, elastic net", med.loc["elastic_net", "AUROC"], 0.840, 0.001)
        chk.close("median LOCO AUROC, gradient boosting", med.loc["grad_boosting", "AUROC"], 0.818, 0.001)
        chk.close("median calibration slope, elastic net", med.loc["elastic_net", "calibration_slope"], 1.00, 0.005)
        chk.close("median calibration slope, gradient boosting",
                  med.loc["grad_boosting", "calibration_slope"], 0.318, 0.001)
        chk.close("internal CV AUROC, elastic net",
                  internal.set_index("model").loc["elastic_net", "AUROC"], 0.848, 0.0005)
        chk.close("internal calibration slope, elastic net",
                  internal.set_index("model").loc["elastic_net", "calibration_slope"], 0.384, 0.0005)

        section("9. SHAP importance and the cohort-label negative control")
        best = med.AUROC.idxmax()
        imp = clf.shap_importance(best, feats_all, sym)
        n17 = int(imp.head(100).ensembl_gene_id.isin(sig.index).sum())
        print("  final model: %s | top-100 SHAP features in the signature: %d" % (best, n17))
        print("  top 15 features: %s" % ", ".join(imp.head(15).symbol))
        chk.eq("top-100 SHAP features that are signature genes", n17, 17)
        nc = clf.cohort_negative_control(list(imp.ensembl_gene_id))
        print("  cohort-label accuracy: LightGBM %.3f | random forest %.3f | majority baseline %.3f | chance %.3f"
              % (nc["lightgbm"], nc["random_forest"], nc["majority_baseline"], nc["uniform_chance"]))
        chk.close("cohort-label accuracy, LightGBM", nc["lightgbm"], 0.965, 0.0005)
        chk.close("cohort-label accuracy, random forest", nc["random_forest"], 0.855, 0.0005)
        chk.close("majority-class baseline", nc["majority_baseline"], 0.357, 0.0005)

        section("10. Prioritised core and the twelve-gene validation panel")
        C, panel, n_core = core.prioritise(M, sig, imp, B, floor)
        tiers = C.tier.value_counts().sort_index()
        panel = panel.assign(symbol=panel.ensembl_gene_id.map(sym))
        print("  genes per tier: %s" % tiers.to_dict())
        print(panel[["symbol", "tier", "in_signature", "pooled_log2FC", "pooled_padj", "k_cohorts"]]
              .round(4).to_string(index=False))
        chk.eq("genes in tiers 1 / 2 / 3", tuple(int(tiers.get(t, 0)) for t in (1, 2, 3)), (4, 39, 39))
        chk.eq("Tier 1 genes", sorted(panel[panel.tier == 1].symbol), ["CYP3A4", "ENO3", "FBXO2", "OAT"])
        chk.eq("validation panel", sorted(panel.symbol),
               sorted(["ENO3", "FBXO2", "OAT", "CYP3A4", "MACROH2A2", "MGST3", "PGAP4", "DHRS9",
                       "UGT1A8", "KRTCAP3", "VIL1", "DEFB1"]))
        chk.eq("panel genes that are not signature genes",
               sorted(panel[~panel.in_signature].symbol), ["MGST3", "UGT1A8"])
        saved.update({"loco": loco, "internal_cv": internal, "shap": imp, "core": C, "panel": panel})

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
    chk.eq("direct-target arm: raw pairs -> candidates",
           (int(a.loc[("A", "1_raw_pairs"), "n_pairs"]), int(a.loc[("A", "7_direction_match"), "n_pairs"])), (136, 0))
    chk.eq("network arm: raw pairs, cardiac, direction-matched",
           tuple(int(a.loc[("B", s), "n_pairs"]) for s in ("1_raw_pairs", "6_cardiac_gate", "7_direction_match")),
           (5834, 376, 23))
    chk.eq("final candidate rows / drugs", (len(cand), len(drugs)), (17, 14))
    chk.eq("drug-target pairs failing the direction gate", len(neg), 22)
    chk.eq("mechanism-concordant / discordant drugs", (conc.get("CONCORDANT", 0), conc.get("DISCORDANT", 0)), (11, 3))
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
