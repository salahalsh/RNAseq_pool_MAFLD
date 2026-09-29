"""Batch structure (PCA variance partition), the leave-one-cohort-out classifier, SHAP, and
the cohort-label negative control."""
import json
import warnings

import numpy as np
import pandas as pd
from scipy import stats

from .config import COHORTS, N_FEATURES, SEED, SEEDS

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------- PCA
def r2_categorical(score, labels):
    """R^2 of a one-way ANOVA of a component score on a grouping."""
    labels = np.asarray(labels)
    grand = score.mean()
    ssb = sum((labels == g).sum() * (score[labels == g].mean() - grand) ** 2 for g in np.unique(labels))
    return ssb / ((score - grand) ** 2).sum()


def variance_partition(H, cohorts, n_pc=5):
    from sklearn.decomposition import PCA
    meta = H.meta[H.meta.series_accession.isin(cohorts)]
    X = H.logexpr[list(meta.gsm)].dropna()
    Xt = X.T.values - X.T.values.mean(0)
    pca = PCA(n_components=n_pc, random_state=20260903).fit(Xt)
    S = pca.transform(Xt)
    return pd.DataFrame({"PC": ["PC%d" % (i + 1) for i in range(n_pc)],
                         "variance_explained": pca.explained_variance_ratio_,
                         "r2_cohort": [r2_categorical(S[:, i], meta.series_accession.values) for i in range(n_pc)],
                         "r2_disease": [r2_categorical(S[:, i], meta.group_class.values) for i in range(n_pc)]}), len(meta), X.shape[0]


def pca_primary(H):
    """All eight cohorts on log2 CPM."""
    return variance_partition(H, COHORTS)


# ---------------------------------------------------------------- classifier
class Classifier:
    def __init__(self, H, de):
        self.Z = H.z.dropna()
        self.meta = H.meta.set_index("gsm").loc[self.Z.columns].reset_index()
        self.y = (self.meta.group_class == "case").astype(int).values
        self.coh = self.meta.series_accession.values
        self.de = de

    def pooled_p(self, cohorts):
        """DerSimonian-Laird pooled p over the given cohorts only; genes need k >= 3."""
        genes = list(self.Z.index)
        Y = pd.DataFrame({a: self.de[a].log2FC.reindex(genes) for a in cohorts}, index=genes).values
        Sd = pd.DataFrame({a: self.de[a].lfcSE.reindex(genes) for a in cohorts}, index=genes).values
        v = np.square(Sd)
        ok = np.isfinite(Y) & np.isfinite(v) & (v > 0)
        Yn, vn = np.where(ok, Y, np.nan), np.where(ok, v, np.nan)
        k = ok.sum(1)
        w = np.where(ok, 1 / vn, 0.0)
        sw = w.sum(1)
        fe = np.divide((w * np.nan_to_num(Yn)).sum(1), sw, out=np.full(len(Y), np.nan), where=sw > 0)
        Q = np.nansum(w * (np.nan_to_num(Yn) - fe[:, None]) ** 2 * ok, axis=1)
        C = sw - np.divide((w ** 2).sum(1), sw, out=np.zeros_like(sw), where=sw > 0)
        tau2 = np.maximum(0, np.divide(Q - (k - 1), C, out=np.zeros_like(Q), where=C > 0))
        wr = np.where(ok, 1 / (vn + tau2[:, None]), 0.0)
        swr = wr.sum(1)
        pooled = np.divide((wr * np.nan_to_num(Yn)).sum(1), swr, out=np.full(len(Y), np.nan), where=swr > 0)
        se = np.divide(1.0, np.sqrt(swr), out=np.full(len(Y), np.nan), where=swr > 0)
        p = 2 * stats.norm.sf(np.abs(pooled / se))
        return pd.Series(np.where(k >= 3, p, np.nan), index=genes)

    def features(self, cohorts):
        return list(self.pooled_p(cohorts).dropna().sort_values().head(N_FEATURES).index)

    @staticmethod
    def models(seed=SEED):
        from lightgbm import LGBMClassifier
        from sklearn.linear_model import LogisticRegression
        return {
            "elastic_net": (LogisticRegression(penalty="elasticnet", solver="saga", max_iter=1500,
                                               tol=1e-3, class_weight="balanced", random_state=seed),
                            {"C": [0.05, 0.5], "l1_ratio": [0.15, 0.85]}),
            "grad_boosting": (LGBMClassifier(random_state=seed, class_weight="balanced", verbose=-1,
                                             n_estimators=300, subsample=0.8, colsample_bytree=0.5),
                              {"num_leaves": [7, 15], "learning_rate": [0.05, 0.1]}),
        }

    @staticmethod
    def fit_best(est, grid, X, y, seed=SEED):
        from sklearn.model_selection import GridSearchCV, StratifiedKFold
        gs = GridSearchCV(est, grid, scoring="roc_auc", n_jobs=2, refit=True,
                          cv=StratifiedKFold(3, shuffle=True, random_state=seed))
        gs.fit(X, y)
        return gs.best_estimator_, gs.best_params_

    @staticmethod
    def calibration(y, p):
        """Calibration slope (logistic recalibration) and calibration-in-the-large."""
        from sklearn.linear_model import LogisticRegression
        p = np.clip(p, 1e-6, 1 - 1e-6)
        lp = np.log(p / (1 - p))
        if len(np.unique(y)) < 2:
            return np.nan, np.nan
        sl = LogisticRegression(penalty=None, solver="lbfgs", max_iter=2000).fit(lp.reshape(-1, 1), y)
        return float(sl.coef_[0][0]), float(np.mean(y) - np.mean(p))

    def fold_features(self, held):
        """Features selected on the training cohorts only; they do not depend on the seed."""
        if not hasattr(self, "_feat_cache"):
            self._feat_cache = {}
        if held not in self._feat_cache:
            self._feat_cache[held] = self.features([a for a in COHORTS if a != held])
        return self._feat_cache[held]

    def loco(self, seed=SEED, log=print):
        from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
        rows = []
        for held in COHORTS:
            tr, te = self.coh != held, self.coh == held
            feats = self.fold_features(held)
            Xtr, Xte = self.Z.loc[feats].T.values[tr], self.Z.loc[feats].T.values[te]
            line = []
            for name, (est, grid) in self.models(seed).items():
                model, best = self.fit_best(est, grid, Xtr, self.y[tr], seed)
                p = model.predict_proba(Xte)[:, 1]
                sl, it = self.calibration(self.y[te], p)
                rows.append({"seed": seed, "held_out": held, "model": name, "n_test": int(te.sum()),
                             "prevalence": float(self.y[te].mean()),
                             "AUROC": roc_auc_score(self.y[te], p),
                             "AUPRC": average_precision_score(self.y[te], p),
                             "brier": brier_score_loss(self.y[te], p),
                             "calibration_slope": sl, "calibration_intercept": it,
                             "best_params": json.dumps(best)})
                line.append("%s AUROC %.3f slope %.2f" % (name, rows[-1]["AUROC"], sl))
            log("    %-10s n=%-3d  %s" % (held, int(te.sum()), " | ".join(line)))
        return pd.DataFrame(rows)

    def internal_cv(self, seed=SEED):
        """Five-fold CV over the pooled samples, with the same nested hyperparameter search
        inside every training fold (like-for-like with leave-one-cohort-out)."""
        from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
        from sklearn.model_selection import StratifiedKFold
        feats = self.features(COHORTS)
        X = self.Z.loc[feats].T.values
        rows = []
        for name, (est, grid) in self.models(seed).items():
            p = np.zeros(len(self.y))
            for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, self.y):
                model, _ = self.fit_best(est, grid, X[tr], self.y[tr], seed)
                p[te] = model.predict_proba(X[te])[:, 1]
            sl, it = self.calibration(self.y, p)
            rows.append({"seed": seed, "model": name, "AUROC": roc_auc_score(self.y, p),
                         "AUPRC": average_precision_score(self.y, p),
                         "brier": brier_score_loss(self.y, p),
                         "calibration_slope": sl, "calibration_intercept": it})
        return pd.DataFrame(rows), feats

    def over_seeds(self, seeds=SEEDS, log=print):
        """LOCO and internal CV for every seed, and the headline summary: for each model the
        median across seeds of each seed's median over folds (decision R2-09)."""
        L, I = [], []
        for seed in seeds:
            L.append(self.loco(seed, log=lambda *a: None))
            I.append(self.internal_cv(seed)[0])
            m = L[-1].groupby("model")[["AUROC", "calibration_slope"]].median()
            log("    seed %2d  %s" % (seed, "  ".join("%s AUROC %.3f slope %.3f" % (k, r.AUROC, r.calibration_slope)
                                                   for k, r in m.iterrows())))
        L, I = pd.concat(L, ignore_index=True), pd.concat(I, ignore_index=True)
        metrics = ["AUROC", "AUPRC", "brier", "calibration_slope"]
        per_seed = L.groupby(["model", "seed"])[metrics].median()
        head = per_seed.groupby("model").agg(["median", "min", "max"])
        head.columns = ["%s_%s" % c for c in head.columns]
        inner = I.groupby("model")[metrics].median().add_prefix("internal_")
        return L, I, head.join(inner)

    def shap_importance(self, feats, symbols, seeds=SEEDS):
        """Mean |SHAP| of the elastic net refitted on all cohorts, averaged over the seed
        refits, with every sample as the background (exact linear SHAP expectation)."""
        import shap
        X = self.Z.loc[feats].T.values
        imps = []
        for seed in seeds:
            est, grid = self.models(seed)["elastic_net"]
            model, _ = self.fit_best(est, grid, X, self.y, seed)
            sv = shap.LinearExplainer(model, shap.maskers.Independent(X, max_samples=X.shape[0])).shap_values(X)
            imps.append(np.abs(sv).mean(0))
        imp = pd.DataFrame({"ensembl_gene_id": feats, "mean_abs_shap": np.mean(imps, axis=0)})
        imp = imp.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
        imp["symbol"] = imp.ensembl_gene_id.map(symbols).fillna("")
        return imp

    def cohort_negative_control(self, feats):
        """Can the same features identify the source cohort after per-cohort standardisation?"""
        from lightgbm import LGBMClassifier
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import StratifiedKFold, cross_val_predict
        from sklearn.preprocessing import LabelEncoder
        X = self.Z.loc[feats].T.values
        cy = LabelEncoder().fit_transform(self.coh)
        cv = StratifiedKFold(5, shuffle=True, random_state=1)
        out = {}
        for name, est in (("random_forest", RandomForestClassifier(n_estimators=400, random_state=1, n_jobs=2)),
                          ("lightgbm", LGBMClassifier(n_estimators=300, random_state=1, verbose=-1, n_jobs=2))):
            out[name] = float((cross_val_predict(est, X, cy, cv=cv, n_jobs=1) == cy).mean())
        out["majority_baseline"] = float(pd.Series(cy).value_counts(normalize=True).max())
        out["uniform_chance"] = 1.0 / len(np.unique(cy))
        return out
