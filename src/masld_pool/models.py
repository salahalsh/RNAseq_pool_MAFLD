"""Batch structure (PCA variance partition), the leave-one-cohort-out classifier, SHAP, and
the cohort-label negative control."""
import json
import warnings

import numpy as np
import pandas as pd
from scipy import stats

from .config import COHORTS, LOG_SCALE_COHORTS, N_FEATURES, SEED

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
    """Seven count-bearing cohorts on true log2 CPM (primary), and all eight (sensitivity)."""
    seven = [a for a in COHORTS if a not in LOG_SCALE_COHORTS]
    return variance_partition(H, seven), variance_partition(H, COHORTS)


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
    def models():
        from lightgbm import LGBMClassifier
        from sklearn.linear_model import LogisticRegression
        return {
            "elastic_net": (LogisticRegression(penalty="elasticnet", solver="saga", max_iter=1500,
                                               tol=1e-3, class_weight="balanced", random_state=SEED),
                            {"C": [0.05, 0.5], "l1_ratio": [0.15, 0.85]}),
            "grad_boosting": (LGBMClassifier(random_state=SEED, class_weight="balanced", verbose=-1,
                                             n_estimators=300, subsample=0.8, colsample_bytree=0.5),
                              {"num_leaves": [7, 15], "learning_rate": [0.05, 0.1]}),
        }

    @staticmethod
    def fit_best(est, grid, X, y):
        from sklearn.model_selection import GridSearchCV, StratifiedKFold
        gs = GridSearchCV(est, grid, scoring="roc_auc", n_jobs=2, refit=True,
                          cv=StratifiedKFold(3, shuffle=True, random_state=SEED))
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

    def loco(self, log=print):
        from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
        rows = []
        for held in COHORTS:
            tr, te = self.coh != held, self.coh == held
            feats = self.features([a for a in COHORTS if a != held])
            Xtr, Xte = self.Z.loc[feats].T.values[tr], self.Z.loc[feats].T.values[te]
            line = []
            for name, (est, grid) in self.models().items():
                model, best = self.fit_best(est, grid, Xtr, self.y[tr])
                p = model.predict_proba(Xte)[:, 1]
                sl, it = self.calibration(self.y[te], p)
                rows.append({"held_out": held, "model": name, "n_test": int(te.sum()),
                             "prevalence": float(self.y[te].mean()),
                             "AUROC": roc_auc_score(self.y[te], p),
                             "AUPRC": average_precision_score(self.y[te], p),
                             "brier": brier_score_loss(self.y[te], p),
                             "calibration_slope": sl, "calibration_intercept": it,
                             "best_params": json.dumps(best)})
                line.append("%s AUROC %.3f slope %.2f" % (name, rows[-1]["AUROC"], sl))
            log("    %-10s n=%-3d  %s" % (held, int(te.sum()), " | ".join(line)))
        return pd.DataFrame(rows)

    def internal_cv(self):
        from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
        from sklearn.model_selection import StratifiedKFold, cross_val_predict
        feats = self.features(COHORTS)
        X = self.Z.loc[feats].T.values
        rows = []
        for name, (est, _) in self.models().items():
            if name == "elastic_net":
                # As in the paper, the internal baseline is not tuned: it uses the estimator's
                # defaults (C = 1, and l1_ratio = 0, the scikit-learn >= 1.8 default). Set
                # explicitly so older versions, which reject an unset l1_ratio, behave the same.
                est = est.set_params(l1_ratio=0.0)
            cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
            p = cross_val_predict(est, X, self.y, cv=cv, method="predict_proba", n_jobs=2)[:, 1]
            sl, it = self.calibration(self.y, p)
            rows.append({"model": name, "AUROC": roc_auc_score(self.y, p),
                         "AUPRC": average_precision_score(self.y, p),
                         "brier": brier_score_loss(self.y, p),
                         "calibration_slope": sl, "calibration_intercept": it})
        return pd.DataFrame(rows), feats

    def shap_importance(self, best_model, feats, symbols):
        import shap
        est, grid = self.models()[best_model]
        X = self.Z.loc[feats].T.values
        model, _ = self.fit_best(est, grid, X, self.y)
        if best_model == "grad_boosting":
            sv = shap.TreeExplainer(model).shap_values(X)
            sv = sv[1] if isinstance(sv, list) else sv
        else:
            sv = shap.LinearExplainer(model, X).shap_values(X)
        imp = pd.DataFrame({"ensembl_gene_id": feats, "mean_abs_shap": np.abs(sv).mean(0)})
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
