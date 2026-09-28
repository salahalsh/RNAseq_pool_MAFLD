"""Multiple testing and random-effects pooling (DerSimonian-Laird, Paule-Mandel, REML)."""
import numpy as np
import pandas as pd
from scipy import stats


def bh(p):
    """Benjamini-Hochberg adjusted p-values; NaN stays NaN and is not counted."""
    p = np.asarray(p, float)
    ok = ~np.isnan(p)
    q = np.full_like(p, np.nan)
    pv = p[ok]
    n = pv.size
    if n == 0:
        return q
    order = np.argsort(pv)
    ranked = pv[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(ranked, 0, 1)
    q[ok] = out
    return q


def dersimonian_laird(Y, S):
    """DL random-effects pooling over a genes x cohorts matrix of log2FC (Y) and SE (S)."""
    y = Y.values.astype(float)
    v = np.square(S.values.astype(float))
    ok = np.isfinite(y) & np.isfinite(v) & (v > 0)
    y = np.where(ok, y, np.nan)
    v = np.where(ok, v, np.nan)
    k = ok.sum(1)
    w = np.where(ok, 1.0 / v, 0.0)
    sw = w.sum(1)
    fe = np.divide((w * np.nan_to_num(y)).sum(1), sw, out=np.full(len(y), np.nan), where=sw > 0)
    Q = np.nansum(w * (np.nan_to_num(y) - fe[:, None]) ** 2 * ok, axis=1)
    C = sw - np.divide((w ** 2).sum(1), sw, out=np.zeros_like(sw), where=sw > 0)
    tau2 = np.maximum(0.0, np.divide(Q - (k - 1), C, out=np.zeros_like(Q), where=C > 0))
    wr = np.where(ok, 1.0 / (v + tau2[:, None]), 0.0)
    swr = wr.sum(1)
    pooled = np.divide((wr * np.nan_to_num(y)).sum(1), swr, out=np.full(len(y), np.nan), where=swr > 0)
    se = np.divide(1.0, np.sqrt(swr), out=np.full(len(y), np.nan), where=swr > 0)
    z = pooled / se
    p = 2 * stats.norm.sf(np.abs(z))
    I2 = np.where(Q > 0, np.maximum(0.0, (Q - (k - 1)) / np.where(Q > 0, Q, np.nan)) * 100, 0.0)
    same = ((np.sign(np.nan_to_num(y)) == np.sign(pooled)[:, None]) & ok).sum(1)
    return pd.DataFrame({"pooled_log2FC": pooled, "pooled_SE": se,
                         "ci_low": pooled - 1.96 * se, "ci_high": pooled + 1.96 * se,
                         "z": z, "pvalue": p, "tau2": tau2, "I2": I2, "Q": Q,
                         "k_cohorts": k, "n_agree_direction": same,
                         "direction_consistency": np.divide(same, k, out=np.zeros_like(same, float),
                                                            where=k > 0)}, index=Y.index)


class EstimatorRefit:
    """Refit the pooled estimate under DL, Paule-Mandel and REML on identical inputs."""

    def __init__(self, Y, S):
        ok = Y.notna().values & S.notna().values & (S.values > 0)
        keep = ok.sum(1) >= 1
        self.index = Y.index[keep]
        self.OK = ok[keep]
        self.K = self.OK.sum(1)
        self.Y = np.where(self.OK, Y.values[keep], 0.0)
        self.V = np.where(self.OK, S.values[keep] ** 2, np.inf)
        self.VSAFE = np.where(self.OK, S.values[keep] ** 2, 1.0)
        self.W0 = np.where(self.OK, 1.0 / self.VSAFE, 0.0)

    def tau2_dl(self):
        sw = self.W0.sum(1)
        fixed = (self.W0 * self.Y).sum(1) / sw
        q = (self.W0 * (self.Y - fixed[:, None]) ** 2 * self.OK).sum(1)
        c = sw - (self.W0 ** 2).sum(1) / sw
        return np.clip((q - (self.K - 1)) / np.where(c > 0, c, np.nan), 0, None), q

    def tau2_pm(self, iters=200, tol=1e-8):
        lo = np.zeros(len(self.Y))
        hi = np.full(len(self.Y), 10.0)

        def qstat(t2):
            w = np.where(self.OK, 1.0 / (self.V + t2[:, None]), 0.0)
            th = (w * self.Y).sum(1) / w.sum(1)
            return (w * (self.Y - th[:, None]) ** 2 * self.OK).sum(1) - (self.K - 1)

        zero = qstat(lo) <= 0
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            neg = qstat(mid) < 0
            hi = np.where(neg, mid, hi)
            lo = np.where(neg, lo, mid)
            if np.max(hi - lo) < tol:
                break
        return np.where(zero, 0.0, 0.5 * (lo + hi))

    def tau2_reml(self, iters=100, tol=1e-9):
        t2, _ = self.tau2_dl()
        t2 = np.nan_to_num(t2)
        for _ in range(iters):
            w = np.where(self.OK, 1.0 / (self.V + t2[:, None]), 0.0)
            sw = w.sum(1)
            th = (w * self.Y).sum(1) / sw
            num = (w ** 2 * ((self.Y - th[:, None]) ** 2 - self.VSAFE) * self.OK).sum(1) + (w ** 2).sum(1) / sw
            new = np.clip(num / (w ** 2).sum(1), 0, None)
            done = np.max(np.abs(new - t2)) < tol
            t2 = new
            if done:
                break
        return t2

    def fit(self):
        t2_dl, Q = self.tau2_dl()
        out = {}
        for name, t2 in (("DL", np.nan_to_num(t2_dl)), ("PM", self.tau2_pm()), ("REML", self.tau2_reml())):
            w = np.where(self.OK, 1.0 / (self.V + t2[:, None]), 0.0)
            sw = w.sum(1)
            theta = (w * self.Y).sum(1) / sw
            se = np.sqrt(1.0 / sw)
            p = 2 * stats.norm.sf(np.abs(theta / se))
            out[name] = pd.DataFrame({"pooled_log2FC": theta, "pooled_SE": se,
                                      "ci_width": 2 * 1.96 * se, "pvalue": p, "padj": bh(p),
                                      "tau2": t2, "k_cohorts": self.K}, index=self.index)
        return out
