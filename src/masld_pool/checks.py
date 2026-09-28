"""Compare every recomputed number with the value reported in the paper.

Each check prints PASS or DIFF. Tolerances match the precision at which the paper reports
the number (for example 0.840 is checked to +/- 0.0005)."""


class Checker:
    def __init__(self, log=print):
        self.log = log
        self.results = []

    def _record(self, label, ok, got, expected):
        self.results.append((label, ok))
        self.log("      [%s] %-58s got %-14s paper %s" % ("PASS" if ok else "DIFF", label, got, expected))

    def eq(self, label, got, expected):
        self._record(label, got == expected, got, expected)

    def close(self, label, got, expected, tol):
        ok = got is not None and abs(float(got) - float(expected)) <= tol
        self._record(label, ok, "%.4g" % float(got), expected)

    def summary(self):
        n_ok = sum(ok for _, ok in self.results)
        self.log("\n%d of %d checks reproduce the published values." % (n_ok, len(self.results)))
        for label, ok in self.results:
            if not ok:
                self.log("   DIFF: %s" % label)
        return n_ok == len(self.results)
