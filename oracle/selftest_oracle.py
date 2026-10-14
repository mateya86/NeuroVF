#!/usr/bin/env python3
"""selftest_oracle.py -- checks the ORACLE's own formulas against textbook / hand-computed values
(and, only here, against scipy's ready-made routines) so the oracle itself is trustworthy."""
import sys, numpy as np
from scipy import stats
import oracle_reference as O
ok = True
def check(name, cond, detail=""):
    global ok
    ok &= bool(cond); print(("PASS" if cond else "FAIL"), "-", name, detail)

# 1. Welch: published worked example (t = -2.46)
a = np.array([27.5,21.0,19.0,23.6,17.0,17.9,16.9,20.1,21.9,22.6,23.1,19.6,19.0,21.7,21.4])
b = np.array([27.1,22.0,20.8,23.4,23.4,23.5,25.8,22.0,24.8,20.2,21.9,22.1,22.9,20.5,24.4])
t, df, p = O.welch(a, b)
# hand Satterthwaite computation, written out independently of oracle_reference
va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
df_hand = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
check("Welch t = -2.46 (published example) and df equals an independent hand Satterthwaite computation",
      abs(t + 2.46) < 0.005 and abs(df - df_hand) < 1e-12, f"t={t:.4f} df={df:.3f} p={p:.4f}")
r = stats.ttest_ind(a, b, equal_var=False)
check("Welch t and p equal scipy.stats.ttest_ind to 1e-12", abs(t - r.statistic) < 1e-12 and abs(p - r.pvalue) < 1e-12)
check("Welch df equals scipy's to 1e-9", abs(df - r.df) < 1e-9)
# 2. Benjamini-Hochberg: hand computed
q = O.bh_fdr(np.array([0.01, 0.04, 0.03, 0.005, np.nan]))
check("BH-FDR hand example [0.02,0.04,0.04,0.02,NaN]", np.allclose(q[:4], [0.02, 0.04, 0.04, 0.02]) and np.isnan(q[4]), str(q))
rng = np.random.default_rng(1); pp = rng.uniform(size=200)
from scipy.stats import false_discovery_control
check("BH-FDR equals scipy.stats.false_discovery_control on 200 random p", np.allclose(O.bh_fdr(pp), false_discovery_control(pp, method="bh")))
# 3. Pearson / Spearman / OLS
x = np.array([1., 2, 3, 4, 5]); y = np.array([2., 4, 5, 4, 5])
r_, p_ = O.pearson(x, y)
check("Pearson r = 0.774596669 (closed form sqrt(0.6))", abs(r_ - np.sqrt(0.6)) < 1e-12, f"r={r_:.9f}")
check("Pearson p equals scipy.stats.pearsonr", abs(p_ - stats.pearsonr(x, y).pvalue) < 1e-12)
fit = O.ols(x, y)
check("OLS slope 0.6, intercept 2.2 (hand computed)", abs(fit["slope"] - 0.6) < 1e-12 and abs(fit["intercept"] - 2.2) < 1e-12)
xs = rng.normal(size=30); ys = xs + rng.normal(size=30)
rs, ps = O.spearman(xs, ys)
check("Spearman equals scipy.stats.spearmanr (rho and p)", abs(rs - stats.spearmanr(xs, ys).statistic) < 1e-12 and abs(ps - stats.spearmanr(xs, ys).pvalue) < 1e-10)
check("OLS equals scipy.stats.linregress", all(abs(fit[k] - getattr(stats.linregress(x, y), k2)) < 1e-12 for k, k2 in (("slope","slope"),("intercept","intercept"),("r","rvalue"),("p","pvalue"))))
# 4. mirror index
pts = np.array([[-3, 3], [3, 3], [-3, -3], [3, -3], [9, 0]], float)
check("mirror index swaps (-3,3)<->(3,3), (-3,-3)<->(3,-3), leaves unmatched (9,0) alone", list(O.mirror_index(pts)) == [1, 0, 3, 2, 4])
# 5. region masks
p2 = np.array([[-6, 6], [6, 6], [-6, -6], [6, -6], [0, 6], [6, 0]], float)
check("Superior Left = y>0 & x<0 only", list(O.region_mask(p2, "Superior Left")) == [True, False, False, False, False, False])
check("2 Right Quarters excludes x==0", list(O.region_mask(p2, "2 Right Quarters")) == [False, True, False, True, False, True])

# 6. group-comparison statistics: identities and calibration that do not depend on the oracle's own formulas
from scipy.special import gamma as _gamma
for df in (3, 8, 10, 25, 60):
    J_ref = _gamma(df / 2) / (np.sqrt(df / 2) * _gamma((df - 1) / 2))
    check(f"Hedges J({df}) equals the gamma-function definition (scipy.special.gamma)", abs(O.hedges_j(df) - J_ref) < 1e-12, f"{O.hedges_j(df):.6f}")
a_, b_ = np.array([1., 2, 3, 4, 5]), np.array([3., 4, 5, 6, 7])
check("Hedges g worked example: d = 2/sqrt(2.5) = 1.264911, J(8) = 0.902703, g = 1.141839", abs(O.hedges_g(a_, b_) - 1.141839) < 1e-6, f"{O.hedges_g(a_, b_):.6f}")
rng = np.random.default_rng(5); ident = True
for _ in range(200):
    g2 = [rng.normal(0, rng.uniform(.5, 3), int(rng.integers(2, 12))) for _ in range(2)]
    t_, df_, p_ = O.welch(g2[0], g2[1]); wa = O.welch_anova(g2)
    gh = O.games_howell(g2)[(0, 1)]
    ident &= abs(wa["F"] - t_ ** 2) < 1e-9 * max(1, wa["F"]) and abs(wa["df2"] - df_) < 1e-9 * df_ and abs(wa["p"] - p_) < 1e-9 and abs(gh["p"] - p_) < 1e-8
check("k=2 identities (200 random designs): Welch-ANOVA F = t^2 with the Welch df and p; Games-Howell p = two-sided Welch p", ident)
rng = np.random.default_rng(11); P = []; fw = 0; R = 4000
for _ in range(R):
    gs = [rng.normal(0, sd, n) for sd, n in ((1, 6), (2, 9), (0.5, 5))]
    P.append(O.welch_anova(gs)["p"]); fw += min(d["p"] for d in O.games_howell(gs).values()) < 0.05
ks = stats.kstest(P, "uniform").pvalue
check(f"Welch-ANOVA p-values uniform under the null (unequal variances and n, {R} seeded datasets): KS p = {ks:.3f}", ks > 0.01)
check(f"Games-Howell family-wise error under the null = {fw / R:.4f} (target 0.05, accepted 0.040-0.060)", 0.040 <= fw / R <= 0.060)
pw = np.mean([O.welch_anova([rng.normal(mu, 1, 8) for mu in (0, 0.0, 2.0)])["p"] < 0.05 for _ in range(2000)])
check(f"Welch-ANOVA has power for a 2-sd shift of one group of three (n=8): {pw:.2f} > 0.8", pw > 0.8)

print("\nORACLE SELF-TEST:", "ALL PASSED" if ok else "FAILURES"); sys.exit(0 if ok else 1)
