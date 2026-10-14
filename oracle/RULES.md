# Rules the oracle implements (pre-specified; derived from the documented behaviour of NeuroVF)

All rules are re-implemented from scratch in `oracle_reference.py` using only NumPy and SciPy
distribution functions. Nothing is imported or copied from NeuroVF.

**R1 Grid.** Point order and coordinates come from `data/databases/reference_points.json`
(a data table; it is also compared with the grid NeuroVF reports in each export's `meta.json`).
**R2 Mirror.** mirror(i) = index j with (x_j, y_j) = (-x_i, y_i); if none exists (24-2 asymmetry) mirror(i) = i.
A mirrored array is `v[mirror]`.
**R3 Registration.** chiasmatic / ptc: the eye named in `__vf_flip_eye__` is mirrored for every subject, then
combined = (OD + OS)/2 where BOTH are finite, else NaN (complete-pair rule; no nanmean across eyes).
post_chiasmatic: no per-eye mirror; combined = (OD + OS)/2 (complete pair); then the WHOLE combined field is
mirrored if the subject's surgery-side value differs from `__vf_surgery_side_reference_value__`.
View "od"/"os": that eye exactly as recorded (never mirrored).
**R4 Longitudinal.** delta = postop - preop of the registered vector of the same view (NaN propagates).
**R5 Cohort mean.** Over included subjects: location available only if EVERY subject is finite there
(complete-case); mean = mean of finite values; otherwise NaN. valid_n = count of finite values.
**R6 Category means.** R5 within each category value (sorted unique values), subjects with a non-empty value.
**R7 Group comparison, two levels (A = first, B = second of the sorted levels).**
Per point, only if every subject in both groups is finite there AND (sd_A > 1e-6 or sd_B > 1e-6):
Welch t = (mean_A - mean_B)/sqrt(s2_A/n_A + s2_B/n_B); Satterthwaite df; two-sided p from the t distribution;
Hedges' g = J(df) * (mean_B - mean_A)/s_pooled with s_pooled = sqrt(((n_A-1)s2_A + (n_B-1)s2_B)/(n_A+n_B-2)), df = n_A+n_B-2 and the exact
J(df) = Gamma(df/2)/(sqrt(df/2)*Gamma((df-1)/2)); g is NaN if s_pooled <= 1e-6. Difference = mean_B - mean_A where both group means exist.
q = Benjamini-Hochberg over all finite p (family = computable locations). n_unc = #(p<0.05); n_sig = #(q<=0.05).
Non-computable locations are recorded by reason: incomplete group (a subject of either group is missing there) or zero-variance group (both sd <= 1e-6).
**R14 Group comparison, three or more levels.** Analysed groups = sorted levels with >= 2 subjects in the view (excluded levels are recorded;
if only two remain R7 applies). A location is computable only if every subject of every group is finite and every group has sd > 1e-6.
Omnibus: Welch heteroscedastic ANOVA, w_i = n_i/s2_i, F = [sum w_i (m_i - m_w)^2/(k-1)] / [1 + 2(k-2)/(k^2-1) * sum (1-w_i/W)^2/(n_i-1)],
df1 = k-1, df2 = (k^2-1) / (3 * sum (1-w_i/W)^2/(n_i-1)), p from the F distribution; BH-FDR across computable locations.
Omnibus effect sizes (numerical output only): eta^2 = SSB/(SSB+SSW) and omega^2 = (SSB-(k-1)MSW)/(SSB+SSW+MSW) from the classical decomposition (omega^2 is not truncated at 0).
Post hoc (numerical output only), only at locations with omnibus BH-FDR q <= 0.05: all k(k-1)/2 unique contrasts (i<j), diff = mean_j - mean_i,
Welch t and df per pair, p = studentized-range survival function at |t|*sqrt(2) with k groups and the Welch df (Games-Howell; family-wise error
controlled across the contrasts at that location), pairwise Hedges' g as in R7; "significant" = p < 0.05. No effect size is mapped; none is shown in the figures.
Recorded with every result: number of groups, number of unique contrasts, correction text, valid n per group and location, effect sizes, and non-computable locations by reason.

**R8 Correlation.** Subjects with a finite covariate. Per point: n_valid = finite count; needs n_valid >= 3 and
Pearson: sd(values) > 1e-6; Spearman: > 2 distinct values. Pearson r (or Spearman rho = Pearson on average ranks);
p two-sided from t = r*sqrt((n-2)/(1-r^2)), df = n-2. q = BH over finite p.
**R9 Cluster / cumulative (Full Field).** valid = finite values. Cluster bands (TD / raw):
severe: v <= -12.01 / <= 14 (and > plausible minimum -40 / 0); moderate: (-12.01,-6.01] / (14,27]; normal: > -6.01 / > 27.
Cumulative: severe (<= severe threshold) and moderate-or-worse (<= moderate threshold). Percent = 100*count/n_valid per
subject; then OLS regression of percent on the covariate (slope, intercept, r, two-sided p, df = n-2).
**R10 MD (PTC).** device-reported if both eyes' md_manual_<tp>_<eye> exist: (OD+OS)/2, n = 2; single-eye view: that eye's value, n = 1;
else (TD databases only) mean of finite registered values. Tier: <= -12.01 Severe/Advanced; <= -6.01 Moderate; else Early/Mild.
MD vs variable: cross = MD at the timepoint; longitudinal = MD(postop) - MD(preop); continuous -> OLS; categorical ->
groups = sorted unique values, Welch t between the first two groups.
**R11 Scotoma area.** Region mask (Superior Left: y>0 & x<0; Superior Right: y>0 & x>0; Inferior Left: y<0 & x<0;
Inferior Right: y<0 & x>0; 2 Left Quarters: x<0; 2 Right Quarters: x>0; Full Field: all); valid = finite & region.
Chiasmatic / PTC report THREE views: "registered" (R3 vector, complete-pair), "od" and "os" (that eye exactly as recorded, never mirrored,
its own valid locations only). Other study types report the registered view only.
For each view: percent severe / moderate / normal and cumulative bands of that view's valid count, plus the valid count.

**R12 Correlation split by a categorical variable.** For each level of the chosen categorical variable (subjects with a finite covariate and a non-empty level),
R8 is applied to that level's subjects only (needs >= 3 subjects, else no coefficients). BH-FDR is applied separately within each level.
A level has displayable statistics only if >= 10 locations are computable (`stats_available`); n_unc / n_sig then count p<0.05 / q<=0.05 within the level.
The split is evaluated for the registered view and for each eye as recorded (OD, OS) -- never mirrored -- and the OD|OS pair path must reproduce the per-eye numbers.
**R13 OD|OS pair path.** In the second window of a chiasmatic / PTC analysis, mean maps, group comparisons and correlations are computed once per eye on that eye as recorded;
they must equal R3's single-eye views.
