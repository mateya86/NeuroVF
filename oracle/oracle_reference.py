"""
oracle_reference.py -- independent re-implementation of the numerical rules in RULES.md.

Imports NOTHING from NeuroVF. Uses only NumPy and SciPy's t-distribution /
rank functions. Every statistic is computed from first principles (closed-form
formulas) so the oracle does not share an implementation with the code under
test; a second SciPy-based value is exposed only for the oracle's own self-test.
"""
import json
import warnings
import numpy as np
from scipy import stats as _st

warnings.filterwarnings("ignore", category=RuntimeWarning)   # all-NaN columns / constant series are handled explicitly

# ---- constants prespecified in RULES.md ------------------------------------------------------
BANDS = {
    "total_deviation": {"severe": -12.01, "moderate": -6.01, "plausible_min": -40.0},
    "raw_threshold":   {"severe": 14.0,   "moderate": 27.0,  "plausible_min": 0.0},
}
REGIONS = ("Full Field", "Superior Left", "Superior Right", "Inferior Left",
           "Inferior Right", "2 Left Quarters", "2 Right Quarters")


def region_mask(points, region):
    x, y = points[:, 0], points[:, 1]
    if region == "Superior Left":  return (y > 0) & (x < 0)
    if region == "Superior Right": return (y > 0) & (x > 0)
    if region == "Inferior Left":  return (y < 0) & (x < 0)
    if region == "Inferior Right": return (y < 0) & (x > 0)
    if region == "2 Left Quarters":  return x < 0
    if region == "2 Right Quarters": return x > 0
    return np.ones(len(points), bool)


# ---- geometry ---------------------------------------------------------------------------------------
def mirror_index(points):
    n = len(points)
    out = np.arange(n)
    for i, (x, y) in enumerate(points):
        j = np.where((np.isclose(points[:, 0], -x)) & (np.isclose(points[:, 1], y)))[0]
        if len(j) == 1:
            out[i] = j[0]
    return out


def arr(rec, key):
    v = rec.get(key)
    if v is None:
        return None
    return np.array([np.nan if e is None else float(e) for e in v], float)


# ---- registration (R3, R4) -------------------------------------------------------------------------------
class Study:
    """Everything the oracle needs to know about one database, read from the JSON itself."""
    def __init__(self, db, points):
        self.db = db
        self.points = np.asarray(points, float)
        self.mirror = mirror_index(self.points)
        self.study_type = db["__vf_study_type__"]
        self.data_type = db["__vf_data_type__"]
        self.pattern = db["__vf_grid_pattern__"]
        self.flip_eye = db.get("__vf_flip_eye__")
        self.surg_var = db.get("__vf_surgery_side_variable__")
        self.surg_ref = db.get("__vf_surgery_side_reference_value__")
        self.schema = db["__schema__"]
        self.subjects = sorted(k for k in db if not k.startswith("__"))
        self.continuous = [k for k, v in self.schema.items() if v["type"] == "Continuous"]
        self.categorical = [k for k, v in self.schema.items() if v["type"] == "Categorical"]
        self.bands = BANDS[self.data_type]

    def registered(self, sid, tp, view):
        """Return the expected 1-D vector (NaN = excluded) or None."""
        rec = self.db[sid]
        od, os_ = arr(rec, f"vf_raw76_{tp}_od"), arr(rec, f"vf_raw76_{tp}_os")
        if view == "od":  return od
        if view == "os":  return os_
        if od is None or os_ is None:
            return None
        if self.study_type in ("chiasmatic", "ptc"):
            if self.flip_eye == "od": od = od[self.mirror]
            else:                     os_ = os_[self.mirror]
        both = np.isfinite(od) & np.isfinite(os_)
        comb = np.full(len(od), np.nan)
        comb[both] = (od[both] + os_[both]) / 2.0
        if self.study_type == "post_chiasmatic" and self.surg_var:
            val = str(rec.get(self.surg_var, "")).strip()
            if val and val != str(self.surg_ref).strip():
                comb = comb[self.mirror]
        return comb

    def vector(self, sid, mode, timepoint, view):
        if mode == "longitudinal":
            a, b = self.registered(sid, "preop", view), self.registered(sid, "postop", view)
            return None if a is None or b is None else b - a
        return self.registered(sid, "postop" if timepoint == "postop" else "preop", view)


# ---- statistics from first principles -------------------------------------------------------------------------
def bh_fdr(p):
    p = np.asarray(p, float)
    q = np.full(p.shape, np.nan)
    idx = np.where(np.isfinite(p))[0]
    m = len(idx)
    if m == 0:
        return q
    order = idx[np.argsort(p[idx], kind="mergesort")]
    ranked = p[order] * m / (np.arange(m) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    q[order] = np.minimum(ranked, 1.0)
    return q


def two_sided_p_from_t(t, df):
    return float(2.0 * _st.t.sf(abs(t), df))


def welch(a, b):
    """a, b: 1-D arrays. Returns (t, df, p) with t = (mean_a - mean_b)/se (group A minus group B)."""
    na, nb = len(a), len(b)
    va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
    sa, sb = va / na, vb / nb
    se = np.sqrt(sa + sb)
    t = (np.mean(a) - np.mean(b)) / se
    df = (sa + sb) ** 2 / (sa ** 2 / (na - 1) + sb ** 2 / (nb - 1))
    return float(t), float(df), two_sided_p_from_t(t, df)


def pearson(x, y):
    n = len(x)
    xm, ym = x - x.mean(), y - y.mean()
    sxx, syy, sxy = (xm ** 2).sum(), (ym ** 2).sum(), (xm * ym).sum()
    r = sxy / np.sqrt(sxx * syy)
    if abs(r) >= 1.0:
        return float(r), 0.0
    t = r * np.sqrt((n - 2) / (1 - r ** 2))
    return float(r), two_sided_p_from_t(t, n - 2)


def spearman(x, y):
    return pearson(_st.rankdata(x, method="average"), _st.rankdata(y, method="average"))


def ols(x, y):
    n = len(x)
    xm, ym = x - x.mean(), y - y.mean()
    sxx, syy, sxy = (xm ** 2).sum(), (ym ** 2).sum(), (xm * ym).sum()
    slope = sxy / sxx
    r, p = pearson(x, y)
    return {"slope": float(slope), "intercept": float(y.mean() - slope * x.mean()), "r": r, "p": p, "n": n}


# ---- analyses --------------------------------------------------------------------------------------------------
def cohort_mean(S, mode, timepoint, view, sids=None):
    sids = sids if sids is not None else S.subjects
    rows = [(s, S.vector(s, mode, timepoint, view)) for s in sids]
    rows = [(s, v) for s, v in rows if v is not None]
    if not rows:
        return None
    M = np.array([v for _, v in rows])
    n = len(rows)
    valid_n = np.isfinite(M).sum(axis=0)
    unavailable = valid_n < n
    with np.errstate(all="ignore"):
        mean = np.nanmean(M, axis=0)
    mean[unavailable] = np.nan
    return {"sids": [s for s, _ in rows], "n": n, "valid_n": valid_n, "unavailable": unavailable, "mean": mean}


# ---- group comparison: two levels (Welch t / Hedges g), three or more (Welch ANOVA / Games-Howell) -------------------
SD_EPS = 1e-6
ALPHA = 0.05


def hedges_j(df):
    """Exact J(df) = Gamma(df/2)/(sqrt(df/2)*Gamma((df-1)/2)), via log-gamma."""
    from math import lgamma, exp, sqrt
    return exp(lgamma(df / 2.0) - lgamma((df - 1.0) / 2.0)) / sqrt(df / 2.0)


def hedges_g(a, b):
    """g = J * (mean_b - mean_a) / pooled sd (pooled over the two groups, weights n-1); NaN if pooled sd <= SD_EPS."""
    na, nb = len(a), len(b)
    df = na + nb - 2
    sp = np.sqrt(((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1)) / df)
    if not sp > SD_EPS:
        return float("nan")
    return float(hedges_j(df) * (np.mean(b) - np.mean(a)) / sp)


def welch_anova(groups):
    """Welch heteroscedastic one-way ANOVA from first principles; None if not computable. Also eta^2 / omega^2 (classical SS)."""
    k = len(groups)
    n = np.array([len(g) for g in groups], float)
    m = np.array([np.mean(g) for g in groups]); v = np.array([np.var(g, ddof=1) for g in groups])
    if np.any(n < 2) or np.any(np.sqrt(v) <= SD_EPS):
        return None
    w = n / v; W = w.sum(); mw = np.sum(w * m) / W
    tmp = np.sum((1 - w / W) ** 2 / (n - 1))
    F = (np.sum(w * (m - mw) ** 2) / (k - 1)) / (1 + 2 * (k - 2) / (k ** 2 - 1) * tmp)
    df2 = (k ** 2 - 1) / (3 * tmp)
    N = n.sum(); gm = np.sum(n * m) / N
    ssb = np.sum(n * (m - gm) ** 2); ssw = np.sum((n - 1) * v); msw = ssw / (N - k)
    return {"F": float(F), "df1": float(k - 1), "df2": float(df2), "p": float(_st.f.sf(F, k - 1, df2)),
            "eta_sq": float(ssb / (ssb + ssw)), "omega_sq": float((ssb - (k - 1) * msw) / (ssb + ssw + msw))}


def games_howell(groups):
    """All unique pairs i<j: diff = mean_j - mean_i, Welch t, Welch df, p from the studentized range distribution (k = number of groups)."""
    k = len(groups); out = {}
    n = [len(g) for g in groups]; m = [np.mean(g) for g in groups]; v = [np.var(g, ddof=1) for g in groups]
    for i in range(k):
        for j in range(i + 1, k):
            se2 = v[i] / n[i] + v[j] / n[j]
            t = (m[j] - m[i]) / np.sqrt(se2)
            df = se2 ** 2 / ((v[i] / n[i]) ** 2 / (n[i] - 1) + (v[j] / n[j]) ** 2 / (n[j] - 1))
            out[(i, j)] = {"diff": float(m[j] - m[i]), "t": float(t), "df": float(df),
                           "p": float(_st.studentized_range.sf(abs(t) * np.sqrt(2.0), k, df)),
                           "hedges_g": hedges_g(groups[i], groups[j])}
    return out


def levels_usable(S, mode, timepoint, view, variable):
    """Levels (sorted) with >= 2 subjects in this view, and the excluded {level: n}."""
    vals = {}
    for s in S.subjects:
        if S.vector(s, mode, timepoint, view) is None:
            continue
        v = str(S.db[s].get(variable, "")).strip()
        if v:
            vals.setdefault(v, []).append(s)
    allv = sorted(vals)
    return [lv for lv in allv if len(vals[lv]) >= 2], {lv: len(vals[lv]) for lv in allv if len(vals[lv]) < 2}, len(allv)


def group_workflow_arrays(mats, labels):
    """mats: list of (n_i x n_points) arrays, one per level (NaN = missing). Returns the full expected result.
    kind 'two' (Welch t + Hedges g) for 2 levels, 'multi' (Welch ANOVA + Games-Howell) for >= 3."""
    k = len(mats); n_pts = mats[0].shape[1]; n = [m.shape[0] for m in mats]
    nanv = lambda: np.full(n_pts, np.nan)
    inc = np.zeros(n_pts, bool); zv = np.zeros(n_pts, bool)
    n_valid = np.array([np.isfinite(m).sum(0) for m in mats])
    with np.errstate(all="ignore"):
        mean = np.array([np.nanmean(m, 0) for m in mats])
    for gi in range(k):
        mean[gi][n_valid[gi] < n[gi]] = np.nan
    if k == 2:
        t, df, p, g = nanv(), nanv(), nanv(), nanv()
        for i in range(n_pts):
            a, b = mats[0][:, i], mats[1][:, i]
            av, bv = a[np.isfinite(a)], b[np.isfinite(b)]
            if len(av) < n[0] or len(bv) < n[1]:
                inc[i] = True; continue
            if np.std(av) > SD_EPS or np.std(bv) > SD_EPS:
                t[i], df[i], p[i] = welch(av, bv)
                g[i] = hedges_g(av, bv)
            else:
                zv[i] = True
        q = bh_fdr(p)
        ok = np.isfinite(mean[0]) & np.isfinite(mean[1])
        diff = nanv(); diff[ok] = mean[1][ok] - mean[0][ok]
        return {"kind": "two", "n_groups": 2, "n_contrasts": 1, "n": n, "n_valid": n_valid, "mean": mean, "diff": diff,
                "difference_valid_mask": ok, "t": t, "df": df, "p": p, "q": q, "hedges_g": g,
                "reason_incomplete": inc, "reason_zero_variance": zv, "computable": np.isfinite(p),
                "family_n": int(np.isfinite(p).sum()), "n_unc": int(np.nansum(p < ALPHA)), "n_sig": int(np.nansum(q <= ALPHA))}
    F, df2, p, eta, om = nanv(), nanv(), nanv(), nanv(), nanv()
    fins = [None] * n_pts
    for i in range(n_pts):
        fin = [m[:, i][np.isfinite(m[:, i])] for m in mats]
        if any(len(f) < nn for f, nn in zip(fin, n)):
            inc[i] = True; continue
        r = welch_anova(fin)
        if r is None:
            zv[i] = True; continue
        F[i], df2[i], p[i], eta[i], om[i] = r["F"], r["df2"], r["p"], r["eta_sq"], r["omega_sq"]
        fins[i] = fin
    q = bh_fdr(p)
    gate = np.isfinite(q) & (q <= ALPHA)
    post = {}
    for i in range(k):
        for j in range(i + 1, k):
            post[(i, j)] = {key: nanv() for key in ("diff", "t", "df", "p", "hedges_g")}
    for loc in np.where(gate)[0]:
        for key, d in games_howell(fins[loc]).items():
            for f_ in ("diff", "t", "df", "p", "hedges_g"):
                post[key][f_][loc] = d[f_]
    for d in post.values():
        d["significant"] = np.isfinite(d["p"]) & (d["p"] < ALPHA)
    return {"kind": "multi", "n_groups": k, "n_contrasts": k * (k - 1) // 2, "n": n, "n_valid": n_valid, "mean": mean,
            "F": F, "df1": float(k - 1), "df2": df2, "p": p, "q": q, "eta_sq": eta, "omega_sq": om,
            "reason_incomplete": inc, "reason_zero_variance": zv, "computable": np.isfinite(p), "gate": gate,
            "posthoc": post, "family_n": int(np.isfinite(p).sum()), "n_unc": int(np.nansum(p < ALPHA)),
            "n_sig": int(np.nansum(q <= ALPHA)), "n_posthoc_locations": int(gate.sum())}


def group_comparison(S, mode, timepoint, view, variable, levels=None):
    """Expected group-comparison result for one variable/view. levels: the groups analysed (default: levels with >= 2 subjects
    in this view). Returns None if fewer than 2 usable levels or any analysed level has < 2 subjects in this view."""
    if levels is None:
        levels, excluded, _ = levels_usable(S, mode, timepoint, view, variable)
    rows = {lv: [] for lv in levels}; sids = {lv: [] for lv in levels}
    for s in S.subjects:
        v = S.vector(s, mode, timepoint, view)
        if v is None:
            continue
        lv = str(S.db[s].get(variable, "")).strip()
        if lv in rows:
            rows[lv].append(v); sids[lv].append(s)
    if len(levels) < 2 or any(len(rows[lv]) < 2 for lv in levels):
        return None
    out = group_workflow_arrays([np.array(rows[lv]) for lv in levels], list(levels))
    out["levels"] = list(levels); out["sids"] = sids
    return out


def correlation(S, mode, timepoint, view, variable, method, sids=None):
    """sids: restrict to these subjects (used for the within-category analyses)."""
    sids_in, sids, xs, rows = (sids if sids is not None else S.subjects), [], [], []
    for s in sids_in:
        v = S.vector(s, mode, timepoint, view)
        if v is None:
            continue
        try:
            xv = float(S.db[s].get(variable, ""))
        except (ValueError, TypeError):
            continue
        sids.append(s); xs.append(xv); rows.append(v)
    if len(xs) < 3:
        return None
    M, x = np.array(rows), np.array(xs)
    n_pts = M.shape[1]
    coef, p = np.full(n_pts, np.nan), np.full(n_pts, np.nan)
    for i in range(n_pts):
        ok = np.isfinite(M[:, i])
        if ok.sum() < 3:
            continue
        pix = M[ok, i]
        if method == "spearman":
            if len(np.unique(pix)) > 2:
                coef[i], p[i] = spearman(pix, x[ok])
        elif np.std(pix) > 1e-6:
            coef[i], p[i] = pearson(pix, x[ok])
    q = bh_fdr(p)
    return {"sids": sids, "x": x, "matrix": M, "n_valid": np.isfinite(M).sum(0), "coef": coef, "p": p, "q": q,
            "computable": np.isfinite(coef) & np.isfinite(p), "family_n": int(np.isfinite(p).sum()),
            "n_unc": int(np.nansum(p < 0.05)), "n_sig": int(np.nansum(q <= 0.05))}


def cluster_cumulative(S, kind, timepoint, view, variable):
    b = S.bands
    xs, sids, pct = [], [], {}
    names = ["severe", "moderate", "normal"] if kind == "cluster" else ["severe", "moderate_or_worse"]
    for nme in names:
        pct[nme] = []
    for s in S.subjects:
        v = S.vector(s, "cross", timepoint, view)
        if v is None or not S.db[s].get(variable):
            continue
        try:
            xv = float(S.db[s][variable])
        except (ValueError, TypeError):
            continue
        vv = v[np.isfinite(v)]
        if len(vv) == 0:
            continue
        xs.append(xv); sids.append(s)
        if kind == "cluster":
            pct["severe"].append(100.0 * np.sum((vv > b["plausible_min"]) & (vv <= b["severe"])) / len(vv))
            pct["moderate"].append(100.0 * np.sum((vv > b["severe"]) & (vv <= b["moderate"])) / len(vv))
            pct["normal"].append(100.0 * np.sum(vv > b["moderate"]) / len(vv))
        else:
            pct["severe"].append(100.0 * np.sum(vv <= b["severe"]) / len(vv))
            pct["moderate_or_worse"].append(100.0 * np.sum(vv <= b["moderate"]) / len(vv))
    if len(xs) < 2:
        return None
    x = np.array(xs)
    return {"sids": sids, "x": x, "pct": {k: np.array(v) for k, v in pct.items()},
            "fits": {k: ols(x, np.array(v)) for k, v in pct.items()}}


def md_expected(S, sid, tp, view):
    """(md, n_points, tier, source) or None. tp in {'preop','postop'}."""
    rec = S.db[sid]
    od, os_ = rec.get(f"md_manual_{tp}_od"), rec.get(f"md_manual_{tp}_os")
    od = None if od in (None, "") else float(od)
    os_ = None if os_ in (None, "") else float(os_)
    tier = lambda m: ("Severe/Advanced" if m <= -12.01 else "Moderate" if m <= -6.01 else "Early/Mild")
    if view in ("od", "os"):
        m = od if view == "od" else os_
        if m is not None:
            return m, 1, tier(m), "device-reported"
    elif od is not None and os_ is not None:
        m = (od + os_) / 2.0
        return m, 2, tier(m), "device-reported"
    if S.data_type != "total_deviation":
        return None
    v = S.registered(sid, tp, view)
    if v is None or not np.isfinite(v).any():
        return None
    fin = v[np.isfinite(v)]
    m = float(fin.mean())
    return m, int(len(fin)), tier(m), "approximated"


def md_vs_variable(S, mode, timepoint, view, variable):
    cat = variable in S.categorical
    sids, vals, md = [], [], []
    for s in S.subjects:
        raw = S.db[s].get(variable, "")
        if raw in (None, ""):
            continue
        if mode == "longitudinal":
            a, b = md_expected(S, s, "preop", view), md_expected(S, s, "postop", view)
            if a is None or b is None:
                continue
            m = b[0] - a[0]
        else:
            r = md_expected(S, s, "postop" if timepoint == "postop" else "preop", view)
            if r is None:
                continue
            m = r[0]
        if cat:
            vals.append(str(raw))
        else:
            try:
                vals.append(float(raw))
            except (ValueError, TypeError):
                continue
        sids.append(s); md.append(m)
    if len(md) < 4:
        return None
    md = np.array(md, float)
    out = {"sids": sids, "values": vals, "md": md, "is_categorical": cat}
    if cat:
        groups = sorted(set(vals))
        out["groups"] = groups
        if len(groups) == 2:
            g0 = md[[v == groups[0] for v in vals]]; g1 = md[[v == groups[1] for v in vals]]
            if len(g0) >= 2 and len(g1) >= 2:
                t, df, p = welch(g0, g1)
                out["welch"] = {"t": t, "p": p, "n_a": len(g0), "n_b": len(g1),
                                "mean_a": float(g0.mean()), "mean_b": float(g1.mean())}
    else:
        out["fit"] = ols(np.array(vals, float), md)
    return out


def scotoma_views(S):
    """Views the scotoma-area report shows: chiasmatic / PTC -> registered, then OD, then OS (as recorded); others -> registered only."""
    return ("registered", "od", "os") if S.study_type in ("chiasmatic", "ptc") else ("registered",)


def scotoma(S, sid, tp, region, view="registered"):
    """view 'registered': complete-pair registered vector; 'od'/'os': that eye exactly as recorded (own valid locations only)."""
    v = S.registered(sid, tp, view)
    if v is None:
        return None
    b = S.bands
    valid = np.isfinite(v) & region_mask(S.points, region)
    n = int(valid.sum())
    if n == 0:
        return None
    sev = valid & (v <= b["severe"])
    mod = valid & (v > b["severe"]) & (v <= b["moderate"])
    nor = valid & (v > b["moderate"])
    pc = lambda m: 100.0 * float(m.sum()) / n
    return [pc(sev), pc(mod), pc(nor), pc(sev), pc(sev | mod), float(n)]


def correlation_by_category(S, mode, timepoint, view, variable, method, split_by):
    """Correlation computed separately within each level of a categorical variable (R12).
    Returns {category: {"sids", "n", "res" (None if < 3 subjects), "stats_available"}}.
    BH-FDR is applied within each category on that category's own p-values."""
    cats = {}
    for s in S.subjects:
        if S.vector(s, mode, timepoint, view) is None:
            continue
        try:
            float(S.db[s].get(variable, ""))
        except (ValueError, TypeError):
            continue
        cv = str(S.db[s].get(split_by, "")).strip()
        if cv:
            cats.setdefault(cv, []).append(s)
    out = {}
    for c, ids in cats.items():
        r = correlation(S, mode, timepoint, view, variable, method, sids=ids)
        out[c] = {"sids": ids, "n": len(ids), "res": r,
                  "stats_available": bool(r is not None and int(r["computable"].sum()) >= 10)}
    return out
