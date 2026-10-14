#!/usr/bin/env python3
"""
run_oracle.py -- independent oracle for NeuroVF.

    python run_oracle.py --databases ../databases/json --exports <dir of exports> --dicom ../dicom --out oracle_report

Reads (a) the synthetic JSON databases, (b) the numerical exports written by
NeuroVF_research.py (one sub-folder per database), (c) the DICOM manifest.
Computes every expected value itself (oracle_reference.py -- no NeuroVF code),
compares with tolerances.json, writes oracle_report.txt / .json / _checks.csv.
Exit status 0 only if every executed check passed.
"""
import argparse, csv, glob, hashlib, json, os, sys, collections
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import oracle_reference as O

TOL = json.load(open(os.path.join(HERE, "tolerances.json")))
POINTS_FILE = os.path.join(HERE, "..", "data", "databases", "reference_points.json")
PATTERN_KEY = {"24-2": "POINTS_24_2", "30-2": "POINTS_30_2", "60-4": "POINTS_60_4"}


class Report:
    def __init__(self):
        self.rows = []
    def add(self, category, db, name, status, err="", detail=""):
        self.rows.append({"category": category, "database": db, "check": name, "status": status,
                          "max_abs_error": err, "detail": detail})
    def counts(self):
        c = collections.Counter(r["status"] for r in self.rows)
        return c["PASS"], c["FAIL"], c["SKIP"]


def nan(a):
    if a is None:
        return None
    return np.array([np.nan if v is None else v for v in a], float) if not isinstance(a, np.ndarray) else a.astype(float)


def cmp_float(rep, cat, db, name, got, exp, tol_key):
    tol = TOL[tol_key]
    g, e = nan(got), nan(exp)
    if g is None or e is None:
        rep.add(cat, db, name, "FAIL", detail=f"missing array (got={'None' if g is None else 'ok'}, expected={'None' if e is None else 'ok'})"); return False
    if g.shape != e.shape:
        rep.add(cat, db, name, "FAIL", detail=f"shape {g.shape} != {e.shape}"); return False
    nan_mismatch = np.isnan(g) != np.isnan(e)
    both = ~np.isnan(g) & ~np.isnan(e)
    diff = np.abs(g[both] - e[both])
    allowed = tol["abs"] + tol["rel"] * np.abs(e[both])
    bad = diff > allowed
    maxerr = float(diff.max()) if diff.size else 0.0
    if nan_mismatch.any() or bad.any():
        idx = list(np.where(nan_mismatch)[0][:5]) + list(np.where(both)[0][np.where(bad)[0][:5]])
        rep.add(cat, db, name, "FAIL", maxerr,
                f"NaN-pattern mismatches={int(nan_mismatch.sum())}; out-of-tolerance={int(bad.sum())}; first indices={[int(i) for i in idx]}"
                f"; example got/expected="
                f"{[(None if np.isnan(g[i]) else float(g[i]), None if np.isnan(e[i]) else float(e[i])) for i in idx[:3]]}")
        return False
    rep.add(cat, db, name, "PASS", maxerr); return True


def cmp_exact(rep, cat, db, name, got, exp):
    if isinstance(got, (list, np.ndarray)) or isinstance(exp, (list, np.ndarray)):
        g = [None if (isinstance(v, float) and np.isnan(v)) else (bool(v) if isinstance(v, (np.bool_,)) else v) for v in (got if got is not None else [])]
        e = [None if (isinstance(v, float) and np.isnan(v)) else (bool(v) if isinstance(v, (np.bool_,)) else v) for v in (exp if exp is not None else [])]
        g = [int(v) if isinstance(v, (np.integer,)) else v for v in g]; e = [int(v) if isinstance(v, (np.integer,)) else v for v in e]
        ok = got is not None and exp is not None and list(g) == list(e)
    else:
        ok = (got == exp)
    rep.add(cat, db, name, "PASS" if ok else "FAIL", detail="" if ok else f"got={str(got)[:120]} expected={str(exp)[:120]}")
    return ok


# ---------------------------------------------------------------------------------------------------------------
def integrity_checks(rep, name, S, index):
    cat = "database-integrity"
    ok_all = True
    bs = index["blind_spot_index"][S.pattern]
    n_pts = len(S.points)
    for sid in S.subjects:
        for eye in ("od", "os"):
            for tp in ("preop", "postop"):
                a = S.db[sid].get(f"vf_raw76_{tp}_{eye}")
                nulls = [i for i, v in enumerate(a) if v is None]
                ok = (len(a) == n_pts and nulls == [bs[eye]])
                vals = [v for v in a if v is not None]
                ok = ok and all(float(v).is_integer() for v in vals)
                if not ok:
                    ok_all = False
                    rep.add(cat, name, f"{sid} {eye} {tp}: one null at the designated blind-spot index; integer dB values",
                            "FAIL", detail=f"len={len(a)} nulls={nulls} expected null at {bs[eye]}")
    if ok_all:
        rep.add(cat, name, "every array: correct length, exactly one null at the designated blind-spot index, integer dB", "PASS")
    # anatomical placement of the designated indices
    p_od, p_os = S.points[bs["od"]], S.points[bs["os"]]
    want = {"24-2": ((15, -3), (-15, -3)), "30-2": ((15, -3), (-15, -3)), "60-4": ((30, -6), (-30, -6))}[S.pattern]
    cmp_exact(rep, cat, name, "blind-spot coordinates (OD temporal=+X, OS temporal=-X convention)",
              [list(map(float, p_od)), list(map(float, p_os))], [list(map(float, want[0])), list(map(float, want[1]))])
    # registered vectors: widespread sand-hatching must not occur
    worst = max(int(np.isnan(S.registered(s, tp, "registered")).sum()) for s in S.subjects for tp in ("preop", "postop"))
    limit = 2 if S.study_type == "post_chiasmatic" else 1
    rep.add(cat, name, f"registered field has at most {limit} missing location(s) per subject", "PASS" if worst <= limit else "FAIL",
            detail=f"worst case {worst}")
    cm = O.cohort_mean(S, "cross", "preop", "registered")
    n_un = int(cm["unavailable"].sum())
    rep.add(cat, name, f"cohort-mean map has at most {limit} unavailable location(s)", "PASS" if n_un <= limit else "FAIL", detail=f"{n_un}")
    rep.add(cat, name, "schema contains no 'Age' variable", "PASS" if "Age" not in S.schema else "FAIL")
    # flip flags stored in the file agree with the study's registration policy
    bad = []
    for sid in S.subjects:
        for eye in ("od", "os"):
            for tp in ("preop", "postop"):
                f = S.db[sid].get(f"vf_flip_{tp}_{eye}")
                exp = (eye == S.flip_eye) if S.study_type != "post_chiasmatic" else (str(S.db[sid].get(S.surg_var)) != str(S.surg_ref))
                if f != exp: bad.append((sid, eye, tp))
    rep.add(cat, name, "stored vf_flip_* flags agree with the registration policy", "PASS" if not bad else "FAIL", detail=str(bad[:3]))


def pair_checks(rep, dbdir, files):
    cat = "database-integrity"
    for study in ("post_chiasmatic", "chiasmatic", "ptc"):
        for pat in ("24-2", "30-2", "60-4"):
            fa, fb = (os.path.join(dbdir, f"neurovf_{study}_{pat}_{k}.json") for k in ("total_deviation", "raw_threshold"))
            if not (os.path.exists(fa) and os.path.exists(fb)):
                continue                       # only pairs of databases that are both present can be compared
            a, b = json.load(open(fa)), json.load(open(fb))
            ok = True
            for sid in [k for k in a if not k.startswith("__")]:
                for eye in ("od", "os"):
                    for tp in ("preop", "postop"):
                        x, y = a[sid][f"vf_raw76_{tp}_{eye}"], b[sid][f"vf_raw76_{tp}_{eye}"]
                        if any((u is None) != (v is None) or (u is not None and v != u + 30) for u, v in zip(x, y)):
                            ok = False
            rep.add(cat, f"{study}_{pat}", "raw_threshold == total_deviation + 30 at every point (same draws)", "PASS" if ok else "FAIL")


def dicom_checks(rep, dicom_dir, dbdir):
    cat = "dicom-manifest"
    mp = os.path.join(dicom_dir, "MANIFEST_dicom.csv")
    if not os.path.exists(mp):
        return
    rows = list(csv.DictReader(open(mp)))
    rep.add(cat, "-", "manifest lists 1440 files (18 databases x 20 subjects x 2 eyes x 2 timepoints)", "PASS" if len(rows) == 1440 else "FAIL", detail=str(len(rows)))
    present = set(os.listdir(dbdir))
    skipped = sorted({r["database_file"] for r in rows if r["database_file"] not in present})
    if skipped:
        rep.add(cat, "-", f"manifest entries of {len(skipped)} database(s) not in the databases folder are not checked", "SKIP", detail=", ".join(s[:-5] for s in skipped[:3]) + (" ..." if len(skipped) > 3 else ""))
    rows = [r for r in rows if r["database_file"] in present]
    dbs, bad_arr, bad_file, bad_hash = {}, 0, 0, 0
    for r in rows:
        db = dbs.setdefault(r["database_file"], json.load(open(os.path.join(dbdir, r["database_file"]))))
        if json.loads(r["expected_array_json"]) != db[r["subject_id"]][r["json_key"]]:
            bad_arr += 1
        p = os.path.join(dicom_dir, r["dicom_path"])
        if not os.path.exists(p):
            bad_file += 1; continue
        if hashlib.sha256(open(p, "rb").read()).hexdigest() != r["sha256"]:
            bad_hash += 1
    rep.add(cat, "-", "every manifest expected_array equals the JSON array it points to (exact)", "PASS" if bad_arr == 0 else "FAIL", detail=f"{bad_arr} mismatches in {len(rows)} files")
    rep.add(cat, "-", "every DICOM file exists", "PASS" if bad_file == 0 else "FAIL", detail=f"{bad_file} missing")
    rep.add(cat, "-", "every DICOM SHA-256 equals the manifest", "PASS" if bad_hash == 0 else "FAIL", detail=f"{bad_hash} mismatches")
    per_db = collections.Counter(r["database_file"] for r in rows)
    rep.add(cat, "-", "80 DICOMs per database present (20 subjects x 2 eyes x 2 timepoints)", "PASS" if per_db and set(per_db.values()) == {80} else "FAIL")


# ---------------------------------------------------------------------------------------------------------------
def ctxkey(c, *fields):
    return tuple(c.get(f) for f in fields)


def compare_export(rep, name, S, edir):
    cat0 = "export"
    meta = json.load(open(os.path.join(edir, "meta.json")))
    recs = json.load(open(os.path.join(edir, "records.json")))
    records, errors = recs["records"], recs["errors"]
    cat = "export-meta"
    cmp_exact(rep, cat, name, "grid pattern / data type / clinical type", [meta["grid_pattern"], meta["data_type"], meta["clinical_type"]],
              [S.pattern, S.data_type, "post_chiasmatic" if S.study_type == "post_chiasmatic" else S.study_type])
    cmp_float(rep, cat, name, "grid point coordinates equal the reference table", np.array(meta["points_xy_deg"]).ravel(), S.points.ravel(), "registered_value_db")
    cmp_exact(rep, cat, name, "grid_flip_remap equals the oracle's mirror index", meta["grid_flip_remap"], [int(i) for i in S.mirror])
    for rg in O.REGIONS:
        cmp_exact(rep, cat, name, f"region mask '{rg}'", meta["region_masks"][rg], [bool(v) for v in O.region_mask(S.points, rg)])
    cmp_float(rep, cat, name, "severe/moderate thresholds", [meta["severe_threshold"], meta["moderate_threshold"]],
              [S.bands["severe"], S.bands["moderate"]], "registered_value_db")
    cmp_exact(rep, cat, name, "flip eye / surgery-side configuration", [meta["flip_eye"], meta["surgery_side_variable"], meta["surgery_side_reference"]],
              [S.flip_eye, S.surg_var, S.surg_ref])
    cmp_exact(rep, cat, name, "subject list", meta["subjects_in_database"], S.subjects)
    rep.add(cat, name, "no exceptions raised inside NeuroVF during export", "PASS" if not errors else "FAIL", detail=f"{len(errors)} errors: " + str(errors[0]["traceback"][-200:]) if errors else "")

    views = meta["views_exported"]
    modes = (("cross", "preop"), ("cross", "postop"), ("longitudinal", None))
    pair_path = bool(meta.get("pair_path_exported", False))
    # (ctx view, eye key as recorded inside the payload, oracle view)
    combos = [(v, "avg" if v == "registered" else v, "registered" if v == "registered" else v) for v in views]
    if pair_path:
        combos += [("pair", e_, e_) for e_ in ("od", "os")]
    rep.add("export-meta", name, "OD|OS pair-path records exported (chiasmatic / PTC)", "PASS" if pair_path == (S.study_type in ("chiasmatic", "ptc")) else "FAIL",
            detail=f"pair_path_exported={pair_path}")

    # ---- completeness: exactly one record per analysis key (a duplicate could hide a bad earlier copy) -------------------
    cat = "export-coverage"
    for kind, keyf in (("cohort_mean", lambda r: (r["ctx"]["view"], r["data"]["view"], r["data"]["mode"], r["data"]["timepoint"])),
                       ("group_stats", lambda r: (r["ctx"]["view"], r["ctx"]["mode"], r["ctx"]["timepoint"], r["data"]["variable"], r["data"]["eye_view"])),
                       ("group_omnibus", lambda r: (r["ctx"]["view"], r["ctx"]["mode"], r["ctx"]["timepoint"], r["data"]["variable"], r["data"]["eye_view"])),
                       ("correlation", lambda r: (r["ctx"]["view"], r["ctx"]["mode"], r["ctx"]["timepoint"], r["data"]["variable"], r["data"]["method"])),
                       ("correlation_pair", lambda r: (r["ctx"]["view"], r["ctx"]["mode"], r["ctx"]["timepoint"], r["data"]["variable"], r["data"]["method"], r["data"]["eye_view"])),
                       ("correlation_by_category", lambda r: (r["ctx"]["view"], r["ctx"]["mode"], r["ctx"]["timepoint"], r["data"]["variable"], r["data"]["method"], r["data"]["split_by"], r["data"]["category"], r["data"]["eye_view"]))):
        cnt = collections.Counter(keyf(r) for r in records if r["kind"] == kind)
        dup = [k for k, v in cnt.items() if v > 1]
        rep.add(cat, name, f"no duplicate '{kind}' records", "PASS" if not dup else "FAIL", detail=str(dup[:3]))
    n_two = n_multi = 0
    for view_, ek_, ov_ in combos:
        for mode_, tp_ in modes:
            for var_ in S.categorical:
                lv_, _x, _y = O.levels_usable(S, mode_, tp_, "registered" if view_ == "pair" else ov_, var_)
                n_two += (len(lv_) == 2); n_multi += (len(lv_) >= 3)
    n_pair_eyes = 2 if pair_path else 0
    n_cat_levels = sum(len({str(S.db[s].get(v, "")).strip() for s in S.subjects}) for v in S.categorical)   # levels over all categorical variables
    n_split_per_eye_view = len(modes) * len(S.continuous) * 2 * n_cat_levels
    expected_counts = {"cohort_mean": (len(views) + n_pair_eyes) * len(modes),
                       "group_stats": n_two, "group_omnibus": n_multi,
                       "correlation": len(views) * len(modes) * len(S.continuous) * 2,
                       "correlation_pair": n_pair_eyes * len(modes) * len(S.continuous) * 2,
                       "correlation_by_category": (len(views) + n_pair_eyes) * n_split_per_eye_view}
    for kind, n_exp in expected_counts.items():
        n_got = sum(1 for r in records if r["kind"] == kind)
        rep.add(cat, name, f"'{kind}' record count equals the number of analyses expected ({n_exp})", "PASS" if n_got == n_exp else "FAIL", detail=f"found {n_got}")

    # ---- registered point values -------------------------------------------------------------------------------------
    cat = "registered-values"
    pts, pts_all = {}, collections.defaultdict(list)
    for r in records:
        if r["kind"] == "points":
            d = r["data"]; pts[(d["view"], d["mode"], d["timepoint"])] = d
            pts_all[(d["view"], d["mode"], d["timepoint"])].append(d)
    for view in views:
        for mode, tp in modes:
            copies = pts_all.get(("avg" if view == "registered" else view, mode, tp), [])
            label = f"{view} | {mode} {tp or 'delta'}"
            if not copies:
                rep.add(cat, name, f"{label}: points record present", "FAIL", detail="missing"); continue
            exp = {s: S.vector(s, mode, tp, "registered" if view == "registered" else view) for s in S.subjects}
            exp = {s: v for s, v in exp.items() if v is not None}
            failed, subj_bad = [], 0
            for ci, d in enumerate(copies):          # NeuroVF records these on every analysis call: check EVERY copy
                got = {row["sid"]: row["values"] for row in d["rows"]}
                if sorted(got) != sorted(exp):
                    subj_bad += 1; continue
                for s in exp:
                    before = len(rep.rows)
                    if not cmp_float(rep, "_tmp", name, "", got[s], exp[s], "registered_value_db"):
                        failed.append((ci, s))
                    rep.rows = rep.rows[:before]
            rep.add(cat, name, f"{label}: included subjects identical in all {len(copies)} recorded copies", "PASS" if subj_bad == 0 else "FAIL", detail=f"{subj_bad} copies differ")
            rep.add(cat, name, f"{label}: every subject's registered vector equals the oracle's, in all {len(copies)} recorded copies ({len(exp)} subjects each)",
                    "PASS" if not failed else "FAIL", detail=f"failing (copy, subject): {failed[:5]}")

    # ---- means ---------------------------------------------------------------------------------------------------------------
    cat = "means"
    cm = {}
    for r in records:
        if r["kind"] == "cohort_mean":
            cm[(r["ctx"]["view"], r["data"]["view"], r["data"]["mode"], r["data"]["timepoint"])] = r["data"]
    for view, ek, ov in combos:
        for mode, tp in modes:
            d = cm.get((view, ek, mode, tp)); label = f"{view}{'/' + ek if view == 'pair' else ''} | {mode} {tp or 'delta'}"
            e = O.cohort_mean(S, mode, tp, ov)
            if d is None or e is None:
                rep.add(cat, name, f"cohort mean {label}", "FAIL", detail="record missing"); continue
            cmp_exact(rep, cat, name, f"cohort mean {label}: n subjects", d["n"], e["n"])
            cmp_exact(rep, cat, name, f"cohort mean {label}: valid counts per location", d["valid_n"], [int(v) for v in e["valid_n"]])
            cmp_exact(rep, cat, name, f"cohort mean {label}: unavailable (complete-case) mask", d["unavailable"], [bool(v) for v in e["unavailable"]])
            cmp_float(rep, cat, name, f"cohort mean {label}: mean values", d["mean"], e["mean"], "mean_db")
    catm = collections.defaultdict(list)
    for r in records:
        if r["kind"] == "category_mean":
            catm[(r["ctx"]["view"], r["data"]["mode"], r["data"]["timepoint"], r["data"]["variable"])].append(r["data"])
    for view in views:
        for mode, tp in modes:
            for var in S.categorical:
                label = f"category means {var} | {view} | {mode} {tp or 'delta'}"
                ds = catm.get((view, mode, tp, var), [])
                sidmap = {}
                for s in S.subjects:
                    if S.vector(s, mode, tp, "registered" if view == "registered" else view) is not None:
                        v = str(S.db[s].get(var, "")).strip()
                        if v: sidmap.setdefault(v, []).append(s)
                if not ds:
                    rep.add(cat, name, label, "FAIL", detail="record missing"); continue
                cmp_exact(rep, cat, name, label + ": categories", sorted(d["category"] for d in ds), sorted(sidmap))
                for d in ds:
                    e = O.cohort_mean(S, mode, tp, "registered" if view == "registered" else view, sids=sidmap[d["category"]])
                    ok1 = cmp_exact(rep, cat, name, f"{label} [{d['category']}]: n, valid counts, complete-case mask",
                                    [d["n"], d["valid_n"], d["unavailable"]], [e["n"], [int(v) for v in e["valid_n"]], [bool(v) for v in e["unavailable"]]])
                    cmp_float(rep, cat, name, f"{label} [{d['category']}]: means", d["mean"], e["mean"], "mean_db")

    # ---- group comparison: two levels (Welch t / Hedges g) or three or more (Welch ANOVA / Games-Howell) ----------------------------
    cat = "group-comparison"
    gs = {}
    for r in records:
        if r["kind"] in ("group_stats", "group_omnibus"):
            gs[(r["ctx"]["view"], r["ctx"]["mode"], r["ctx"]["timepoint"], r["data"]["variable"], r["data"]["eye_view"])] = (r["kind"], r["data"])
    for view, ek, ov in combos:
        for mode, tp in modes:
            for var in S.categorical:
                label = f"{var} | {view}{'/' + ek if view == 'pair' else ''} | {mode} {tp or 'delta'}"
                levels, excl, _n_all = O.levels_usable(S, mode, tp, "registered" if view == "pair" else ov, var)
                e = O.group_comparison(S, mode, tp, ov, var, levels=levels) if len(levels) >= 2 else None
                got = gs.get((view, mode, tp, var, ek))
                if e is None:
                    rep.add(cat, name, label + ": not computable (fewer than 2 analysable groups)", "PASS" if got is None else "FAIL",
                            detail="" if got is None else "record present although the oracle expects none"); continue
                if got is None:
                    rep.add(cat, name, label, "FAIL", detail="record missing"); continue
                kind, d = got
                if (kind == "group_stats") != (e["kind"] == "two"):
                    rep.add(cat, name, label + ": workflow", "FAIL", detail=f"app used {kind}, oracle expects {e['kind']} ({len(levels)} analysable levels)"); continue
                cmp_exact(rep, cat, name, label + ": number of groups / unique contrasts", [d["n_groups"], d["n_contrasts"]], [e["n_groups"], e["n_contrasts"]])
                cmp_exact(rep, cat, name, label + ": non-computable masks (incomplete group, zero-variance group)",
                          [d["reason_incomplete"], d["reason_zero_variance"]], [[bool(v) for v in e["reason_incomplete"]], [bool(v) for v in e["reason_zero_variance"]]])
                cmp_exact(rep, cat, name, label + ": family_n / n_unc / n_sig", [d["family_n"], d["n_unc"], d["n_sig"]], [e["family_n"], e["n_unc"], e["n_sig"]])
                if e["kind"] == "two":
                    cmp_exact(rep, cat, name, label + ": group labels, n_A, n_B", [d["group_a"], d["group_b"], d["n_a"], d["n_b"]], [levels[0], levels[1], e["n"][0], e["n"][1]])
                    cmp_exact(rep, cat, name, label + ": valid counts A/B", [d["n_valid_a"], d["n_valid_b"]], [[int(v) for v in e["n_valid"][0]], [int(v) for v in e["n_valid"][1]]])
                    cmp_exact(rep, cat, name, label + ": difference-valid mask", d["difference_valid_mask"], [bool(v) for v in e["difference_valid_mask"]])
                    cmp_exact(rep, cat, name, label + ": correction text names BH-FDR", "BH-FDR" in d["correction"], True)
                    for k, ek_, tk in (("mean_a", None, "mean_db"), ("mean_b", None, "mean_db"), ("diff", "diff", "mean_db"), ("t", "t", "welch_t"), ("df", "df", "welch_df"),
                                       ("p", "p", "p_value"), ("q", "q", "q_value"), ("hedges_g", "hedges_g", "hedges_g")):
                        exp = e["mean"][0] if k == "mean_a" else (e["mean"][1] if k == "mean_b" else e[ek_])
                        cmp_float(rep, cat, name, f"{label}: {k}", d[k], exp, tk)
                else:
                    kk = e["n_groups"]; m_ = e["n_contrasts"]
                    cmp_exact(rep, cat, name, label + ": levels analysed, n per level, excluded levels (n<2)", [d["levels"], d["n"], d["excluded_levels"]], [levels, e["n"], excl])
                    cmp_exact(rep, cat, name, label + ": unique contrasts in order (i<j)", d["contrasts"], [[levels[i], levels[j]] for i in range(kk) for j in range(i + 1, kk)])
                    corr_txt = d["correction"]
                    cmp_exact(rep, cat, name, label + ": correction text states omnibus BH-FDR, Games-Howell studentized range and the contrast count",
                              ("Welch ANOVA" in corr_txt["omnibus"] and "BH-FDR" in corr_txt["omnibus"] and "Games" in corr_txt["posthoc"]
                               and "studentized range" in corr_txt["posthoc"] and f"{m_} unique contrasts" in corr_txt["posthoc"]), True)
                    cmp_exact(rep, cat, name, label + ": valid counts per level", d["n_valid"], [[int(v) for v in row] for row in e["n_valid"]])
                    cmp_float(rep, cat, name, label + ": group means", np.array([[np.nan if v is None else v for v in row] for row in d["mean"]]).ravel(), e["mean"].ravel(), "mean_db")
                    cmp_exact(rep, cat, name, label + ": df1 / post-hoc applicable mask / locations", [d["df1"], d["posthoc_applicable"], d["n_posthoc_locations"]],
                              [e["df1"], [bool(v) for v in e["gate"]], e["n_posthoc_locations"]])
                    for k, tk in (("F", "welch_F"), ("df2", "welch_df"), ("p", "p_value"), ("q", "q_value"), ("eta_sq", "omnibus_effect_size"), ("omega_sq", "omnibus_effect_size")):
                        cmp_float(rep, cat, name, f"{label}: omnibus {k}", d[k], e[k], tk)
                    for i in range(kk):
                        for j in range(i + 1, kk):
                            key = f"{levels[i]} vs {levels[j]}"; ph = d["posthoc"].get(key); eph = e["posthoc"][(i, j)]
                            if ph is None:
                                rep.add(cat, name, f"{label}: contrast '{key}'", "FAIL", detail="missing"); continue
                            for k, tk in (("diff", "mean_db"), ("t", "welch_t"), ("df", "welch_df"), ("p", "games_howell_p"), ("hedges_g", "hedges_g")):
                                cmp_float(rep, cat, name, f"{label}: contrast '{key}' {k}", ph[k], eph[k], tk)
                            cmp_exact(rep, cat, name, f"{label}: contrast '{key}' significant mask / count", [ph["significant"], ph["n_significant"]],
                                      [[bool(v) for v in eph["significant"]], int(eph["significant"].sum())])

    # ---- correlation ---------------------------------------------------------------------------------------------------------------------
    cat = "correlation"
    cr = {}
    for r in records:
        if r["kind"] in ("correlation", "correlation_pair"):
            ek_ = r["data"].get("eye_view") or ("avg" if r["ctx"]["view"] == "registered" else r["ctx"]["view"])
            cr[(r["ctx"]["view"], r["ctx"]["mode"], r["ctx"]["timepoint"], r["data"]["variable"], r["data"]["method"], ek_)] = r["data"]
    for view, ek, ov in combos:
        for mode, tp in modes:
            for var in S.continuous:
                for method in ("pearson", "spearman"):
                    label = f"{method} {var} | {view}{'/' + ek if view == 'pair' else ''} | {mode} {tp or 'delta'}"
                    d = cr.get((view, mode, tp, var, method, ek))
                    e = O.correlation(S, mode, tp, ov, var, method)
                    if d is None or e is None:
                        rep.add(cat, name, label, "FAIL" if (d is None) != (e is None) else "PASS",
                                detail="record/oracle disagree on whether the analysis is computable"); continue
                    cmp_exact(rep, cat, name, label + ": included subjects", d["sids"], e["sids"])
                    cmp_float(rep, cat, name, label + ": covariate values", d["x"], e["x"], "registered_value_db")
                    cmp_exact(rep, cat, name, label + ": valid counts, computable mask", [d["n_valid"], d["computable"]],
                              [[int(v) for v in e["n_valid"]], [bool(v) for v in e["computable"]]])
                    for k, tk in (("coef", "correlation_coefficient"), ("p", "p_value"), ("q", "q_value")):
                        cmp_float(rep, cat, name, f"{label}: {k}", d[k], e[k], tk)
                    cmp_exact(rep, cat, name, label + ": family_n / n_unc / n_sig", [d["family_n"], d["n_unc"], d["n_sig"]], [e["family_n"], e["n_unc"], e["n_sig"]])

    # ---- correlation split by a categorical variable (stratified; BH-FDR within each category) ----------------------------------
    cat = "correlation-by-category"
    cbc = {}
    for r in records:
        if r["kind"] == "correlation_by_category":
            d = r["data"]
            cbc[(r["ctx"]["view"], r["ctx"]["mode"], r["ctx"]["timepoint"], d["variable"], d["method"], d["split_by"], d["category"], d["eye_view"])] = d
    for view, ek, ov in combos:
        for mode, tp in modes:
            for var in S.continuous:
                for method in ("pearson", "spearman"):
                    for split in S.categorical:
                        exp_c = O.correlation_by_category(S, mode, tp, ov, var, method, split)
                        lab0 = f"{method} {var} split by {split} | {view}{'/' + ek if view == 'pair' else ''} | {mode} {tp or 'delta'}"
                        found = sorted(k[6] for k in cbc if k[:5] == (view, mode, tp, var, method) and k[5] == split and k[7] == ek)
                        cmp_exact(rep, cat, name, lab0 + ": categories present", found, sorted(exp_c))
                        for cval, ec in exp_c.items():
                            d = cbc.get((view, mode, tp, var, method, split, cval, ek)); label = f"{lab0} [{cval}]"
                            if d is None:
                                rep.add(cat, name, label, "FAIL", detail="record missing"); continue
                            cmp_exact(rep, cat, name, label + ": subjects and n", [d["sids"], d["n"]], [ec["sids"], ec["n"]])
                            er = ec["res"]
                            if er is None:
                                cmp_exact(rep, cat, name, label + ": < 3 subjects -> no coefficients", d["coef"], None); continue
                            cmp_float(rep, cat, name, label + ": covariate values", d["x"], er["x"], "registered_value_db")
                            cmp_exact(rep, cat, name, label + ": valid counts, computable mask", [d["n_valid"], d["computable"]],
                                      [[int(v) for v in er["n_valid"]], [bool(v) for v in er["computable"]]])
                            for k, tk in (("coef", "correlation_coefficient"), ("p", "p_value"), ("q", "q_value")):
                                cmp_float(rep, cat, name, f"{label}: {k}", d[k], er[k], tk)
                            cmp_exact(rep, cat, name, label + ": statistics displayable (>= 10 computable locations)", d["stats_available"], ec["stats_available"])
                            if ec["stats_available"]:
                                cmp_exact(rep, cat, name, label + ": n_unc / n_sig within category", [d["n_unc"], d["n_sig"]], [er["n_unc"], er["n_sig"]])

    # ---- cluster / cumulative ---------------------------------------------------------------------------------------------------------------
    cat = "cluster-cumulative"
    for kind in ("cluster", "cumulative"):
        subj, fits = {}, collections.defaultdict(list)
        for r in records:
            c = r["ctx"]
            if r["kind"] == f"{kind}_subjects":
                subj[(c["view"], c["timepoint"], c["variable"])] = r["data"]
            if r["kind"] == f"{kind}_fit":
                fits[(c["view"], c["timepoint"], c["variable"])].append(r["data"])
        for view in views:
            for tp in ("preop", "postop"):
                for var in S.continuous:
                    label = f"{kind} {var} | {view} | {tp}"
                    d = subj.get((view, tp, var)); f = fits.get((view, tp, var), [])
                    e = O.cluster_cumulative(S, kind, tp, "registered" if view == "registered" else view, var)
                    if d is None or e is None:
                        rep.add(cat, name, label, "FAIL" if (d is None) != (e is None) else "PASS", detail="record/oracle disagree"); continue
                    cmp_exact(rep, cat, name, label + ": included subjects", d["sids"], e["sids"])
                    cmp_float(rep, cat, name, label + ": covariate values", d["x"], e["x"], "registered_value_db")
                    for (blabel, vals), (ename, evals) in zip(d["pct"].items(), e["pct"].items()):
                        cmp_float(rep, cat, name, f"{label}: % of locations per subject [{ename}]", vals, evals, "percentage")
                    rep.add(cat, name, f"{label}: number of regression fits", "PASS" if len(f) == len(e["fits"]) else "FAIL", detail=f"{len(f)} vs {len(e['fits'])}")
                    for fit, (ename, ef) in zip(f, e["fits"].items()):
                        cmp_exact(rep, cat, name, f"{label} [{ename}]: n", fit["n"], ef["n"])
                        for k in ("slope", "intercept", "r"):
                            cmp_float(rep, cat, name, f"{label} [{ename}]: {k}", [fit[k]], [ef[k]], "regression")
                        cmp_float(rep, cat, name, f"{label} [{ename}]: p", [fit["p"]], [ef["p"]], "p_value")

    # ---- MD (PTC) ---------------------------------------------------------------------------------------------------------------------------------
    if S.study_type == "ptc":
        cat = "md"
        got = {(r["ctx"]["view"], r["data"]["subject"], r["data"]["timepoint"]): r["data"] for r in records if r["kind"] == "md_subject"}
        for view in views:
            bad = []
            for s in S.subjects:
                for tp in ("preop", "postop"):
                    e = O.md_expected(S, s, tp, "registered" if view == "registered" else view)
                    d = got.get((view, s, tp))
                    if (d is None) != (e is None):
                        bad.append((s, tp, "presence")); continue
                    if e is None: continue
                    if (d["n_points"], d["tier"], d["source"]) != (e[1], e[2], e[3]) or abs(d["md"] - e[0]) > TOL["md_db"]["abs"]:
                        bad.append((s, tp, d["md"], e[0]))
            rep.add(cat, name, f"per-subject MD value / n / tier / source | {view}", "PASS" if not bad else "FAIL", detail=str(bad[:4]))
        mv = {}
        fitrec = collections.defaultdict(list)
        for r in records:
            c = r["ctx"]
            if r["kind"] == "md_vs_variable":
                mv[(c["view"], c["mode"], c["timepoint"], r["data"]["variable"])] = r["data"]
            if r["kind"] in ("md_fit", "md_welch"):
                fitrec[(c["view"], c["mode"], c["timepoint"], c["variable"])].append(r)
        for view in views:
            for mode, tp in modes:
                for var in S.continuous + S.categorical:
                    label = f"MD vs {var} | {view} | {mode} {tp or 'delta'}"
                    d = mv.get((view, mode, tp, var))
                    e = O.md_vs_variable(S, mode, tp, "registered" if view == "registered" else view, var)
                    if d is None or e is None:
                        rep.add(cat, name, label, "FAIL" if (d is None) != (e is None) else "PASS", detail="record/oracle disagree"); continue
                    cmp_exact(rep, cat, name, label + ": included subjects", d["sids"], e["sids"])
                    cmp_float(rep, cat, name, label + ": MD values", d["md"], e["md"], "md_db")
                    fr = fitrec.get((view, mode, tp, var), [])
                    if e["is_categorical"]:
                        if "welch" in e:
                            w = [x for x in fr if x["kind"] == "md_welch"]
                            if not w: rep.add(cat, name, label + ": Welch record", "FAIL", detail="missing")
                            else:
                                w = w[0]["data"]
                                cmp_float(rep, cat, name, label + ": Welch t and p", [w["t"], w["p"], w["mean_a"], w["mean_b"]],
                                          [e["welch"]["t"], e["welch"]["p"], e["welch"]["mean_a"], e["welch"]["mean_b"]], "welch_t")
                    else:
                        w = [x for x in fr if x["kind"] == "md_fit"]
                        if not w: rep.add(cat, name, label + ": regression record", "FAIL", detail="missing")
                        else:
                            w = w[0]["data"]
                            cmp_float(rep, cat, name, label + ": slope, intercept, r", [w["slope"], w["intercept"], w["r"]],
                                      [e["fit"]["slope"], e["fit"]["intercept"], e["fit"]["r"]], "regression")
                            cmp_float(rep, cat, name, label + ": p", [w["p"]], [e["fit"]["p"]], "p_value")

    # ---- scotoma area ----------------------------------------------------------------------------------------------------------------------------------------
    cat = "scotoma-area"
    sc = {(r["data"]["subject"], r["data"]["timepoint"], r["data"]["region"]): r["data"]["views"] for r in records if r["kind"] == "scotoma"}
    exp_views = O.scotoma_views(S)
    bad, bad_names = [], []
    for s in S.subjects:
        for tp in ("preop", "postop"):
            for rg in O.REGIONS:
                d = sc.get((s, tp, rg))
                if d is None: bad.append((s, tp, rg, "record missing")); continue
                if tuple(d.keys()) != exp_views: bad.append((s, tp, rg, "views", list(d.keys()))); continue
                for view in exp_views:
                    e = O.scotoma(S, s, tp, rg, view); g = d[view]
                    if (g is None) != (e is None): bad.append((s, tp, rg, view, "None mismatch")); continue
                    if e is None: continue
                    vals = list(g.values())
                    if len(vals) != 6 or any(abs(a - b) > TOL["percentage"]["abs"] for a, b in zip(vals, e)):
                        bad.append((s, tp, rg, view, vals, e))
                    last = list(g.keys())[-1]
                    if last != ("Valid_Paired_ROI_Locations" if view == "registered" else "Valid_ROI_Locations"): bad_names.append((view, last))
    rep.add(cat, name, f"{len(S.subjects)} subjects x 2 timepoints x {len(O.REGIONS)} regions x views {exp_views}: all six reported values", "PASS" if not bad else "FAIL", detail=str(bad[:3]))
    rep.add(cat, name, "valid-count label says 'paired' only for the registered view", "PASS" if not bad_names else "FAIL", detail=str(bad_names[:3]))

    # ---- CSV tables are consistent with the JSON records -----------------------------------------------------------------------------------------------------------
    cat = "csv-consistency"
    T = os.path.join(edir, "tables")
    rows = list(csv.DictReader(open(os.path.join(T, "registered_points.csv"))))
    tab = collections.defaultdict(dict)
    for r in rows:
        tab[(r["view"], r["mode"], r["timepoint"], r["subject"])][int(r["point_index"])] = float(r["value_db"]) if r["value_db"] != "" else np.nan
    bad = 0
    for (view, mode, tp), d in pts.items():
        for row in d["rows"]:
            v = tab.get((view, mode, tp or "", row["sid"]))
            if v is None or not np.array_equal(np.array([v[i] for i in range(len(row["values"]))]), nan(row["values"]), equal_nan=True):
                bad += 1
    rep.add(cat, name, "registered_points.csv equals the registered-point records", "PASS" if bad == 0 else "FAIL", detail=f"{bad} mismatching vectors")
    rows = list(csv.DictReader(open(os.path.join(T, "group_comparison.csv"))))
    gtab = collections.defaultdict(lambda: collections.defaultdict(dict))
    for r in rows:
        k = (r["view"], r["mode"], r["timepoint"], r["variable"], r["eye_view"])
        for col in ("p", "q_bh", "welch_t", "hedges_g"):
            gtab[k][col][int(r["point_index"])] = float(r[col]) if r[col] != "" else np.nan
    bad = 0
    for (view, mode, tp, var, eyev), (kind_, d) in gs.items():
        if kind_ != "group_stats":
            continue
        k = (view, mode, tp or "", var, eyev)
        for col, key in (("p", "p"), ("q_bh", "q"), ("welch_t", "t"), ("hedges_g", "hedges_g")):
            a = np.array([gtab[k][col][i] for i in range(len(d[key]))])
            if not np.allclose(a, nan(d[key]), rtol=1e-12, atol=1e-15, equal_nan=True): bad += 1
    rep.add(cat, name, "group_comparison.csv equals the two-level group-statistics records (p, q, t, Hedges' g)", "PASS" if bad == 0 else "FAIL", detail=f"{bad} mismatching columns")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--databases", required=True)
    ap.add_argument("--exports", default=None, help="folder containing one sub-folder per database (output of NeuroVF_research.py --out)")
    ap.add_argument("--dicom", default=None)
    ap.add_argument("--out", default="oracle_report")
    ap.add_argument("--only", default=None, help="substring filter on database file name")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rep = Report()
    ref = json.load(open(POINTS_FILE))
    index = json.load(open(os.path.join(a.databases, "_database_index.json")))
    files = sorted(f for f in os.listdir(a.databases) if f.endswith(".json") and not f.startswith("_"))
    if a.only: files = [f for f in files if a.only in f]
    for f in files:
        name = f[:-5]
        db = json.load(open(os.path.join(a.databases, f)))
        S = O.Study(db, ref[PATTERN_KEY[db["__vf_grid_pattern__"]]])
        integrity_checks(rep, name, S, index)
        edir = os.path.join(a.exports, name) if a.exports else None
        if edir and os.path.isdir(edir):
            compare_export(rep, name, S, edir)
        else:
            rep.add("export", name, "NeuroVF export present", "SKIP", detail=f"no folder {edir}")
    if not a.only:
        pair_checks(rep, a.databases, files)
    if a.dicom:
        dicom_checks(rep, a.dicom, a.databases)
    p, f_, s = rep.counts()
    by = collections.defaultdict(collections.Counter)
    for r in rep.rows: by[r["category"]][r["status"]] += 1
    lines = [f"NeuroVF oracle report -- {p} PASS, {f_} FAIL, {s} SKIP ({len(rep.rows)} checks)", "=" * 78, "",
             f"{'category':<22}{'PASS':>8}{'FAIL':>8}{'SKIP':>8}"]
    for k, c in sorted(by.items()):
        lines.append(f"{k:<22}{c['PASS']:>8}{c['FAIL']:>8}{c['SKIP']:>8}")
    lines += ["", "TOLERANCES (tolerances.json): " + json.dumps(TOL), ""]
    fails = [r for r in rep.rows if r["status"] == "FAIL"]
    lines.append(f"FAILURES ({len(fails)}):")
    for r in fails[:400]:
        lines.append(f"  [{r['category']}] {r['database']} :: {r['check']}  (max|err|={r['max_abs_error']})  {r['detail']}")
    open(os.path.join(a.out, "oracle_report.txt"), "w").write("\n".join(lines))
    json.dump({"pass": p, "fail": f_, "skip": s, "checks": rep.rows}, open(os.path.join(a.out, "oracle_report.json"), "w"), indent=1)
    with open(os.path.join(a.out, "oracle_checks.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rep.rows[0].keys())); w.writeheader(); w.writerows(rep.rows)
    print("\n".join(lines[:14 + len(by)]))
    print(f"\nfull report: {os.path.join(a.out, 'oracle_report.txt')}")
    return 1 if f_ else 0


if __name__ == "__main__":
    sys.exit(main())
