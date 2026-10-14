#!/usr/bin/env python3
"""mutation_selfcheck.py -- proves the oracle can FAIL. Copies one NeuroVF export, injects a known error,
runs the oracle, and requires at least one FAIL for every injected error.
Usage: python mutation_selfcheck.py --databases ../databases/json --export <export_root>/<one_database_folder>"""
import argparse, copy, json, os, shutil, subprocess, sys, tempfile
ap = argparse.ArgumentParser(); ap.add_argument("--databases", required=True); ap.add_argument("--export", required=True); ap.add_argument("--only", default=None, help="run only mutations whose label contains this text")
ap.add_argument("--shard", default=None, help="i/n: run every n-th mutation starting at i (for parallel runs)")
a = ap.parse_args()
HERE = os.path.dirname(os.path.abspath(__file__))
name = os.path.basename(os.path.normpath(a.export))
base = json.load(open(os.path.join(a.export, "records.json")))

def first(kind, **ctx):
    for r in base["records"]:
        if r["kind"] == kind and all(r["ctx"].get(k) == v for k, v in ctx.items()): return r
MUT = {}
def m_points(R):
    r = [x for x in R["records"] if x["kind"] == "points" and x["data"]["view"] == "avg"][0]; r["data"]["rows"][0]["values"][5] += 0.5
def m_mean(R):  first_(R, "cohort_mean")["data"]["mean"][3] += 1e-6
def m_unavail(R): d = first_(R, "cohort_mean")["data"]["unavailable"]; d[7] = not d[7]
def _gs(R, kind="group_stats"): return next(x for x in R["records"] if x["kind"] == kind)["data"]
def _first_idx(arr): return next(k for k, v in enumerate(arr) if v is not None)
def m_p(R):     d = _gs(R)["p"]; i = _first_idx(d); d[i] *= 1.001
def m_q(R):
    d = _gs(R)["q"]; i = _first_idx(d); d[i] = d[i] - 0.01 if d[i] >= 0.99 else d[i] * 1.01          # q is capped at 1.0: make sure the value really changes
def m_t(R):     d = _gs(R)["t"]; i = _first_idx(d); d[i] = -d[i]
def m_hg(R):    d = _gs(R)["hedges_g"]; i = _first_idx(d); d[i] += 1e-4
def m_hg_cohen(R):          # Hedges g replaced by a Cohen-style d (no small-sample correction)
    d = _gs(R)["hedges_g"]
    for i, v in enumerate(d):
        if v is not None: d[i] = v / 0.95
def m_nc(R):                # a non-computable location silently reported as computable (fabricated value)
    d = _gs(R); i = next(k for k, v in enumerate(d["p"]) if v is None); d["p"][i] = 0.5
def m_ngroups(R): _gs(R, "group_omnibus")["n_groups"] += 1
def m_ncontrast(R): _gs(R, "group_omnibus")["n_contrasts"] -= 1
def m_F(R):     d = _gs(R, "group_omnibus")["F"]; i = _first_idx(d); d[i] *= 1.0001
def m_omni_p(R): d = _gs(R, "group_omnibus")["p"]; i = _first_idx(d); d[i] *= 1.001
def m_omni_q(R):
    d = _gs(R, "group_omnibus")["q"]; i = _first_idx(d); d[i] = d[i] - 0.01 if d[i] >= 0.99 else d[i] * 1.01
def m_omega(R): d = _gs(R, "group_omnibus")["omega_sq"]; i = _first_idx(d); d[i] += 1e-5
def m_gh_p(R):
    ph = _gs(R, "group_omnibus")["posthoc"]; d = next(v for v in ph.values() if any(x is not None for x in v["p"])); i = _first_idx(d["p"]); d["p"][i] *= 1.01
def m_gh_g(R):
    ph = _gs(R, "group_omnibus")["posthoc"]; d = next(v for v in ph.values() if any(x is not None for x in v["hedges_g"])); i = _first_idx(d["hedges_g"]); d["hedges_g"][i] += 1e-4
def m_gate(R):              # post hoc applied outside the omnibus gate
    d = _gs(R, "group_omnibus"); i = next(k for k, v in enumerate(d["posthoc_applicable"]) if not v); d["posthoc_applicable"][i] = True
def m_ntests(R):            # Games-Howell family-wise correction dropped (p replaced by the uncorrected two-sided Welch p)
    ph = _gs(R, "group_omnibus")["posthoc"]
    from scipy import stats as st
    for d in ph.values():
        for i, (t, df) in enumerate(zip(d["t"], d["df"])):
            if t is not None: d["p"][i] = float(2 * st.t.sf(abs(t), df))
def m_text(R):  _gs(R, "group_omnibus")["correction"]["posthoc"] = "none"
def m_corr(R):  d = first_(R, "correlation")["data"]["coef"]; i = next(k for k, v in enumerate(d) if v is not None); d[i] += 1e-5
def m_nsig(R):  first_(R, "group_stats")["data"]["n_sig"] += 1
def m_cluster(R): first_(R, "cluster_subjects")["data"]["pct"][next(iter(first_(R, "cluster_subjects")["data"]["pct"]))][0] += 0.01
def m_fit(R):   first_(R, "cluster_fit")["data"]["slope"] += 1e-6
def m_scot(R):
    r = next(x for x in R["records"] if x["kind"] == "scotoma" and x["data"]["views"]["registered"]); d = r["data"]["views"]["registered"]
    k = next(iter(d)); d[k] += 0.1
def m_scot_od(R):
    r = next(x for x in R["records"] if x["kind"] == "scotoma" and (x["data"]["views"].get("od") or None)); d = r["data"]["views"]["od"]
    k = next(iter(d)); d[k] += 0.1
def m_scot_noeye(R):          # only the registered table, no OD / OS rows
    for x in R["records"]:
        if x["kind"] == "scotoma": x["data"]["views"] = {"registered": x["data"]["views"]["registered"]}
def m_scot_mirrored(R):       # OS row replaced by the registered row (a mirrored result presented as 'as recorded')
    for x in R["records"]:
        if x["kind"] == "scotoma" and x["data"]["views"].get("os") is not None: x["data"]["views"]["os"] = dict(x["data"]["views"]["registered"] or x["data"]["views"]["os"])
def m_md(R):
    r = next((x for x in R["records"] if x["kind"] == "md_subject" and x["data"]["md"] is not None), None)
    if r: r["data"]["md"] += 0.01
    else: raise SystemExit("no MD records (non-PTC database): skipping this mutation")
def m_cbc_coef(R):  d = first_(R, "correlation_by_category")["data"]["coef"]; i = next(k for k, v in enumerate(d) if v is not None); d[i] += 1e-5
def m_cbc_q(R):
    d = first_(R, "correlation_by_category")["data"]["q"]; i = next(k for k, v in enumerate(d) if v is not None)
    d[i] = d[i] - 0.01 if d[i] >= 0.99 else d[i] * 1.01
def m_pair_corr(R): d = next(x for x in R["records"] if x["kind"] == "correlation_pair")["data"]["p"]; i = next(k for k, v in enumerate(d) if v is not None); d[i] *= 1.001
def m_pair_group(R): d = next(x for x in R["records"] if x["kind"] == "group_stats" and x["ctx"]["view"] == "pair")["data"]["t"]; i = next(k for k, v in enumerate(d) if v is not None); d[i] = -d[i]
def m_drop_pair_split(R): R["records"] = [x for x in R["records"] if x["ctx"].get("analysis") != "correlation_by_category_pair"]
def m_pair_mean(R): d = next(x for x in R["records"] if x["kind"] == "cohort_mean" and x["ctx"]["view"] == "pair")["data"]["mean"]; i = next(k for k, v in enumerate(d) if v is not None); d[i] += 1e-6
def m_swap_eye(R):
    r = next(x for x in R["records"] if x["kind"] == "correlation_pair" and x["data"]["eye_view"] == "od"); r["data"]["eye_view"] = "os"
def first_(R, kind): return next(x for x in R["records"] if x["kind"] == kind)
MUT = {"registered point value +0.5": m_points, "cohort mean +1e-6": m_mean, "complete-case mask flipped": m_unavail,
       "two-level p-value x1.001": m_p, "two-level q-value": m_q, "Welch t sign flipped": m_t, "two-level Hedges g +1e-4": m_hg,
       "Hedges g replaced by an uncorrected d": m_hg_cohen, "non-computable location reported with a p-value": m_nc,
       "multi-level number of groups +1": m_ngroups, "multi-level contrast count -1": m_ncontrast, "Welch ANOVA F x1.0001": m_F,
       "omnibus p x1.001": m_omni_p, "omnibus q (BH)": m_omni_q, "omnibus omega^2 +1e-5": m_omega, "Games-Howell p x1.01": m_gh_p,
       "pairwise Hedges g +1e-4": m_gh_g, "post hoc applied outside the omnibus gate": m_gate,
       "Games-Howell replaced by uncorrected Welch p (no multiplicity control)": m_ntests, "correction text removed": m_text, "correlation +1e-5": m_corr,
       "n_sig +1": m_nsig, "by-category coefficient +1e-5": m_cbc_coef, "by-category q (BH within category) x1.01": m_cbc_q,
       "pair-path correlation p x1.001": m_pair_corr, "pair-path group t sign flipped": m_pair_group,
       "OD|OS categorical split records missing (missing per-eye split)": m_drop_pair_split, "pair-path mean +1e-6": m_pair_mean,
       "pair-path OD record labelled as OS": m_swap_eye, "cluster % +0.01": m_cluster, "regression slope +1e-6": m_fit, "scotoma registered % +0.1": m_scot, "scotoma OD row % +0.1": m_scot_od,
       "scotoma: OD/OS rows missing (old single-table behaviour)": m_scot_noeye, "scotoma: OS row = registered values": m_scot_mirrored}
if base["records"] and any(r["kind"] == "md_subject" for r in base["records"]): MUT["MD +0.01"] = m_md
bad = 0
for idx, (label, fn) in enumerate(MUT.items()):
    if a.only and a.only not in label: continue
    if a.shard and idx % int(a.shard.split('/')[1]) != int(a.shard.split('/')[0]): continue
    tmp = tempfile.mkdtemp(); dst = os.path.join(tmp, name); shutil.copytree(a.export, dst)
    R = copy.deepcopy(base); fn(R); json.dump(R, open(os.path.join(dst, "records.json"), "w"))
    out = os.path.join(tmp, "rep")
    subprocess.run([sys.executable, os.path.join(HERE, "run_oracle.py"), "--databases", a.databases, "--exports", tmp, "--out", out,
                    "--only", name.replace("neurovf_", "")], capture_output=True)
    rep = json.load(open(os.path.join(out, "oracle_report.json")))
    caught = rep["fail"] > 0
    bad += (not caught)
    print(("CAUGHT " if caught else "MISSED ") + label + f"  ({rep['fail']} failing checks)")
    shutil.rmtree(tmp, ignore_errors=True)
print("\nMUTATION SELF-CHECK:", "every injected error was detected" if bad == 0 else f"{bad} injected error(s) NOT detected")
sys.exit(1 if bad else 0)
