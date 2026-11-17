#!/usr/bin/env python3
"""
stats.py -- WP6: statistical rigor for the CCM-vs-global comparison (R4: "3 seeds, no stat tests").

Reads the per-seed detection AUROCs already saved by eval_baselines (results/baselines/*.json) and
computes, for the class-conditional CCM vs its global ablation:
  * per-setting paired difference (CCM - global), pooled across all (dataset x model x trigger x attack)
    settings and seeds;
  * a 95% bootstrap CI on the mean paired difference;
  * a Wilcoxon signed-rank test (paired, non-parametric);
  * the fraction of settings where CCM >= global;
broken down overall and by dataset (the story differs: strong on CWRU, weak on MFPT).

  conda activate qsentry && python stats.py
"""
import os, sys, glob, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

try:
    from scipy.stats import wilcoxon
except Exception:
    wilcoxon = None

_RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
BASE = os.path.join(_RES, "baselines_repr") if os.path.isdir(os.path.join(_RES, "baselines_repr")) \
    else os.path.join(_RES, "baselines")


def boot_ci(x, iters=10000, seed=0):
    x = np.asarray(x, float); rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), size=(iters, len(x)))].mean(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def collect():
    """Return list of dicts: one per (file, attack) with per-seed CCM and global AUROC arrays."""
    rows = []
    for f in sorted(glob.glob(os.path.join(BASE, "baselines_*.json"))):
        d = json.load(open(f)); cfg = d["config"]
        ds, model, trig = cfg["dataset"], cfg["model"], cfg["trig_kind"]
        for attack, methods in d["results"].items():
            ccm = np.array([s[0] for s in methods["ccm_classcond"]])   # per-seed AUROC
            glob_ = np.array([s[0] for s in methods["global"]])
            # TPR at the 1% false-alarm budget, when the result file carries it
            ccm1 = np.array([s[2] for s in methods["ccm_classcond"] if len(s) > 2])
            glob1 = np.array([s[2] for s in methods["global"] if len(s) > 2])
            rows.append({"dataset": ds, "model": model, "trigger": trig, "attack": attack,
                         "ccm": ccm, "global": glob_, "ccm_tpr1": ccm1, "global_tpr1": glob1})
    return rows


def summarize(rows, label, key=("ccm", "global"), metric="AUROC"):
    ccm = np.concatenate([r[key[0]] for r in rows])
    glo = np.concatenate([r[key[1]] for r in rows])
    if len(ccm) == 0:
        return None
    diff = ccm - glo
    lo, hi = boot_ci(diff)
    frac_setting = np.mean([r[key[0]].mean() >= r[key[1]].mean() for r in rows
                            if len(r[key[0]])])
    w = ""
    if wilcoxon is not None and len(diff) >= 6 and np.any(diff != 0):
        try:
            stat, p = wilcoxon(ccm, glo); w = f"Wilcoxon p={p:.2e}"
        except Exception as e:
            w = f"Wilcoxon n/a ({type(e).__name__})"
    print(f"\n[{label}]  metric={metric}  n_settings={len(rows)}  n_pairs={len(diff)}")
    print(f"  CCM mean {metric}   = {ccm.mean():.3f}")
    print(f"  global mean {metric}= {glo.mean():.3f}")
    print(f"  mean diff (CCM-global) = {diff.mean():+.3f}  95% bootstrap CI [{lo:+.3f}, {hi:+.3f}]")
    print(f"  settings with CCM>=global: {frac_setting*100:.0f}%   {w}")
    return {"label": label, "metric": metric, "n_settings": len(rows), "n_pairs": int(len(diff)),
            "ccm_mean": round(float(ccm.mean()), 4), "global_mean": round(float(glo.mean()), 4),
            "mean_diff": round(float(diff.mean()), 4), "ci95": [round(lo, 4), round(hi, 4)],
            "frac_ccm_ge_global": round(float(frac_setting), 3),
            "wilcoxon": w}


def hier_boot(rows, key, iters=10000, seed=0):
    """Two-stage (hierarchical) bootstrap of the mean paired difference: resample SETTINGS with
    replacement, then resample SEEDS within each drawn setting. Seeds of one setting share a model
    family, a dataset and a trigger, so they are not independent draws (R4, minor revision 2); this
    keeps the setting as the unit of replication. Returns (mean, lo, hi, p_two_sided)."""
    rng = np.random.default_rng(seed)
    diffs = [r[key[0]] - r[key[1]] for r in rows if len(r[key[0]])]
    S = len(diffs)
    means = np.empty(iters)
    for b in range(iters):
        pick = rng.integers(0, S, S)
        means[b] = np.mean([diffs[i][rng.integers(0, len(diffs[i]), len(diffs[i]))].mean()
                            for i in pick])
    obs = float(np.mean([d.mean() for d in diffs]))
    lo, hi = np.percentile(means, [2.5, 97.5])
    # two-sided bootstrap p for H0: mean diff = 0 (bootstrap distribution shifted to the null);
    # resolution is bounded by 1/(iters+1)
    null = means - means.mean()
    p = float((np.sum(np.abs(null) >= abs(obs)) + 1) / (iters + 1))
    return obs, float(lo), float(hi), p


def summarize_settings(rows, label, key=("ccm", "global"), metric="AUROC"):
    """Settings-level analysis: one paired value per setting (seed-averaged), Wilcoxon over settings,
    plus the hierarchical bootstrap CI. This is what the paper reports (Table tab:stats)."""
    rows = [r for r in rows if len(r[key[0]])]
    if not rows:
        return None
    a = np.array([r[key[0]].mean() for r in rows]); g = np.array([r[key[1]].mean() for r in rows])
    d = a - g
    w = ""
    if wilcoxon is not None and len(d) >= 6 and np.any(d != 0):
        try:
            _, p_w = wilcoxon(a, g); w = f"Wilcoxon p={p_w:.2e}"
        except Exception as e:
            w = f"Wilcoxon n/a ({type(e).__name__})"
    obs, lo, hi, p_b = hier_boot(rows, key)
    print(f"\n[{label}]  SETTINGS-LEVEL  metric={metric}  n_settings={len(rows)}")
    print(f"  CCM {a.mean():.3f}  global {g.mean():.3f}  diff {obs:+.3f}  "
          f"hier-bootstrap 95% CI [{lo:+.3f}, {hi:+.3f}]  p_boot={p_b:.1e}  {w}"
          f"  CCM>global in {np.mean(d > 0)*100:.0f}% of settings")
    return {"label": label, "metric": metric, "level": "settings", "n_settings": len(rows),
            "ccm_mean": round(float(a.mean()), 4), "global_mean": round(float(g.mean()), 4),
            "mean_diff": round(obs, 4), "ci95": [round(lo, 4), round(hi, 4)],
            "p_boot": p_b, "wilcoxon": w, "frac_ccm_gt_global": round(float(np.mean(d > 0)), 3)}


def main():
    rows = collect()
    if not rows:
        print("No baseline JSONs found in results/baselines/. Run eval_baselines.py first."); return
    # settings-level (reported): one paired value per setting, hierarchical bootstrap over seeds
    sl = []
    for metric, key in (("AUROC", ("ccm", "global")), ("TPR@1%", ("ccm_tpr1", "global_tpr1"))):
        sl.append(summarize_settings(rows, "ALL settings", key, metric))
        for ds in sorted(set(r["dataset"] for r in rows)):
            sl.append(summarize_settings([r for r in rows if r["dataset"] == ds],
                                         f"dataset={ds}", key, metric))
    outp = os.path.join(BASE, "..", "stats_ccm_vs_global_settings.json")
    json.dump([x for x in sl if x], open(outp, "w"), indent=2)
    print(f"\nSaved settings-level -> {os.path.abspath(outp)}")
    print("\n# ---- previous seed-pooled analysis (kept for the response letter; treats seeds as "
          "independent, which overstates certainty) ----")
    out = [x for x in [summarize(rows, "ALL settings")] if x]
    for ds in sorted(set(r["dataset"] for r in rows)):
        out.append(summarize([r for r in rows if r["dataset"] == ds], f"dataset={ds}"))
    # the same paired comparison at the 1% false-alarm operating point
    t1 = [summarize(rows, "ALL settings", ("ccm_tpr1", "global_tpr1"), "TPR@1%")]
    for ds in sorted(set(r["dataset"] for r in rows)):
        t1.append(summarize([r for r in rows if r["dataset"] == ds],
                            f"dataset={ds}", ("ccm_tpr1", "global_tpr1"), "TPR@1%"))
    out += [x for x in t1 if x]
    print("\nInterpretation: a positive mean diff with a CI excluding 0 => CCM significantly beats the "
          "global ablation in that stratum. Expect strong on CWRU, near-zero/negative on MFPT.")
    outp = os.path.join(BASE, "..", "stats_ccm_vs_global.json")
    json.dump(out, open(outp, "w"), indent=2)
    print(f"\nSaved -> {os.path.abspath(outp)}")


if __name__ == "__main__":
    main()
