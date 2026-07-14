#%% ===============================================================
# 3_final_system.py — deployed system and its interpretable artifacts
# -----------------------------------------------------------------
# Refits the complete chain on the full dataset (deployment only;
# every performance figure comes from script 2) and produces:
#   figures/tree_th2.pdf, tree_th8.pdf   custom tree diagrams
#   figures/mf_three_panel.pdf           interval MFs (2 inputs + output)
#   figures/consequent_singletons.pdf    tuned consequents
#   figures/rule_coverage.pdf            rule dominance / activation
#   figures/response_curve.pdf           smooth output vs v_mod
#   results/rules.csv                    rulebase with physics symbols
# =================================================================
import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.dirname(HERE)
PAPER_DIR = os.path.dirname(CODE_DIR)
sys.path.insert(0, HERE)

from nomenclature import TARGET, MEANING, UNIT, mt, lx
from fis_core import (EDGES, LEVELS, collapse_tree, fit_fis,
                      feature_partitions, mf_interval,
                      select_fis_config, select_tree_params,
                      to_levels)
from style import (LEVEL_EDGE, LEVEL_FILL, GRAY, save_fig)
from diagrams import draw_tree, glossary

DATA_CSV = os.path.join(CODE_DIR, "data", "processed.csv")
RESULTS_DIR = os.path.join(CODE_DIR, "results")
FIG_DIR = os.path.join(PAPER_DIR, "figures")

df = pd.read_csv(DATA_CSV)
v = df[TARGET].values.astype(float)
X = df.drop(columns=[TARGET])

#%% ===============================================================
# Deployed system (full-data refit)
# =================================================================
tp = select_tree_params(X, v)
tpf, sc, lam = select_fis_config(X, v, tp)
fis = fit_fis(X, v, tree_params=tpf, shoulder_scale=sc, 
              reg_lambda=lam)
print(f"deployed system: {fis.meta['n_rules']} rules, "
      f"{len(fis.used)} variables, scale {sc}, lambda {lam}")
print("variables:", fis.used)
with open(os.path.join(RESULTS_DIR, "deployed_params.json"), "w",
          encoding="utf-8") as fp:
  json.dump({"tree_params": tpf, "shoulder_scale": sc,
             "reg_lambda": lam,
             "consequents": fis.consequents, "used": fis.used},
            fp, indent=2, default=str)

#%% ===============================================================
# Rulebase table with physics symbols + coverage statistics
# =================================================================
# Contribution share = mean normalized activation weight: the share
# of the weighted-average output that each rule carries, overall and
# within the records of its own severity zone.
W = fis.activation_matrix(X)
Wn = W / np.clip(W.sum(axis=1, keepdims=True), 1e-9, None)
y3 = to_levels(v)


def cond_latex(f, op, thr):
  op_tex = r"\leq" if op == "<=" else ">"
  return f"{lx(f).strip('$')} {op_tex} {thr:.3g}"


rows = []
for j, r in enumerate(fis.rules):
  ante = r" \wedge ".join(cond_latex(*c) for c in r["conds"])
  zone = y3 == r["level"]
  rows.append({
    "rule": f"R{j + 1}",
    "level": LEVELS[r["level"]],
    "consequent_mpy": round(fis.consequents[j], 2),
    "n_conditions": len(r["conds"]),
    "antecedent_latex": ante,
    "antecedent_plain": " AND ".join(
      f"{f} {op} {thr:.3g}" for f, op, thr in r["conds"]),
    "contrib_pct": round(100 * float(Wn[:, j].mean()), 1),
    "zone_contrib_pct": round(100 * float(Wn[zone, j].mean()), 1)})
rules_df = pd.DataFrame(rows)
rules_df.to_csv(os.path.join(RESULTS_DIR, "rules.csv"), index=False)
print(rules_df[["rule", "level", "consequent_mpy", "contrib_pct",
                "zone_contrib_pct"]].to_string(index=False))

#%% ===============================================================
# Tree diagrams (custom horizontal rendering)
# =================================================================
for th, key, labels, levels in [
    (EDGES[0], "th2", (r"$v > 2$ mpy", r"$v \leq 2$ mpy"), (1, 0)),
    (EDGES[1], "th8", (r"$v > 8$ mpy", r"$v \leq 8$ mpy"), (2, 0))]:
  tree = fis.trees[key]
  root = collapse_tree(tree, fis.meta["leaf_thr"][key])
  fig = draw_tree(root, fis.meta["feat_names"], mt,
                  positive_label=labels[0], negative_label=labels[1],
                  pos_level=levels[0], neg_level=levels[1],
                  figsize=(11, 4) if key == "th2" else (9, 7))

  def used_feats(node, acc):
    if not node["leaf"]:
      acc.add(fis.meta["feat_names"][node["feature"]])
      used_feats(node["L"], acc)
      used_feats(node["R"], acc)
    return acc

  used = sorted(used_feats(root, set()))

  def plain(s):
    return (s.replace("$", "").replace("\\mu", "u").replace("\\circ", "")
            .replace("^{", "").replace("_{", "").replace("{", "")
            .replace("}", "").replace("\\", ""))

  glossary(fig, [f"{f}: {plain(MEANING[f])} [{plain(UNIT[f])}]"
                 for f in used], per=3)
  save_fig(FIG_DIR, f"tree_th{int(th)}", fig)
print("tree diagrams saved")

#%% ===============================================================
# Interval membership functions: two inputs + the output partition
# =================================================================
parts = feature_partitions(fis.rules)
Xf = X.fillna(fis.medians_)
# the two variables that appear most often in the rulebase
from collections import Counter
freq = Counter(f for r in fis.rules for (f, _, _) in r["conds"])
var_a, var_b = [f for f, _ in freq.most_common(2)]
print(f"MF panel variables: {var_a}, {var_b}")
fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))
for ax, f in zip(axes[:2], (var_a, var_b)):
  thrs = parts[f]
  xv = Xf[f].values
  span = float(np.percentile(xv, 99) - np.percentile(xv, 1)) or 1.0
  # robust view window: outliers must not hide the transitions
  wlo = min(float(np.percentile(xv, 1)), min(thrs) - 1)
  whi = max(float(np.percentile(xv, 99)), max(thrs) * 1.3)
  uni = np.linspace(wlo, whi, 1500)
  bounds = [None] + thrs + [None]
  axh = ax.twinx()
  axh.hist(xv[(xv >= wlo) & (xv <= whi)], bins=40, color=(0.8, 1, 1),
           edgecolor="none", alpha=0.8, zorder=0)
  axh.set_yticks([])
  ax.set_xlim(wlo, whi)
  shades = [(0, 0.5, 0.5), (1, 0.5, 0.5), (0.6, 0, 0), (0, 0.3, 0.3),
            (1, 0, 0)]
  for i in range(len(thrs) + 1):
    blo, bhi = bounds[i], bounds[i + 1]
    m = np.ones(len(xv), dtype=bool)
    if blo is not None:
      m &= xv > blo
    if bhi is not None:
      m &= xv <= bhi
    mf = mf_interval(uni, blo, bhi, xv[m], span,
                     fis.meta["shoulder_scale"])
    lab = (f"$\\leq {thrs[0]:.3g}$" if i == 0 else
           f"$> {thrs[-1]:.3g}$" if i == len(thrs) else
           f"$({thrs[i - 1]:.3g}, {thrs[i]:.3g}]$")
    ax.plot(uni, mf, lw=2, color=shades[i % len(shades)], label=lab,
            zorder=3)
  for t in thrs:
    ax.axvline(t, color=(0.8, 0, 0), ls="--", lw=1.1, zorder=2)
  ax.set_xlabel(f"{mt(f)} [{UNIT[f]}]")
  ax.set_ylim(-0.03, 1.09)
  ax.legend(fontsize=7, loc="center right")
  ax.set_zorder(axh.get_zorder() + 1)
  ax.patch.set_visible(False)
axes[0].set_ylabel("Membership")

# output linguistic partition on the target axis
ax = axes[2]
vuni = np.linspace(0, float(v.max()), 500)
cuts = [-np.inf] + EDGES + [np.inf]
for k in range(3):
  d = v[(v > cuts[k]) & (v <= cuts[k + 1])]
  lo_t = None if k == 0 else EDGES[k - 1]
  hi_t = None if k == 2 else EDGES[k]
  vspan = float(np.percentile(v, 99) - np.percentile(v, 1))
  mf = mf_interval(vuni, lo_t, hi_t, d, vspan,
                   fis.meta["shoulder_scale"])
  ax.plot(vuni, mf, lw=2, color=LEVEL_EDGE[k], label=LEVELS[k])
for e in EDGES:
  ax.axvline(e, color=(0.8, 0, 0), ls="--", lw=1.1)
ax.set_xlabel("$v$ [mpy]")
ax.set_ylim(-0.03, 1.09)
ax.legend(fontsize=8, loc="center right")
plt.tight_layout()
save_fig(FIG_DIR, "mf_three_panel", fig)
print("membership figure saved")

#%% ===============================================================
# Tuned consequent singletons (separate figure)
# =================================================================
fig = plt.figure(figsize=(7, 2.8))
top = float(v.max()) * 1.04
for (a, b), k in zip([(0, EDGES[0]), (EDGES[0], EDGES[1]),
                      (EDGES[1], top)], range(3)):
  plt.axvspan(a, b, color=LEVEL_FILL[k], alpha=0.55,
              label=f"{LEVELS[k]} range")
lv_y = {0: 0.3, 1: 0.55, 2: 0.8}
for c, r in zip(fis.consequents, fis.rules):
  k = r["level"]
  plt.plot([c, c], [0, lv_y[k]], color=LEVEL_EDGE[k], lw=1.4)
  plt.plot(c, lv_y[k], "o", color=LEVEL_EDGE[k], ms=6)
for e in EDGES:
  plt.axvline(e, color=(0.8, 0, 0), ls="--", lw=1.2)
plt.yticks(list(lv_y.values()), LEVELS)
plt.xlabel("$v$ [mpy]")
plt.xlim(0, top)
plt.legend(fontsize=8, loc="upper right")
save_fig(FIG_DIR, "consequent_singletons")
print("singletons figure saved")

#%% ===============================================================
# Rule contribution figure (share of the output per rule)
# =================================================================
fig, ax = plt.subplots(figsize=(7.5, 3.2))
xs = np.arange(len(rules_df))
cols = [LEVEL_EDGE[LEVELS.index(l)] for l in rules_df.level]
ax.bar(xs, rules_df.zone_contrib_pct, color=cols, edgecolor="white",
       label=None)
ax.plot(xs, rules_df.contrib_pct, "o--", color=GRAY, ms=4, lw=1.1)
ax.set_xticks(xs)
ax.set_xticklabels(rules_df.rule, rotation=0, fontsize=8)
ax.set_ylabel("Contribution share [%]")
handles = ([plt.Rectangle((0, 0), 1, 1, color=LEVEL_EDGE[k])
            for k in range(3)]
           + [plt.Line2D([0], [0], ls="--", marker="o", color=GRAY,
                         ms=4)])
ax.legend(handles, LEVELS + ["overall"], fontsize=8,
          loc="upper right", ncol=2)
ax.grid(alpha=0.3, axis="y")
save_fig(FIG_DIR, "rule_coverage", fig)
print("rule contribution figure saved")

#%% ===============================================================
# Response curve: smooth transitions along v_mod
# =================================================================
med_row = Xf.median(numeric_only=True)
grid_vmod = np.linspace(float(np.percentile(Xf[var_a], 1)),
                        float(np.percentile(Xf[var_a], 99)), 400)
Xs = pd.DataFrame([med_row] * len(grid_vmod))
Xs[var_a] = grid_vmod
pred = fis.predict(Xs)
fig = plt.figure(figsize=(6, 3.4))
plt.plot(grid_vmod, pred, lw=2, color=(0.6, 0, 0))
for e in EDGES:
  plt.axhline(e, color=(0, 0.5, 0.5), ls=":", lw=1.1)
  plt.axvline(e, color=(0.8, 0, 0), ls="--", lw=1.0, alpha=0.6)
plt.xlabel(f"{mt(var_a)} [{UNIT[var_a]}] (other variables at their medians)")
plt.ylabel(r"Predicted $v$ [mpy]")
plt.grid(alpha=0.3)
save_fig(FIG_DIR, "response_curve")
print("response curve saved")
