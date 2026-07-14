#%% ===============================================================
# 4_make_figures_tables.py — evaluation figures and LaTeX tables
# -----------------------------------------------------------------
# Consumes the results of scripts 1-3 and writes every remaining
# paper artifact into ../figures and ../tables:
#   nested_cv_scheme, hist_target, confusion_matrix, parity_fis,
#   nested_boxplots, error_by_zone (figures)
#   nomenclature, feature_stats, class_distribution, tree_metrics,
#   rules_compact, comparison (tables)
# =================================================================
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.dirname(HERE)
PAPER_DIR = os.path.dirname(CODE_DIR)
sys.path.insert(0, HERE)

from nomenclature import TARGET, NOMENCLATURE
from fis_core import EDGES, LEVELS, to_levels
from style import (LEVEL_EDGE, LEVEL_FILL, METHOD_COLOR, GRAY,
                   heat_cmap, save_fig)
from diagrams import autobox, connect, glossary

RESULTS_DIR = os.path.join(CODE_DIR, "results")
FIG_DIR = os.path.join(PAPER_DIR, "figures")
TAB_DIR = os.path.join(PAPER_DIR, "tables")
os.makedirs(TAB_DIR, exist_ok=True)

oof = pd.read_csv(os.path.join(RESULTS_DIR, "oof_predictions.csv"))
folds = pd.read_csv(os.path.join(RESULTS_DIR, "method_folds.csv"))
comp = pd.read_csv(os.path.join(RESULTS_DIR, "comparison.csv"))
v, y3 = oof["v"].values, oof["class"].values
METHODS = list(METHOD_COLOR)

#%% ===============================================================
# Figure: nested cross-validation schematic
# =================================================================
fig, ax = plt.subplots(figsize=(8.6, 5.6))
ax.axis("off")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

n_top = autobox(ax, 0.5, "Complete dataset  (n = 897, stratified by "
                "normative class)", (0.8, 1, 1), (0, 0.5, 0.5), fs=11,
                top=0.985)
# outer folds band
y0, h, w = 0.775, 0.085, 0.155
for k in range(5):
  x = 0.115 + k * 0.165
  fc = (1, 0.8, 0.8) if k == 1 else (0.9, 1, 1)
  ec = (0.8, 0, 0) if k == 1 else (0, 0.5, 0.5)
  ax.add_patch(plt.Rectangle((x - w / 2, y0 - h / 2), w, h, fc=fc, ec=ec,
                             lw=1.5))
  ax.text(x, y0, f"outer\nfold {k + 1}", ha="center", va="center",
          fontsize=9)
ax.annotate("", xy=(0.5, y0 + h / 2 + 0.005),
            xytext=(n_top["cx"], n_top["cy"] - n_top["h"] / 2),
            arrowprops=dict(arrowstyle="-|>", color="#444"))
ax.text(0.92, y0, "x5", fontsize=11, ha="center", color="#444")

n_tr = autobox(ax, 0.30, "Outer training fold (4/5)\nALL selection "
               "happens here", (0.9, 1, 1), (0, 0.5, 0.5), fs=10,
               top=0.655)
n_te = autobox(ax, 0.80, "Outer test fold (1/5)\nnever seen during\n"
               "any selection or fit", (1, 0.8, 0.8), (0.8, 0, 0),
               fs=10, top=0.655)
ax.annotate("", xy=(n_tr["cx"], n_tr["cy"] + n_tr["h"] / 2),
            xytext=(0.28, y0 - h / 2),
            arrowprops=dict(arrowstyle="-|>", color="#444"))
ax.annotate("", xy=(n_te["cx"], n_te["cy"] + n_te["h"] / 2),
            xytext=(0.28 + 0.165, y0 - h / 2),
            arrowprops=dict(arrowstyle="-|>", color="#444"))

n_in = autobox(ax, 0.30, "Inner 4-fold CV:\ntree depth, leaf size, "
               "pruning,\nleaf threshold  (mean recall)",
               (0.9, 1, 1), (0, 0.7, 0.7), fs=9.5,
               top=n_tr["cy"] - n_tr["h"] / 2 - 0.035)
n_fit = autobox(ax, 0.30, "Refit the complete chain:\ntrees "
                r"$\rightarrow$ rules $\rightarrow$ membership "
                "functions\n"
                r"$\rightarrow$ consequent tuning",
                (0.9, 1, 1), (0, 0.5, 0.5), fs=9.5,
                top=n_in["cy"] - n_in["h"] / 2 - 0.035)
connect(ax, n_tr, n_in)
connect(ax, n_in, n_fit)

n_pred = autobox(ax, 0.80, "Out-of-fold predictions",
                 (1, 0.8, 0.8), (0.6, 0, 0), fs=10, cy=n_fit["cy"])
ax.annotate("", xy=(n_pred["cx"] - n_pred["w"] / 2, n_pred["cy"]),
            xytext=(n_fit["cx"] + n_fit["w"] / 2, n_fit["cy"]),
            arrowprops=dict(arrowstyle="-|>", color="#444"))
ax.annotate("", xy=(n_pred["cx"], n_pred["cy"] + n_pred["h"] / 2),
            xytext=(n_te["cx"], n_te["cy"] - n_te["h"] / 2),
            arrowprops=dict(arrowstyle="-|>", color="#444"))

n_out = autobox(ax, 0.5, "Metrics over the union of the 5 outer test "
                "folds\n(every record predicted exactly once, always "
                "by a model that never saw it)",
                (0.8, 1, 1), (0, 0.3, 0.3), fs=10,
                top=n_fit["cy"] - n_fit["h"] / 2 - 0.05)
connect(ax, n_fit, n_out)
connect(ax, n_pred, n_out)
save_fig(FIG_DIR, "nested_cv_scheme", fig)
print("nested CV scheme saved")

#%% ===============================================================
# Figure: target histogram with normative limits
# =================================================================
fig = plt.figure(figsize=(6, 3.2))
plt.hist(v, bins=40, color=(0.4, 1, 1), edgecolor=(0, 0.5, 0.5))
for e in EDGES:
  plt.axvline(e, color=(0.8, 0, 0), ls="--", lw=1.5)
plt.axvline(20, color=(0.6, 0, 0), ls="-", lw=1.5)
plt.xlabel("$v$ [mpy]")
plt.ylabel("Records")
plt.grid(alpha=0.3)
save_fig(FIG_DIR, "hist_target")

#%% ===============================================================
# Figure: confusion matrix (compact, red/teal scale)
# =================================================================
cm = pd.read_csv(os.path.join(RESULTS_DIR, "confusion.csv")).values
cmn = cm / cm.sum(axis=1, keepdims=True)
fig = plt.figure(figsize=(3.6, 3.2))
plt.imshow(cmn, cmap=heat_cmap(), vmin=0, vmax=1)
for i in range(3):
  for j in range(3):
    plt.text(j, i, f"{cmn[i, j]:.2f}\n({cm[i, j]})", ha="center",
             va="center", fontsize=9,
             color="white" if cmn[i, j] > 0.55 else "black")
ticks = [f"{l}" for l in LEVELS]
plt.xticks(range(3), ticks, fontsize=9)
plt.yticks(range(3), ticks, fontsize=9)
plt.xlabel("Predicted class")
plt.ylabel("Measured class")
save_fig(FIG_DIR, "confusion_matrix")

#%% ===============================================================
# Figure: parity with error bands (house style)
# =================================================================
pred = oof["Proposed FIS"].values
fig = plt.figure(figsize=(4.8, 4.8))
hi = float(max(v.max(), np.nanmax(pred))) * 1.03
xx = np.linspace(0, hi, 10)
plt.fill_between(xx, xx - 2, xx + 2, color=(0.8, 1, 1), alpha=0.7,
                 label=r"$\pm 2$ mpy")
plt.fill_between(xx, xx - 1, xx + 1, color=(0.4, 1, 1), alpha=0.7,
                 label=r"$\pm 1$ mpy")
plt.plot(xx, xx, "k--", lw=1)
plt.scatter(v, pred, s=16, alpha=0.55, color=(0.6, 0, 0),
            edgecolors="k", linewidths=0.2)
for e in EDGES:
  plt.axvline(e, color=GRAY, ls=":", lw=1)
  plt.axhline(e, color=GRAY, ls=":", lw=1)
plt.xlabel("Measured $v$ [mpy]")
plt.ylabel(r"Predicted $v$ [mpy] (out-of-fold)")
plt.xlim(0, hi)
plt.ylim(0, hi)
plt.legend(fontsize=8, loc="upper left")
plt.grid(alpha=0.25)
save_fig(FIG_DIR, "parity_fis")

#%% ===============================================================
# Figure: per-fold boxplots (R2 and MAE, four methods)
# =================================================================
fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
for ax, metric, label in [(axes[0], "r2", "$R^2$ per outer fold"),
                          (axes[1], "mae", "MAE per outer fold [mpy]")]:
  data = [folds[folds.method == m][metric].values for m in METHODS]
  bp = ax.boxplot(data, patch_artist=True, widths=0.55,
                  medianprops=dict(color="black", lw=1.4),
                  whiskerprops=dict(color=GRAY),
                  capprops=dict(color=GRAY), showfliers=False)
  for patch, m in zip(bp["boxes"], METHODS):
    c = METHOD_COLOR[m]
    patch.set_facecolor((*c, 0.35) if len(c) == 3 else c)
    patch.set_edgecolor(c)
    patch.set_linewidth(1.6)
  for i, d in enumerate(data):
    ax.scatter(np.full(len(d), i + 1), d, s=18, zorder=3,
               color=METHOD_COLOR[METHODS[i]], edgecolors="k",
               linewidths=0.3)
  ax.set_xticks(range(1, len(METHODS) + 1))
  ax.set_xticklabels(["FIS", "Linear", "Tree reg.", "SVR"], fontsize=9)
  ax.set_ylabel(label)
  ax.grid(alpha=0.3, axis="y")
plt.tight_layout()
save_fig(FIG_DIR, "nested_boxplots", fig)

#%% ===============================================================
# Figure: absolute error by severity zone (four methods)
# =================================================================
fig = plt.figure(figsize=(7, 3.2))
w = 0.19
xs = np.arange(3)
for i, m in enumerate(METHODS):
  err = np.abs(v - oof[m].values)
  means = [err[y3 == k].mean() for k in range(3)]
  plt.bar(xs + (i - 1.5) * w, means, w, color=METHOD_COLOR[m],
          label=m, edgecolor="white")
plt.xticks(xs, [f"{LEVELS[k]}" for k in range(3)])
plt.ylabel("Mean absolute error [mpy]")
plt.legend(fontsize=8)
plt.grid(alpha=0.3, axis="y")
save_fig(FIG_DIR, "error_by_zone")
print("evaluation figures saved")


#%% ===============================================================
# LaTeX tables (see tables_gen.py)
# =================================================================
import tables_gen
tables_gen.make_all(RESULTS_DIR, TAB_DIR, comp)
