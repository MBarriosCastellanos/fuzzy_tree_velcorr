#%% ===============================================================
# 2_run_nested_evaluation.py — honest evaluation of all methods
# -----------------------------------------------------------------
# One outer StratifiedKFold(5, seed 42) on the normative class is
# shared by every method. Inside each outer training fold, every
# selection step (tree hyperparameters, benchmark grids) runs in
# inner folds only; the outer test folds provide out-of-fold
# predictions and per-fold metrics.
#
# Methods: proposed FIS, linear regression, decision-tree regressor
# and SVR (methods suited to datasets of this size).
# =================================================================
import json
import os
import sys
from itertools import product

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

HERE = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from nomenclature import TARGET
from fis_core import (EDGES, RANDOM_STATE, nested_cv_fis, recall_metrics,
                      select_tree_params, to_levels)

DATA_CSV = os.path.join(CODE_DIR, "data", "processed.csv")
RESULTS_DIR = os.path.join(CODE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

N_OUTER, N_INNER = 5, 4

df = pd.read_csv(DATA_CSV)
v = df[TARGET].values.astype(float)
X = df.drop(columns=[TARGET])
y3 = to_levels(v)
outer = StratifiedKFold(N_OUTER, shuffle=True, random_state=RANDOM_STATE)

#%% ===============================================================
# 1. Binomial trees: OOF recall metrics + 3-class confusion
# =================================================================
print("== binomial trees (nested CV) ==")
oof_bin = {th: np.full(len(v), -1) for th in EDGES}
for tr, te in outer.split(X, y3):
  tp = select_tree_params(X.iloc[tr], v[tr])
  for th, key in zip(EDGES, ("th2", "th8")):
    p = dict(tp[key] if "th2" in tp else tp)
    lt = p.pop("leaf_thr", None) or 0.5
    clf = DecisionTreeClassifier(class_weight="balanced",
                                 random_state=RANDOM_STATE, **p)
    Xtr = X.iloc[tr].fillna(X.iloc[tr].median(numeric_only=True))
    Xte = X.iloc[te].fillna(X.iloc[tr].median(numeric_only=True))
    clf.fit(Xtr, (v[tr] > th).astype(int))
    oof_bin[th][te] = (clf.predict_proba(Xte)[:, 1] >= lt).astype(int)

tree_rows = []
for th in EDGES:
  yb = (v > th).astype(int)
  mr, ratio, recs = recall_metrics(yb, oof_bin[th])
  tree_rows.append({"limit_mpy": th, "mean_recall": round(mr, 3),
                    "ratio_recall": round(ratio, 3),
                    "recall_below": round(recs[0], 3),
                    "recall_above": round(recs[1], 3),
                    "accuracy": round(float((yb == oof_bin[th]).mean()), 3)})
  print(tree_rows[-1])
pd.DataFrame(tree_rows).to_csv(
  os.path.join(RESULTS_DIR, "tree_metrics.csv"), index=False)

ycomp = oof_bin[EDGES[0]] + oof_bin[EDGES[1]]
cm = np.zeros((3, 3), dtype=int)
for t, p in zip(y3, ycomp):
  cm[t, p] += 1
pd.DataFrame(cm).to_csv(os.path.join(RESULTS_DIR, "confusion.csv"),
                        index=False)
mr3, _, recs3 = recall_metrics(y3, ycomp)
print(f"3-class composed: mean_recall={mr3:.3f} "
      f"recalls={['%.2f' % r for r in recs3]}")

#%% ===============================================================
# 2. Proposed FIS (per-fold metrics + OOF predictions)
# =================================================================
print("\n== proposed FIS (nested CV) ==")
fis_res = nested_cv_fis(X, v)
fis_folds = pd.DataFrame(fis_res["folds"])
fis_folds["method"] = "Proposed FIS"

with open(os.path.join(RESULTS_DIR, "fis_params_by_fold.json"), "w",
          encoding="utf-8") as fp:
  json.dump(fis_res["params_by_fold"], fp, indent=2, default=str)

#%% ===============================================================
# 3. Benchmarks on identical folds
# =================================================================
BENCHMARKS = {
  "Linear regression": (LinearRegression(), {}),
  "Decision tree": (DecisionTreeRegressor(random_state=RANDOM_STATE),
                    {"model__max_depth": [3, 4, 5, 6],
                     "model__min_samples_leaf": [3, 5, 10]}),
  "SVR": (SVR(), {"model__C": [1, 10, 100],
                  "model__epsilon": [0.1, 1.0]}),
}


def grid_iter(grid):
  if not grid:
    yield {}
    return
  keys = list(grid)
  for combo in product(*(grid[k] for k in keys)):
    yield dict(zip(keys, combo))


def make_pipe(model):
  return Pipeline([("imputer", SimpleImputer(strategy="median")),
                   ("scaler", StandardScaler()), ("model", model)])


print("\n== benchmarks (nested CV) ==")
bench_folds, oof_cols = [], {"v": v, "class": y3,
                             "Proposed FIS": fis_res["oof_pred"]}
for mname, (model, grid) in BENCHMARKS.items():
  oof = np.full(len(v), np.nan)
  for k, (tr, te) in enumerate(outer.split(X, y3)):
    inner = StratifiedKFold(N_INNER, shuffle=True,
                            random_state=RANDOM_STATE)
    best_p, best_mae = None, np.inf
    for p in grid_iter(grid):
      maes = []
      for itr, iva in inner.split(X.iloc[tr], y3[tr]):
        pipe = make_pipe(model).set_params(**p)
        pipe.fit(X.iloc[tr].iloc[itr], v[tr][itr])
        pr = np.clip(pipe.predict(X.iloc[tr].iloc[iva]), 0, None)
        maes.append(mean_absolute_error(v[tr][iva], pr))
      if np.mean(maes) < best_mae:
        best_p, best_mae = p, float(np.mean(maes))
    pipe = make_pipe(model).set_params(**best_p)
    pipe.fit(X.iloc[tr], v[tr])
    pred = np.clip(pipe.predict(X.iloc[te]), 0, None)
    oof[te] = pred
    bench_folds.append({"fold": k, "mae": mean_absolute_error(v[te], pred),
                        "r2": r2_score(v[te], pred), "method": mname})
  oof_cols[mname] = oof
  print(f"{mname}: OOF MAE={mean_absolute_error(v, oof):.3f} "
        f"R2={r2_score(v, oof):.3f}")

folds = pd.concat([fis_folds[["fold", "mae", "r2", "method"]],
                   pd.DataFrame(bench_folds)], ignore_index=True)
folds.to_csv(os.path.join(RESULTS_DIR, "method_folds.csv"), index=False)
pd.DataFrame(oof_cols).to_csv(
  os.path.join(RESULTS_DIR, "oof_predictions.csv"), index=False)

#%% ===============================================================
# 4. Comparison summary (mean and min-max across outer folds)
# =================================================================
rows = []
for mname in ["Proposed FIS"] + list(BENCHMARKS):
  oof = oof_cols[mname]
  g = folds[folds.method == mname]
  mr, _, recs = recall_metrics(y3, to_levels(oof))
  rows.append({
    "method": mname,
    "mae_oof": round(mean_absolute_error(v, oof), 3),
    "mae_min": round(g.mae.min(), 3), "mae_max": round(g.mae.max(), 3),
    "r2_oof": round(r2_score(v, oof), 3),
    "r2_min": round(g.r2.min(), 3), "r2_max": round(g.r2.max(), 3),
    "recall_high": round(recs[-1], 3)})
comp = pd.DataFrame(rows)
comp.to_csv(os.path.join(RESULTS_DIR, "comparison.csv"), index=False)
print("\n" + comp.to_string(index=False))
