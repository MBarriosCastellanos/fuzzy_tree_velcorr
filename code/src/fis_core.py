#%% ===============================================================
# fis_core.py — tree-rule fuzzy inference system (proposed method)
# -----------------------------------------------------------------
# Self-contained implementation of the winner methodology:
#   1. two binomial decision trees at the normative limits (2, 8 mpy)
#      with hyperparameters selected by inner cross-validation;
#   2. root-to-leaf paths extracted as IF/THEN rules with an ordinal
#      exclusion on the mid rules;
#   3. one interval membership function per region between tree
#      thresholds, shaped by the training-data distribution, with
#      0.5 crossings anchored exactly at the thresholds;
#   4. consequent singletons tuned by box-constrained least squares
#      within the normative range of each rule;
#   5. weighted-average inference (Mamdani with singleton outputs,
#      equivalent to a zero-order Sugeno system).
# Depends only on numpy / pandas / scipy / scikit-learn.
# =================================================================
import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42
EDGES = [2.0, 8.0]                      # normative limits [mpy]
LEVELS = ["low", "mid", "high"]
_S95 = float(np.log(19.0))              # sigmoid: 0.5 at c, 0.95 at c+w


#%% ===============================================================
# Target discretization and recall metrics
# =================================================================
def to_levels(v, edges=EDGES):
  """Class index 0..len(edges) of each rate by the normative cuts."""
  y = np.zeros(len(v), dtype=int)
  for e in edges:
    y += (np.asarray(v, dtype=float) > e).astype(int)
  return y


def recall_metrics(y_true, y_pred):
  """(mean recall, min/max recall ratio, per-class recalls)."""
  recs = []
  for k in sorted(set(y_true)):
    m = y_true == k
    recs.append(float((y_pred[m] == k).mean()))
  return float(np.mean(recs)), float(min(recs) / max(recs)), recs


#%% ===============================================================
# Rule extraction from the binomial trees
# =================================================================
def collapse_tree(tree, leaf_thr=None):
  """Simplified tree structure: sibling leaves of the same class are
  merged recursively (their split does not change the decision), so
  the extracted rules contain no redundant conditions and the tree
  diagrams match the rulebase one-to-one."""
  t = tree.tree_

  def leaf_class(node):
    val = t.value[node][0]
    if leaf_thr is None:
      return int(np.argmax(val))
    return int(val[1] / max(val.sum(), 1e-12) >= leaf_thr)

  def build(node):
    if t.children_left[node] == -1:
      return {"leaf": True, "cls": leaf_class(node),
              "n": int(t.n_node_samples[node])}
    left = build(t.children_left[node])
    right = build(t.children_right[node])
    if left["leaf"] and right["leaf"] and left["cls"] == right["cls"]:
      return {"leaf": True, "cls": left["cls"],
              "n": int(t.n_node_samples[node])}
    return {"leaf": False, "feature": int(t.feature[node]),
            "thr": float(t.threshold[node]), "L": left, "R": right,
            "n": int(t.n_node_samples[node])}

  return build(0)


def extract_rules(tree, feat_names, positive=1, leaf_thr=None):
  """Root-to-leaf paths (of the collapsed tree) whose leaf votes for
  class `positive`. Returns [{conds: [(feat, op, thr)], n}]."""
  rules = []

  def walk(node, conds):
    if node["leaf"]:
      if node["cls"] == positive:
        rules.append({"conds": list(conds), "n": node["n"]})
      return
    f = feat_names[node["feature"]]
    walk(node["L"], conds + [(f, "<=", node["thr"])])
    walk(node["R"], conds + [(f, ">", node["thr"])])

  walk(collapse_tree(tree, leaf_thr), [])
  return rules


def build_rulebase(tree_low, tree_high, feat_names, leaf_thr=None):
  """Rules from both trees: T2 negative -> low, T2 positive -> mid,
  T8 positive -> high. The ordinal exclusion of the mid rules is
  applied at inference time (see FISModel)."""
  lt = leaf_thr if isinstance(leaf_thr, dict) else {"th2": leaf_thr,
                                                    "th8": leaf_thr}
  rules = []
  for r in extract_rules(tree_low, feat_names, 0, lt["th2"]):
    rules.append({**r, "level": 0, "tree": "th2"})
  for r in extract_rules(tree_low, feat_names, 1, lt["th2"]):
    rules.append({**r, "level": 1, "tree": "th2"})
  for r in extract_rules(tree_high, feat_names, 1, lt["th8"]):
    rules.append({**r, "level": 2, "tree": "th8"})
  return rules


def merge_thresholds(rules, X, tol=0.01):
  """Snap thresholds of one variable closer than tol times its
  robust span (1st-99th percentile, so outliers do not inflate the
  merging window)."""
  by_feat = {}
  for r in rules:
    for (f, _, thr) in r["conds"]:
      by_feat.setdefault(f, set()).add(thr)
  mapping = {}
  for f, thrs in by_feat.items():
    x = X[f].dropna().values
    rng = float(np.percentile(x, 99) - np.percentile(x, 1)) or 1.0
    groups, cur = [], []
    for t in sorted(thrs):
      if cur and (t - cur[-1]) > tol * rng:
        groups.append(cur)
        cur = []
      cur.append(t)
    groups.append(cur)
    for g in groups:
      for t in g:
        mapping[(f, t)] = float(np.mean(g))
  return [{**r, "conds": [(f, op, mapping[(f, t)])
                          for (f, op, t) in r["conds"]]} for r in rules]


def simplify_rules(rules):
  """Combine the conditions of each rule per variable into a single
  interval (max of lower bounds, min of upper bounds) and drop rules
  whose interval is empty (artifacts of threshold merging)."""
  out = []
  for r in rules:
    lo, hi = {}, {}
    for (f, op, thr) in r["conds"]:
      if op == ">":
        lo[f] = max(lo.get(f, -np.inf), thr)
      else:
        hi[f] = min(hi.get(f, np.inf), thr)
    if any(lo.get(f, -np.inf) >= hi.get(f, np.inf)
           for f in set(lo) | set(hi)):
      continue
    conds = []
    for f in dict.fromkeys(f for f, _, _ in r["conds"]):
      if f in lo:
        conds.append((f, ">", lo[f]))
      if f in hi:
        conds.append((f, "<=", hi[f]))
    out.append({**r, "conds": conds})
  return out


def feature_partitions(rules):
  """feature -> sorted unique thresholds in the rulebase."""
  parts = {}
  for r in rules:
    for (f, _, thr) in r["conds"]:
      parts.setdefault(f, set()).add(thr)
  return {f: sorted(t) for f, t in parts.items()}


def rules_readable(rules, symbols=None, digits=3, exclusive_mid=True):
  """IF/THEN strings using the physics symbols."""
  sym = symbols or {}
  lines = []
  for r in rules:
    c = " AND ".join(f"{sym.get(f, f)} {op} {thr:.{digits}g}"
                     for f, op, thr in r["conds"])
    if exclusive_mid and r["level"] == 1:
      c += " AND NOT (severe conditions)"
    lines.append(f"IF {c} THEN v IS {LEVELS[r['level']]}")
  return lines


#%% ===============================================================
# Interval membership functions
# =================================================================
def sigmf(x, c, s):
  """Logistic sigmoid centered at c with steepness s."""
  return 1.0 / (1.0 + np.exp(-np.clip(s * (np.asarray(x, float) - c),
                                      -60, 60)))


def mf_interval(x, lo_thr, hi_thr, data, span, scale=0.1):
  """Membership of the interval (lo_thr, hi_thr]: min of a rising
  and a falling sigmoid, each crossing 0.5 exactly at its threshold.
  The transition width is the distance from the threshold to the
  median of the training data inside the interval, times `scale`."""
  data = np.asarray(data, dtype=float)
  med = float(np.median(data)) if len(data) else None
  m = np.ones_like(np.asarray(x, dtype=float))
  if lo_thr is not None:
    w = (med - lo_thr) if med is not None and med > lo_thr else 0.1 * span
    w = float(np.clip(w * scale, 0.005 * span, span))
    m = np.fmin(m, sigmf(x, lo_thr, _S95 / w))
  if hi_thr is not None:
    w = (hi_thr - med) if med is not None and med < hi_thr else 0.1 * span
    w = float(np.clip(w * scale, 0.005 * span, span))
    m = np.fmin(m, sigmf(x, hi_thr, -_S95 / w))
  return m


#%% ===============================================================
# The fuzzy inference system
# =================================================================
class FISModel:
  """Fitted tree-rule FIS. Build with fit_fis()."""

  def __init__(self):
    self.rules = []
    self.used = []
    self.consequents = []      # tuned singleton per rule [mpy]
    self.medians_ = None       # train medians (NaN imputation)
    self.fallback_ = 0.0       # train median of the target
    self.trees = {}
    self.meta = {}
    self._cond_fns = []        # per rule: [(feat, membership fn)]
    self._levels = None

  def _activations(self, row):
    """Firing strength of each rule (min over its conditions) with
    the ordinal exclusion: mid rules cannot exceed 1 - max(high)."""
    w = np.empty(len(self._cond_fns))
    for j, conds in enumerate(self._cond_fns):
      wj = 1.0
      for f, fn in conds:
        wj = min(wj, fn(float(row[f])))
      w[j] = wj
    if self.meta["exclusive_mid"] and (self._levels == 2).any():
      w_high = float(w[self._levels == 2].max())
      mid = self._levels == 1
      w[mid] = np.minimum(w[mid], 1.0 - w_high)
    return w

  def predict(self, Xd):
    """Weighted average of the tuned singletons (zero-order Sugeno)."""
    Xd = Xd[self.meta["feat_names"]].fillna(self.medians_)
    out = np.empty(len(Xd))
    c = np.asarray(self.consequents)
    for i in range(len(Xd)):
      w = self._activations(Xd.iloc[i])
      s = w.sum()
      out[i] = float(w @ c / s) if s > 1e-9 else self.fallback_
    return out

  def activation_matrix(self, Xd):
    """n x R firing strengths (for coverage/interpretation plots)."""
    Xd = Xd[self.meta["feat_names"]].fillna(self.medians_)
    return np.vstack([self._activations(Xd.iloc[i])
                      for i in range(len(Xd))])


def _split_params(tree_params):
  base = dict(class_weight="balanced", random_state=RANDOM_STATE)
  tp = tree_params or {}
  if "th2" in tp or "th8" in tp:
    return {**base, **tp.get("th2", {})}, {**base, **tp.get("th8", {})}
  return {**base, **tp}, {**base, **tp}


def fit_fis(Xtr, vtr, edges=EDGES, tree_params=None, shoulder_scale=0.1,
            exclusive_mid=True, thr_merge_tol=0.01, reg_lambda=1.0):
  """Fit the complete chain on (Xtr, vtr) only — leakage-free when
  called inside a cross-validation fold. Returns a FISModel."""
  model = FISModel()
  medians = Xtr.median(numeric_only=True)
  Xtr = Xtr.fillna(medians)
  vtr = np.asarray(vtr, dtype=float)

  # 1. binomial trees at the normative limits
  p2, p8 = _split_params(tree_params)
  lt = {"th2": p2.pop("leaf_thr", None), "th8": p8.pop("leaf_thr", None)}
  t2 = DecisionTreeClassifier(**p2).fit(Xtr, (vtr > edges[0]).astype(int))
  t8 = DecisionTreeClassifier(**p8).fit(Xtr, (vtr > edges[1]).astype(int))

  # 2. rule extraction + threshold merging + condition simplification
  feats = list(Xtr.columns)
  rules = simplify_rules(merge_thresholds(
    build_rulebase(t2, t8, feats, leaf_thr=lt), Xtr, tol=thr_merge_tol))
  used = sorted({f for r in rules for (f, _, _) in r["conds"]})
  parts = feature_partitions(rules)

  # 3. interval membership functions per condition
  # robust span (1st-99th pct) so outliers do not widen the shoulders
  spans, bounds_by_feat = {}, {}
  for f in used:
    xv = Xtr[f].values
    spans[f] = float(np.percentile(xv, 99) - np.percentile(xv, 1)) or 1.0
    bounds_by_feat[f] = [None] + parts[f] + [None]

  def cond_fn(f, op, thr):
    """Condition membership = max over its side's interval terms."""
    thrs, bounds = parts[f], bounds_by_feat[f]
    xv = Xtr[f].values
    j = thrs.index(thr)
    idxs = range(0, j + 1) if op == "<=" else range(j + 1, len(thrs) + 1)
    terms = []
    for i in idxs:
      lo, hi = bounds[i], bounds[i + 1]
      m = np.ones(len(xv), dtype=bool)
      if lo is not None:
        m &= xv > lo
      if hi is not None:
        m &= xv <= hi
      terms.append((lo, hi, xv[m]))
    span = spans[f]

    def fn(x):
      return max(float(mf_interval(x, lo, hi, d, span, shoulder_scale))
                 for lo, hi, d in terms)
    return fn

  model._cond_fns = [[(f, cond_fn(f, op, thr)) for (f, op, thr) in r["conds"]]
                     for r in rules]
  model._levels = np.array([r["level"] for r in rules])
  model.rules = rules
  model.used = used
  model.trees = {"th2": t2, "th8": t8}
  model.medians_ = medians
  model.fallback_ = float(np.median(vtr))
  model.meta = {"edges": list(edges), "exclusive_mid": exclusive_mid,
                "shoulder_scale": shoulder_scale, "feat_names": feats,
                "n_rules": len(rules), "leaf_thr": lt}

  # 4. consequent singletons: leaf medians adjusted by constrained LS
  n, R = len(Xtr), len(rules)
  W = model.activation_matrix(Xtr)
  rs = W.sum(axis=1)
  ok = rs > 1e-9
  Wn = W[ok] / rs[ok][:, None]
  bnds = [0.0] + list(edges) + [float(vtr.max())]
  c0, lob, hib = [], [], []
  cuts = [-np.inf] + list(edges) + [np.inf]
  for lv in model._levels:
    d = vtr[(vtr > cuts[lv]) & (vtr <= cuts[lv + 1])]
    c0.append(float(np.median(d)) if len(d) else float(np.median(vtr)))
    lob.append(bnds[lv])
    hib.append(max(bnds[lv + 1], bnds[lv] + 1e-6))
  A = np.vstack([Wn, np.sqrt(reg_lambda) * np.eye(R)])
  b = np.concatenate([vtr[ok], np.sqrt(reg_lambda) * np.array(c0)])
  sol = lsq_linear(A, b, bounds=(np.array(lob), np.array(hib)))
  model.consequents = [float(c) for c in sol.x]
  return model


#%% ===============================================================
# Inner-fold selection of the tree hyperparameters
# =================================================================
TREE_GRID = [{"max_depth": d, "min_samples_leaf": msl, "ccp_alpha": ca,
              "leaf_thr": lt}
             for d in (3, 4, 5)
             for msl in (3, 5, 10)
             for ca in (0.0, 1e-3)
             for lt in (0.4, 0.5, 0.6)]


def select_tree_params(Xtr, vtr, edges=EDGES, grid=None, n_inner=4):
  """Per-threshold grid search by inner-CV mean recall + 0.5*ratio."""
  grid = grid or TREE_GRID
  Xtr = Xtr.fillna(Xtr.median(numeric_only=True))
  vtr = np.asarray(vtr, dtype=float)

  def score(th, params):
    params = dict(params)
    leaf_thr = params.pop("leaf_thr", None)
    yb = (vtr > th).astype(int)
    skf = StratifiedKFold(n_inner, shuffle=True, random_state=RANDOM_STATE)
    scores = []
    for itr, iva in skf.split(Xtr, yb):
      clf = DecisionTreeClassifier(class_weight="balanced",
                                   random_state=RANDOM_STATE, **params)
      clf.fit(Xtr.iloc[itr], yb[itr])
      proba = clf.predict_proba(Xtr.iloc[iva])[:, 1]
      yhat = (proba >= (leaf_thr if leaf_thr is not None else 0.5))
      mr, ratio, _ = recall_metrics(yb[iva], yhat.astype(int))
      scores.append(mr + 0.5 * ratio)
    return float(np.mean(scores))

  best = {}
  for th in edges:
    sc = [score(th, p) for p in grid]
    best[th] = grid[int(np.argmax(sc))]
  if best[edges[0]] == best[edges[1]]:
    return best[edges[0]]
  return {"th2": best[edges[0]], "th8": best[edges[1]]}


#%% ===============================================================
# Honest nested evaluation of the complete chain
# =================================================================
SCALE_GRID = (0.03, 0.1, 0.3, 1.0)
LAMBDA_GRID = (0.3, 1.0, 3.0)
# richer rulebase candidates for the numeric layer: the recall-selected
# trees plus deeper variants (finer partitions carry regression signal)
DEPTH_VARIANTS = ({"max_depth": 6, "min_samples_leaf": 5,
                   "ccp_alpha": 0.0, "leaf_thr": 0.5},
                  {"max_depth": 7, "min_samples_leaf": 10,
                   "ccp_alpha": 0.0, "leaf_thr": 0.5})


def select_fis_config(Xtr, vtr, tp, edges=EDGES, scale_grid=(0.03, 0.1),
                      lambda_grid=(0.3, 1.0), tree_variants=None,
                      n_inner=3):
  """Inner-CV joint selection (by MAE) of the rule-source trees, the
  transition-width factor, and the consequent regularization.
  Returns (tree_params, scale, reg_lambda)."""
  from sklearn.metrics import mean_absolute_error
  vtr = np.asarray(vtr, dtype=float)
  y3 = to_levels(vtr, edges)
  skf = StratifiedKFold(n_inner, shuffle=True, random_state=RANDOM_STATE)
  trees = [tp] + list(tree_variants if tree_variants is not None
                      else DEPTH_VARIANTS)
  combos = [(t, s, l) for t in range(len(trees))
            for s in scale_grid for l in lambda_grid]
  maes = {c: [] for c in combos}
  for itr, iva in skf.split(Xtr, y3):
    for c in combos:
      fis = fit_fis(Xtr.iloc[itr], vtr[itr], edges,
                    tree_params=trees[c[0]], shoulder_scale=c[1],
                    reg_lambda=c[2])
      maes[c].append(mean_absolute_error(vtr[iva],
                                         fis.predict(Xtr.iloc[iva])))
  best = min(combos, key=lambda c: float(np.mean(maes[c])))
  return trees[best[0]], best[1], best[2]


def select_shoulder_scale(Xtr, vtr, tp, edges=EDGES, grid=None,
                          n_inner=3):
  """Backward-compatible wrapper: returns only the scale factor."""
  return select_fis_config(Xtr, vtr, tp, edges, n_inner=n_inner)[1]


def nested_cv_fis(X, v, edges=EDGES, n_outer=5, n_inner=4, grid=None,
                  scale_grid=SCALE_GRID, verbose=True):
  """Outer StratifiedKFold on the normative class; the whole chain
  (tree and transition-width selection included) refits inside every
  outer train fold. Returns OOF predictions and per-fold metrics."""
  from sklearn.metrics import mean_absolute_error, r2_score
  v = np.asarray(v, dtype=float)
  y3 = to_levels(v, edges)
  outer = StratifiedKFold(n_outer, shuffle=True, random_state=RANDOM_STATE)
  oof = np.full(len(v), np.nan)
  folds, params_by_fold = [], []
  for k, (tr, te) in enumerate(outer.split(X, y3)):
    tp = select_tree_params(X.iloc[tr], v[tr], edges, grid, n_inner)
    tpf, sc, lam = select_fis_config(X.iloc[tr], v[tr], tp, edges)
    fis = fit_fis(X.iloc[tr], v[tr], edges, tree_params=tpf,
                  shoulder_scale=sc, reg_lambda=lam)
    pred = fis.predict(X.iloc[te])
    oof[te] = pred
    params_by_fold.append({"trees": tpf, "shoulder_scale": sc,
                           "reg_lambda": lam})
    folds.append({"fold": k, "mae": mean_absolute_error(v[te], pred),
                  "r2": r2_score(v[te], pred),
                  "n_rules": fis.meta["n_rules"]})
    if verbose:
      print(f"  fold {k}: MAE={folds[-1]['mae']:.3f} "
            f"R2={folds[-1]['r2']:.3f} rules={fis.meta['n_rules']} "
            f"scale={sc} lambda={lam}")
  mae, r2 = mean_absolute_error(v, oof), r2_score(v, oof)
  if verbose:
    print(f"  OOF: MAE={mae:.3f} R2={r2:.3f}")
  return {"oof_pred": oof, "mae": float(mae), "r2": float(r2),
          "folds": folds, "params_by_fold": params_by_fold}
