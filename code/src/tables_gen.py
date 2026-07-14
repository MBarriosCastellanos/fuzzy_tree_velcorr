#%% ===============================================================
# tables_gen.py — LaTeX tables of the paper
# -----------------------------------------------------------------
# Strict booktabs structure for every table (one \toprule, \midrule
# after the header, \bottomrule; extra \midrule only as group
# separator) and width-safe layouts: the wide tables use tabularx
# with an X column so they can never exceed the text width.
# =================================================================
import os

import pandas as pd

from fis_core import LEVELS

BS = "\\"


def write_table(body, caption, label, path, size=r"\footnotesize",
                star=False):
  env = "table*" if star else "table"
  tex = (f"{BS}begin{{{env}}}[htbp]\n{BS}centering\n{size}\n"
         f"{BS}setlength{{{BS}tabcolsep}}{{4pt}}\n"
         f"{BS}caption{{{caption}}}\n{BS}label{{{label}}}\n{body}"
         f"{BS}end{{{env}}}\n")
  with open(path, "w", encoding="utf-8") as fp:
    fp.write(tex)
  print(f"  -> {os.path.basename(path)}")


def make_all(results_dir, tab_dir, comp):
  # 1. variables: meaning, units, descriptive statistics (tabularx)
  stats = pd.read_csv(os.path.join(results_dir, "feature_stats.csv"))
  lines = [r"\begin{tabularx}{\textwidth}{@{}lXrrrrr@{}}", r"\toprule",
           r"Symbol & Meaning [unit] & Min & Median & Mean & Max & "
           r"Missing \\", r"\midrule"]
  for _, r in stats.iterrows():
    lines.append(
      f"{r['latex']} & {r['meaning']} [{r['unit']}] & {r['min']:.3g} & "
      f"{r['median']:.3g} & {r['mean']:.3g} & {r['max']:.3g} & "
      f"{int(r['n_missing'])} " + r"\\")
  lines += [r"\bottomrule", r"\end{tabularx}"]
  write_table("\n".join(lines) + "\n",
              "Variables of the dataset: physical meaning, units and "
              "descriptive statistics (897 records; the measured rate "
              "$v$ is reported after saturation at 20 [mpy]).",
              "tab:features", os.path.join(tab_dir, "feature_stats.tex"),
              star=True)

  # 2. class distribution
  dist = pd.read_csv(os.path.join(results_dir, "class_distribution.csv"))
  lines = [r"\begin{tabular}{@{}cccc@{}}", r"\toprule",
           r"$n$ & low $(0,2]$ & mid $(2,8]$ & high $(>8)$ \\",
           r"\midrule",
           f"{int(dist.iloc[0, 0])} & {int(dist.iloc[0, 1])} & "
           f"{int(dist.iloc[0, 2])} & {int(dist.iloc[0, 3])} " + r"\\",
           r"\bottomrule", r"\end{tabular}"]
  write_table("\n".join(lines) + "\n",
              "Distribution of the records across the normative "
              "severity classes [mpy].", "tab:classes",
              os.path.join(tab_dir, "class_distribution.tex"))

  # 3. tree metrics
  tm = pd.read_csv(os.path.join(results_dir, "tree_metrics.csv"))
  lines = [r"\begin{tabular}{@{}cccccc@{}}", r"\toprule",
           r"Limit & Mean & Ratio & Recall$_{\leq}$ & Recall$_{>}$ & "
           r"Acc. \\", r"\midrule"]
  for _, r in tm.iterrows():
    lines.append(f"{r['limit_mpy']:.0f} [mpy] & {r['mean_recall']:.3f} & "
                 f"{r['ratio_recall']:.3f} & {r['recall_below']:.3f} & "
                 f"{r['recall_above']:.3f} & {r['accuracy']:.3f} " + r"\\")
  lines += [r"\bottomrule", r"\end{tabular}"]
  write_table("\n".join(lines) + "\n",
              "Binomial decision trees at the normative limits: "
              "out-of-fold metrics of the nested cross-validation "
              "(Mean: mean per-class recall; Ratio: min/max recall).",
              "tab:trees", os.path.join(tab_dir, "tree_metrics.tex"))

  # 4. compact rulebase grouped by consequent (tabularx)
  rules = pd.read_csv(os.path.join(results_dir, "rules.csv"))
  lines = [r"\begin{tabularx}{\textwidth}{@{}lXrr@{}}", r"\toprule",
           r"Rule & Antecedent & $c_r$ [mpy] & Contrib. [\%] \\",
           r"\midrule"]
  for gi, lv in enumerate(LEVELS):
    g = rules[rules.level == lv]
    if gi:
      lines.append(r"\midrule")
    note = (r" (each rule also carries the exclusion $\neg S$ of the "
            "severe region)" if lv == "mid" else "")
    lines.append(r"\multicolumn{4}{@{}l}{\emph{Consequent: " + lv
                 + note + r"}} \\")
    for _, r in g.iterrows():
      lines.append(f"{r['rule']} & ${r['antecedent_latex']}$ & "
                   f"{r['consequent_mpy']:.2f} & "
                   f"{r['zone_contrib_pct']:.1f} " + r"\\")
  lines += [r"\bottomrule", r"\end{tabularx}"]
  write_table("\n".join(lines) + "\n",
              "Deployed rulebase: antecedents extracted from the "
              "binomial trees (physics nomenclature), tuned consequent "
              "singletons $c_r$ and contribution share of each rule to "
              "the output within its own severity zone.",
              "tab:rules", os.path.join(tab_dir, "rules_compact.tex"),
              star=True)

  # 5. four-method comparison (compact, single column)
  lines = [r"\begin{tabular}{@{}lccc@{}}", r"\toprule",
           r"Method & MAE [mpy] & $R^2$ & Recall$_{high}$ \\",
           r"\midrule"]
  for _, r in comp.iterrows():
    name = (r["method"].replace("Decision tree", "Decision-tree reg.")
            .replace("Linear regression", "Linear reg."))
    lines.append(
      f"{name} & {r['mae_oof']:.2f} ({r['mae_min']:.2f}--"
      f"{r['mae_max']:.2f}) & {r['r2_oof']:.3f} ({r['r2_min']:.2f}--"
      f"{r['r2_max']:.2f}) & {r['recall_high']:.2f} " + r"\\")
  lines += [r"\bottomrule", r"\end{tabular}"]
  write_table("\n".join(lines) + "\n",
              "Out-of-fold comparison on identical folds: global value "
              "(min--max across the five outer folds).",
              "tab:comparison", os.path.join(tab_dir, "comparison.tex"))
  print("tables saved")
