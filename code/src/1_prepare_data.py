#%% ===============================================================
# 1_prepare_data.py — field data preparation
# -----------------------------------------------------------------
# Loads the raw internal-corrosion dataset, applies the cleaning
# rules, saturates the regression target at 20 mpy (2.5 times the
# highest normative limit) and renames every column to the physics
# nomenclature. Writes data/processed.csv plus the descriptive
# statistics used by the paper's data section.
# =================================================================
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from nomenclature import NOMENCLATURE, RENAME, TARGET
from fis_core import EDGES, to_levels

DATA_DIR = os.path.join(CODE_DIR, "data")
RESULTS_DIR = os.path.join(CODE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

RAW_CSV = os.path.join(DATA_DIR, "int_canolimon.csv")
SATURATION = 20.0                     # [mpy]

# columns outside the modeling scope:
# - velcorrcal_cmpy reproduces the target exactly for 76.5% of the
#   records (97.8% of the failure records): the reported rate was
#   computed with the same thickness-loss formula -> leakage;
# - desgaste / vida remanente / espesor minimo derive from the same
#   inspection measurement that defines the target;
# - fault_xx labels the record itself as a failure event, an outcome
#   unknown at prediction time.
# Service time (anos_y) STAYS: the installation date is known a
# priori, before any inspection (expert decision).
EXCLUDE = ["velcorrcal_cmpy", "fault_xx", "desgaste_xx",
           "vidaremanenteactual_yr", "vidaremanenteinstalacion_yr",
           "espesormin_mm"]

#%% ===============================================================
# Load, clean, saturate, rename
# =================================================================
df = pd.read_csv(RAW_CSV, encoding="utf-8-sig")
n_raw = len(df)

df = df.drop(columns=[c for c in EXCLUDE if c in df.columns])
neg = df["velcorr_mpy"] < 0
df = df.loc[~neg].reset_index(drop=True)          # invalid measurements
n_sat = int((df["velcorr_mpy"] > SATURATION).sum())
df["velcorr_mpy"] = df["velcorr_mpy"].clip(upper=SATURATION)
df = df.replace([np.inf, -np.inf], np.nan)
df = df.rename(columns=RENAME)

df.to_csv(os.path.join(DATA_DIR, "processed.csv"), index=False)
print(f"raw rows: {n_raw}  kept: {len(df)}  (dropped {int(neg.sum())} "
      f"negative-rate records)")
print(f"target saturated at {SATURATION} mpy: {n_sat} records "
      f"({100 * n_sat / len(df):.1f}%)")
print(f"features: {df.shape[1] - 1}")

#%% ===============================================================
# Class distribution and descriptive statistics
# =================================================================
y3 = to_levels(df[TARGET].values)
counts = np.bincount(y3, minlength=3)
dist = pd.DataFrame([{
  "n": len(df),
  f"low (0-{EDGES[0]:g}]": counts[0],
  f"mid ({EDGES[0]:g}-{EDGES[1]:g}]": counts[1],
  f"high (>{EDGES[1]:g})": counts[2]}])
dist.to_csv(os.path.join(RESULTS_DIR, "class_distribution.csv"), index=False)
print(dist.to_string(index=False))

rows = []
for raw, (col, mtx, latex, meaning, unit) in NOMENCLATURE.items():
  if col not in df.columns:
    continue
  s = df[col].dropna()
  rows.append({"symbol": col, "latex": latex, "meaning": meaning,
               "unit": unit, "n_missing": int(df[col].isna().sum()),
               "min": s.min(), "median": s.median(), "mean": s.mean(),
               "max": s.max()})
stats = pd.DataFrame(rows)
stats.to_csv(os.path.join(RESULTS_DIR, "feature_stats.csv"), index=False)
print(f"\ndescriptive statistics saved for {len(stats)} variables")
