#%% ===============================================================
# nomenclature.py — physics nomenclature of the field variables
# -----------------------------------------------------------------
# Single source of truth: raw CSV column -> (short symbol used in
# code/dataframes, mathtext symbol for figures, LaTeX symbol for
# tables, physical meaning, unit). Columns are renamed at load time
# so every rule, figure and table speaks physics, never code names.
# =================================================================

TARGET_RAW = "velcorr_mpy"
TARGET = "v"

# raw column: (column, mathtext, latex, meaning, unit)
NOMENCLATURE = {
  "velcorr_mpy":      ("v",     r"$v$",                  r"$v$",
                       "measured corrosion rate", "mpy"),
  "anos_y":           ("t_s",   r"$t_s$",                r"$t_s$",
                       "service time of the monitoring point", "yr"),
  "pco2_psig":        ("pCO2",  r"$p_{CO_2}$",           r"$p_{\mathrm{CO_2}}$",
                       "CO$_2$ partial pressure", "psig"),
  "co2_mol":          ("yCO2",  r"$y_{CO_2}$",           r"$y_{\mathrm{CO_2}}$",
                       "CO$_2$ gas mole fraction", "--"),
  "metano_mol":       ("yCH4",  r"$y_{CH_4}$",           r"$y_{\mathrm{CH_4}}$",
                       "methane gas mole fraction", "--"),
  "n2_mol":           ("yN2",   r"$y_{N_2}$",            r"$y_{\mathrm{N_2}}$",
                       "nitrogen gas mole fraction", "--"),
  "velocidad_fts":    ("u",     r"$u$",                  r"$u$",
                       "production fluid velocity", "ft/s"),
  "presion_psig":     ("P",     r"$P$",                  r"$P$",
                       "operating pressure", "psig"),
  "temperatura_f":    ("T",     r"$T$",                  r"$T$",
                       "operating temperature", "$^{\\circ}$F"),
  "ph_xx":            ("pH",    r"$pH$",                 r"pH",
                       "produced-water pH", "--"),
  "conductividad_uscm": ("sigma", r"$\sigma$",           r"$\sigma$",
                       "water electrical conductivity", "$\\mu$S/cm"),
  "cl_ppm":           ("Cl",    r"$[Cl^-]$",             r"$[\mathrm{Cl^-}]$",
                       "chloride concentration", "ppm"),
  "na_ppm":           ("Na",    r"$[Na^+]$",             r"$[\mathrm{Na^+}]$",
                       "sodium concentration", "ppm"),
  "ca_ppm":           ("Ca",    r"$[Ca^{2+}]$",          r"$[\mathrm{Ca^{2+}}]$",
                       "calcium concentration", "ppm"),
  "mg_ppm":           ("Mg",    r"$[Mg^{2+}]$",          r"$[\mathrm{Mg^{2+}}]$",
                       "magnesium concentration", "ppm"),
  "ba_ppm":           ("Ba",    r"$[Ba^{2+}]$",          r"$[\mathrm{Ba^{2+}}]$",
                       "barium concentration", "ppm"),
  "sr_ppm":           ("Sr",    r"$[Sr^{2+}]$",          r"$[\mathrm{Sr^{2+}}]$",
                       "strontium concentration", "ppm"),
  "fe_total":         ("Fe",    r"$[Fe]$",               r"$[\mathrm{Fe}]$",
                       "total dissolved iron", "ppm"),
  "so4_ppm":          ("SO4",   r"$[SO_4^{2-}]$",        r"$[\mathrm{SO_4^{2-}}]$",
                       "sulfate concentration", "ppm"),
  "hco3_megl":        ("HCO3",  r"$[HCO_3^-]$",          r"$[\mathrm{HCO_3^-}]$",
                       "bicarbonate concentration", "meq/L"),
  "alcalinidad_ppm":  ("Alk",   r"$Alk$",                r"Alk",
                       "total alkalinity", "ppm CaCO$_3$"),
  "durezatotal_ppm":  ("TH",    r"$TH$",                 r"TH",
                       "total hardness", "ppm CaCO$_3$"),
  "sst_ppm":          ("TSS",   r"$TSS$",                r"TSS",
                       "total suspended solids", "ppm"),
  "sdt_ppm":          ("TDS",   r"$TDS$",                r"TDS",
                       "total dissolved solids", "ppm"),
  "agua_bwpd":        ("Qw",    r"$Q_w$",                r"$Q_w$",
                       "water production rate", "BWPD"),
  "crudo_bopd":       ("Qo",    r"$Q_o$",                r"$Q_o$",
                       "oil production rate", "BOPD"),
  "flujogas_mmscfd":  ("Qg",    r"$Q_g$",                r"$Q_g$",
                       "gas production rate", "MMSCFD"),
  "patronflujo_xx":   ("FP",    r"$FP$",                 r"FP",
                       "flow-pattern class", "--"),
  "fault_xx":         ("NF",    r"$N_F$",                r"$N_F$",
                       "recorded failure indicator", "--"),
  "diametronom_in":   ("D",     r"$D$",                  r"$D$",
                       "nominal pipe diameter", "in"),
  "diametroint_in":   ("Di",    r"$D_i$",                r"$D_i$",
                       "internal pipe diameter", "in"),
  "diametroext_in":   ("De",    r"$D_e$",                r"$D_e$",
                       "external pipe diameter", "in"),
  "espesor_mm":       ("e",     r"$e$",                  r"$e$",
                       "wall thickness", "mm"),
}

RENAME = {raw: props[0] for raw, props in NOMENCLATURE.items()}
MATHTEXT = {props[0]: props[1] for props in NOMENCLATURE.values()}
LATEX = {props[0]: props[2] for props in NOMENCLATURE.values()}
MEANING = {props[0]: props[3] for props in NOMENCLATURE.values()}
UNIT = {props[0]: props[4] for props in NOMENCLATURE.values()}


def mt(name):
  """Mathtext symbol of a variable (falls back to the plain name)."""
  return MATHTEXT.get(name, name)


def lx(name):
  """LaTeX symbol of a variable (falls back to the plain name)."""
  return LATEX.get(name, name)
