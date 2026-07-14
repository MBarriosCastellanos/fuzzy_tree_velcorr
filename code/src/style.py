#%% ===============================================================
# style.py — figure style shared by every plot of the paper
# -----------------------------------------------------------------
# All colors derive from the discrete red/teal scale of the
# project's correlation-matrix colormap: teal tones for the low
# class, light red for mid, dark red for high; gray for neutral
# elements. Figures carry no titles (captions live in the paper).
# =================================================================
import os

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# discrete correlation-matrix scale (source palette)
REDS = [(1, 0.8, 0.8), (1, 0.5, 0.5), (1, 0, 0), (0.8, 0, 0), (0.6, 0, 0)]
TEALS = [(0.8, 1, 1), (0.4, 1, 1), (0, 0.7, 0.7), (0, 0.5, 0.5),
         (0, 0.3, 0.3)]

# severity classes: low / mid / high
LEVEL_EDGE = {0: (0, 0.5, 0.5), 1: (1, 0, 0), 2: (0.6, 0, 0)}
LEVEL_FILL = {0: (0.8, 1, 1), 1: (1, 0.8, 0.8), 2: (1, 0.5, 0.5)}
LEVEL_NAME = {0: "low", 1: "mid", 2: "high"}

# the four compared methods
METHOD_COLOR = {"Proposed FIS": (0.6, 0, 0),
                "Linear regression": (0, 0.7, 0.7),
                "Decision tree": (0, 0.5, 0.5),
                "SVR": (0, 0.3, 0.3)}

NEUTRAL_FILL, NEUTRAL_EDGE = "#f4f4f4", "#555555"
GRAY = "#666666"


def corr_cmap():
  """Discrete red/teal colormap (as in the correlation matrix)."""
  return ListedColormap(REDS[::-1] + TEALS)


def heat_cmap():
  """Sequential teal->red scale for 0..1 heatmaps (confusion matrix)."""
  return ListedColormap([(0.9, 1, 1), (0.8, 1, 1), (0.4, 1, 1),
                         (1, 0.8, 0.8), (1, 0.5, 0.5), (1, 0, 0),
                         (0.8, 0, 0), (0.6, 0, 0)])


def save_fig(fig_dir, name, fig=None):
  """Save the current (or given) figure as PDF + PNG, no title."""
  fig = fig or plt.gcf()
  os.makedirs(fig_dir, exist_ok=True)
  for ext in ("pdf", "png"):
    fig.savefig(os.path.join(fig_dir, f"{name}.{ext}"), dpi=200,
                bbox_inches="tight")
  plt.close(fig)
