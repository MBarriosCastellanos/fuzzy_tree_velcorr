#%% ===============================================================
# diagrams.py — box/arrow drawing helpers for schematic figures
# -----------------------------------------------------------------
# Text-fitted rounded boxes, diamonds and L-shaped connectors used
# by the tree diagrams and the nested-CV schematic. Ported from the
# project's model-architecture figures and recolored to the
# red/teal palette of style.py.
# =================================================================
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon

from style import LEVEL_EDGE, LEVEL_FILL, NEUTRAL_EDGE, NEUTRAL_FILL


def _measure(ax, text, fs):
  """Text size in data coordinates (to fit boxes to their label)."""
  t = ax.text(0, 0, text, ha="center", va="center", fontsize=fs)
  ax.figure.canvas.draw()
  bb = t.get_window_extent(renderer=ax.figure.canvas.get_renderer())
  t.remove()
  (x0, y0), (x1, y1) = ax.transData.inverted().transform(
    [(bb.x0, bb.y0), (bb.x1, bb.y1)])
  return abs(x1 - x0), abs(y1 - y0)


def autobox(ax, cx, text, fc=NEUTRAL_FILL, ec=NEUTRAL_EDGE, fs=11,
            cy=None, top=None, padx=0.02, pady=0.02):
  """Rounded box fitted to its text; anchored by center or top edge."""
  w, h = _measure(ax, text, fs)
  w += 2 * padx
  h += 2 * pady
  if top is not None:
    cy = top - h / 2
  ax.add_patch(FancyBboxPatch(
    (cx - w / 2, cy - h / 2), w, h,
    boxstyle="round,pad=0.004,rounding_size=0.02", fc=fc, ec=ec, lw=1.6,
    zorder=2))
  ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, zorder=3)
  return {"cx": cx, "cy": cy, "w": w, "h": h}


def autodiamond(ax, cx, cy, text, fs=11, fc=(1, 0.8, 0.8), ec=(0.8, 0, 0),
                padx=0.03, pady=0.03):
  """Decision diamond fitted to its text."""
  tw, th = _measure(ax, text, fs)
  w, h = 2 * tw + 2 * padx, 2 * th + 2 * pady
  ax.add_patch(Polygon(
    [(cx, cy + h / 2), (cx + w / 2, cy), (cx, cy - h / 2),
     (cx - w / 2, cy)], closed=True, fc=fc, ec=ec, lw=1.6, zorder=2))
  ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, zorder=3)
  return {"cx": cx, "cy": cy, "w": w, "h": h}


def pt(node, side):
  """Point on a node border: top / bottom / left / right."""
  cx, cy, w, h = node["cx"], node["cy"], node["w"], node["h"]
  return {"top": (cx, cy + h / 2), "bottom": (cx, cy - h / 2),
          "left": (cx - w / 2, cy), "right": (cx + w / 2, cy)}[side]


def arrow(ax, x0, y0, x1, y1, color="#444444"):
  ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                               mutation_scale=16, color=color, lw=1.6))


def connect(ax, a, b, sa="bottom", sb="top", color="#444444"):
  (x0, y0), (x1, y1) = pt(a, sa), pt(b, sb)
  arrow(ax, x0, y0, x1, y1, color)


def elbow(ax, x0, y0, xm, y1, x1, color="#666666"):
  """L-shaped connector (x0,y0)->(xm,y0)->(xm,y1)->(x1,y1)."""
  ax.plot([x0, xm, xm, x1], [y0, y0, y1, y1], color=color, lw=1.3,
          solid_capstyle="round", zorder=1)


def glossary(fig, pairs, fs=8.5, per=4):
  """'symbol = meaning' footer, `per` entries per line."""
  lines = ["      ".join(pairs[i:i + per]) for i in range(0, len(pairs), per)]
  fig.text(0.5, 0.005, "\n".join(lines), ha="center", va="bottom",
           fontsize=fs, color="#555555")


#%% ===============================================================
# Horizontal decision-tree rendering (pruned, class-colored)
# =================================================================
def draw_tree(root, feat_names, mt, positive_label, negative_label,
              pos_level, neg_level, figsize=(11, 6), fs=10):
  """Custom horizontal tree diagram: text-fitted boxes, elbow
  connectors labeled True/False, fills by severity class.

  root: collapsed structure from fis_core.collapse_tree (so the
  diagram matches the extracted rulebase one-to-one).
  mt: variable -> mathtext symbol. pos/neg_level: severity class of
  each leaf label (colors). Returns the matplotlib figure.
  """
  import copy
  root = copy.deepcopy(root)

  def name_feats(node):
    if not node["leaf"]:
      node["feat"] = feat_names[node["feature"]]
      name_feats(node["L"])
      name_feats(node["R"])

  name_feats(root)
  leaf_count = [0]

  def layout(node, depth):
    node["depth"] = depth
    if node["leaf"]:
      node["y"] = leaf_count[0]
      leaf_count[0] += 1
    else:
      node["y"] = (layout(node["L"], depth + 1)
                   + layout(node["R"], depth + 1)) / 2.0
    return node["y"]

  layout(root, 0)

  def max_depth(node):
    return (node["depth"] if node["leaf"]
            else max(max_depth(node["L"]), max_depth(node["R"])))

  md, nl = max(max_depth(root), 1), max(leaf_count[0] - 1, 1)

  def X(d):
    return 0.08 + d * (0.82 / md)

  def Y(y):
    return 0.92 - y * (0.84 / nl)

  fig, ax = plt.subplots(figsize=figsize)
  ax.axis("off")
  ax.set_xlim(0, 1)
  ax.set_ylim(0, 1)

  def place(node):
    cx, cy = X(node["depth"]), Y(node["y"])
    if node["leaf"]:
      lv = pos_level if node["cls"] == 1 else neg_level
      lab = positive_label if node["cls"] == 1 else negative_label
      node["box"] = autobox(ax, cx, f"{lab}\nn = {node['n']}",
                            LEVEL_FILL[lv], LEVEL_EDGE[lv], fs=fs, cy=cy,
                            padx=0.012, pady=0.012)
    else:
      txt = r"$%s \leq %.3g$" % (mt(node["feat"]).strip("$"), node["thr"])
      node["box"] = autobox(ax, cx, txt, NEUTRAL_FILL, NEUTRAL_EDGE,
                            fs=fs, cy=cy, padx=0.012, pady=0.012)
      place(node["L"])
      place(node["R"])

  def link(node):
    if node["leaf"]:
      return
    prx, pry = pt(node["box"], "right")
    for child, lab in [(node["L"], "True"), (node["R"], "False")]:
      clx, cly = pt(child["box"], "left")
      xm = (prx + clx) / 2.0
      elbow(ax, prx, pry, xm, cly, clx)
      ax.text(xm, (pry + cly) / 2.0, lab, fontsize=fs - 2, ha="center",
              va="center", color="#333333", zorder=4,
              bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none"))
      link(child)

  place(root)
  link(root)
  return fig
