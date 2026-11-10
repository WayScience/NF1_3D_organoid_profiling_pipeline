#!/usr/bin/env python
# coding: utf-8

# # 3D Segmentation by Sliding-Window Max Projection
#
# A polygon-based redraw of the 2.5D segmentation pipeline figure. Every shape (tiles, slices, cells, spheres, arrows, badges, funnels) is a matplotlib `Polygon` or `RegularPolygon`.
#
# The figure reads as eight numbered stages: input stack → original slices → window max projection → 2D segmentation → decoupling → collapse into a per-slice consensus → linking across z → 3D masks.
#
# Run all cells; the last one displays the figure and saves `segmentation_pipeline.png` / `.svg`.

# In[1]:


import pathlib

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from image_analysis_3D.file_utils.notebook_init_utils import init_notebook
from matplotlib.colors import to_rgb
from matplotlib.patches import Polygon, RegularPolygon

root_dir, in_notebook = init_notebook()

# In[2]:


segmentation_pipeline_figure_file_path = pathlib.Path(
    f"{root_dir}/figures/nuclei_segmentation_figure/figures/segmentation_pipeline.png"
)
segmentation_pipeline_figure_file_path.parent.mkdir(parents=True, exist_ok=True)


# ## Palette
# - **Grey** – raw, unlabelled signal
# - **Blue / orange / pink / teal** – object labels
# - **Violet / green / rust** – the three sliding windows (z0–2, z1–3, z2–4); kept distinct from the object colours
# - **Gold** – consensus and 3D output
#
# `OFF` holds the four dot positions, reused on every tile and slice so the same object always sits in the same place.

# In[3]:


# ----------------------------------------------------------------- palette --
X0, W, Y0, Y1 = 1.85, 22, 1.95, 17.1  # canvas: x 0..W, y Y0..Y1
RAW = "#8A94A6"  # raw, unlabelled signal
BLUE, ORANGE, TEAL, PINK = "#3D6BFF", "#FF9A2E", "#1FD8AC", "#F0399B"
OBJ = [BLUE, ORANGE, PINK, TEAL]  # object labels 1-4
TEXT, MUTED = "#1A2238", "#5A6B8C"
SLATE = "#5A6B8C"  # input / original slices
WIN_COLORS = ["#7C4DFF", "#3E9B47", "#B5533C"]  # windows z0-2 / z1-3 / z2-4
WIN_NAMES = ["z0–2", "z1–3", "z2–4"]
GOLD = "#E09A00"  # consensus + 3D output
STRIPE = "#F1F4F9"
TILE = 1.4
# the same four dot positions are used on every tile and slice
OFF = [(-0.32, 0.06), (0.32, 0.08), (0.34, -0.38), (-0.28, -0.36)]
WHITE_EDGE = [pe.withStroke(linewidth=4, foreground="white")]


def mix(c1, c2, t):
    return tuple(np.array(to_rgb(c1)) * (1 - t) + np.array(to_rgb(c2)) * t)


# ## Polygon helpers
# - `rect` – square tiles and panels
# - `gradient_fill` – soft gradient from stacked, shrinking polygons
# - `outline` / `line` – borders and connectors
# - `gem` – hexagonal cell
# - `tile`, `iso_slice` – image tiles and isometric slices
# - `collapse_funnel` – step F: each slice's copies stream into a gold funnel and collapse (max) into one consensus tile
# - `sphere` – shaded 3D ball
# - `arrow_h`, `tri_down`, `badge`, `label` – arrows, stage letters (A–H) and text

# In[4]:


# --------------------------------------------------------- polygon helpers --
def rect(cx, cy, w, h):
    return [
        (cx - w / 2, cy - h / 2),
        (cx + w / 2, cy - h / 2),
        (cx + w / 2, cy + h / 2),
        (cx - w / 2, cy + h / 2),
    ]


def scaled(pts, cx, cy, k):
    return [(cx + (x - cx) * k, cy + (y - cy) * k) for x, y in pts]


def gradient_fill(ax, pts, cx, cy, c_out, c_in, z, n=10):
    """Soft gradient from stacked, shrinking copies of the polygon."""
    for i in range(n):
        t = i / (n - 1)
        ax.add_patch(
            Polygon(
                scaled(pts, cx, cy, 1 - 0.85 * t),
                closed=True,
                facecolor=mix(c_out, c_in, t),
                edgecolor="none",
                zorder=z + i * 0.001,
            )
        )


def outline(ax, pts, color, z, lw=2.2, **kw):
    ax.add_patch(
        Polygon(
            pts,
            closed=True,
            fill=False,
            edgecolor=color,
            lw=lw,
            zorder=z,
            joinstyle="miter",
            **kw,
        )
    )


def line(ax, xs, ys, color, z=3, lw=2.0, alpha=1.0):
    ax.plot(xs, ys, color=color, lw=lw, alpha=alpha, solid_capstyle="round", zorder=z)


def gem(ax, x, y, color, r=0.13, z=10):
    """Hexagonal cell with a soft halo and a specular highlight."""
    for rr, a in [(1.6, 0.12), (1.3, 0.18)]:
        ax.add_patch(
            RegularPolygon(
                (x, y),
                6,
                radius=r * rr,
                facecolor=color,
                edgecolor="none",
                alpha=a,
                zorder=z,
            )
        )
    ax.add_patch(
        RegularPolygon(
            (x, y),
            6,
            radius=r,
            facecolor=color,
            edgecolor="white",
            lw=1,
            zorder=z + 0.1,
        )
    )
    ax.add_patch(
        RegularPolygon(
            (x - r * 0.28, y + r * 0.28),
            6,
            radius=r * 0.38,
            facecolor="white",
            alpha=0.55,
            edgecolor="none",
            zorder=z + 0.2,
        )
    )


def tile(ax, cx, cy, text, colors, edge, z=5, lw=2.2):
    pts = rect(cx, cy, TILE, TILE)
    gradient_fill(ax, pts, cx, cy, "#E3EAF5", "#FFFFFF", z)
    outline(ax, pts, edge, z + 0.5, lw=lw)
    for (dx, dy), c in zip(OFF, colors):
        gem(ax, cx + dx, cy + dy, c, z=z + 1)
    ax.text(
        cx,
        cy + 0.46,
        text,
        ha="center",
        va="center",
        fontsize=18.5,
        color=TEXT,
        weight="bold",
        zorder=z + 2,
        path_effects=WHITE_EDGE,
    )


def iso_slice(ax, cx, cy, s, colors, edge, z):
    """Flattened isometric slice; dots placed with the same OFF layout as tiles."""
    k = 0.55
    pts = [(cx, cy + s * k), (cx + s, cy), (cx, cy - s * k), (cx - s, cy)]
    gradient_fill(ax, pts, cx, cy, "#E3EAF5", "#FFFFFF", z, n=6)
    outline(ax, pts, edge, z + 0.1, lw=1.4)
    for (dx, dy), c in zip(OFF, colors):
        u, v = dx / TILE, dy / TILE
        gem(ax, cx + (u + v) * s, cy + (v - u) * s * k, c, r=0.065, z=z + 0.2)


def sphere(ax, x, y, r, color, z=10, n=40):
    """Shaded 3D ball from stacked 64-gons drifting toward a highlight."""
    dark = mix(color, "black", 0.45)
    for i in range(n):
        t = i / (n - 1)
        c = (
            mix(dark, color, t / 0.65)
            if t < 0.65
            else mix(color, "white", (t - 0.65) / 0.35)
        )
        ax.add_patch(
            RegularPolygon(
                (x - 0.32 * r * t, y + 0.32 * r * t),
                64,
                radius=r * (1 - 0.92 * t),
                facecolor=c,
                edgecolor="none",
                zorder=z + i * 0.001,
            )
        )


def arrow_h(ax, x0, x1, y, color, w=0.14, hl=0.28, hw=0.36, z=6):
    ax.add_patch(
        Polygon(
            [
                (x0, y - w / 2),
                (x1 - hl, y - w / 2),
                (x1 - hl, y - hw / 2),
                (x1, y),
                (x1 - hl, y + hw / 2),
                (x1 - hl, y + w / 2),
                (x0, y + w / 2),
            ],
            facecolor=color,
            edgecolor="none",
            zorder=z,
        )
    )


def tri_down(ax, x, y, color, s=0.12, z=6):
    ax.add_patch(
        Polygon(
            [(x - s, y + s * 0.6), (x + s, y + s * 0.6), (x, y - s * 0.6)],
            facecolor=color,
            edgecolor="none",
            zorder=z,
        )
    )


def badge(ax, x, y, n, color=TEXT, r=0.37):
    ax.add_patch(
        RegularPolygon(
            (x, y), 64, radius=r, facecolor=color, edgecolor="white", lw=1.5, zorder=31
        )
    )
    ax.text(
        x,
        y - 0.01,
        str(n),
        ha="center",
        va="center",
        fontsize=21,
        color="white",
        weight="bold",
        zorder=32,
    )


def collapse_funnel(ax, x, y_top, y_bot, windows):
    """Step 6: the copies stacked in one slice column collapse into one tile."""
    wt, wb = TILE * 0.5, 0.2  # half-widths top / bottom
    n = 12
    for i in range(n):  # gold funnel, darker as it narrows
        t0, t1 = i / n, (i + 1) / n
        ya, yb = y_top + (y_bot - y_top) * t0, y_top + (y_bot - y_top) * t1
        ha, hb = wt + (wb - wt) * t0, wt + (wb - wt) * t1
        ax.add_patch(
            Polygon(
                [(x - ha, ya), (x + ha, ya), (x + hb, yb), (x - hb, yb)],
                facecolor=GOLD,
                alpha=0.08 + 0.22 * t1,
                edgecolor="none",
                zorder=3,
            )
        )
    k = len(windows)
    for j, w in enumerate(windows):  # one stream per contributing copy
        o = (j - (k - 1) / 2) * 0.42
        bw, tw = 0.11, 0.035
        ax.add_patch(
            Polygon(
                [
                    (x + o - bw, y_top),
                    (x + o + bw, y_top),
                    (x + tw, y_bot),
                    (x - tw, y_bot),
                ],
                facecolor=WIN_COLORS[w],
                alpha=0.85,
                edgecolor="none",
                zorder=4,
            )
        )
    yp = y_bot + 0.33
    ax.add_patch(
        Polygon(
            rect(x, yp, 0.9, 0.4), facecolor=GOLD, edgecolor="white", lw=1.5, zorder=7
        )
    )
    ax.text(
        x,
        yp - 0.01,
        "max",
        ha="center",
        va="center",
        fontsize=18,
        color="white",
        weight="bold",
        zorder=8,
    )
    tri_down(ax, x, y_bot - 0.02, GOLD, s=0.13, z=5)


def label(ax, x, y, s, ha="left", size=25, color=TEXT, **kw):
    ax.text(
        x,
        y,
        s,
        ha=ha,
        va="center",
        fontsize=size,
        color=color,
        zorder=30,
        linespacing=1.2,
        path_effects=WHITE_EDGE,
        **kw,
    )


# ## Build and save the figure
# Layout coordinates are at the top of this cell (`cols`, `y_orig`, `rows_dec`, …) if you want to adjust spacing.

# In[5]:


# ------------------------------------------------------------------ canvas --
fig = plt.figure(figsize=(W - X0, Y1 - Y0), dpi=100)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(X0, W)
ax.set_ylim(Y0, Y1)
ax.axis("off")
fig.patch.set_facecolor("white")

ax.text(
    (X0 + W) / 2,
    16.6,
    "3D Segmentation by Sliding-Window Max Projection",
    ha="center",
    va="center",
    fontsize=44,
    weight="bold",
    color=TEXT,
    zorder=30,
)

cols = [7.5, 9.1, 10.7, 12.3, 13.9]
y_orig, y_proj, y_seg, y_cons = 15.0, 12.85, 11.15, 2.85
rows_dec = [8.92, 7.37, 5.82]  # one sub-row per window
dec_top, dec_bot = 9.7, 5.02
win_cols = [0, 2, 4]  # where each window tile sits
x_label, x_badge = 2.88, 2.4

# row stripes so each label clearly owns its row
for lo, hi in [
    (14.22, 15.78),
    (12.07, 13.63),
    (10.41, 11.93),
    (4.98, 10.35),
    (2.05, 4.92),
]:
    ax.add_patch(
        Polygon(
            [(1.95, lo), (14.75, lo), (14.75, hi), (1.95, hi)],
            facecolor=STRIPE,
            edgecolor="none",
            zorder=1,
        )
    )

# ------------------------------------------------ stage A: input z-stack --
xs = 5.7
for i in range(5):  # z0 at the bottom
    iso_slice(ax, xs, 14.52 + i * 0.24, 0.62, [RAW] * 4, SLATE, z=5 + i)
badge(ax, 4.3, 15.95, "A")
label(ax, 4.75, 15.95, "Input z-stack", size=21, weight="bold")
arrow_h(ax, 6.42, 6.75, y_orig, SLATE)

# ------------------------------------------------ stage B: original slices --
for i, x in enumerate(cols):
    label(ax, x, 15.95, f"z{i}", ha="center", size=23, color=MUTED, weight="bold")
    tile(ax, x, y_orig, f"z{i}", [RAW] * 4, SLATE)

# ------------------------------------------- stage C: window max projection --
for w, col in enumerate(win_cols):
    for s in range(w, w + 3):
        line(ax, [cols[s], cols[col]], [y_orig - 0.71, y_proj + 0.71], WIN_COLORS[w])
    tile(ax, cols[col], y_proj, f"max {WIN_NAMES[w]}", [RAW] * 4, WIN_COLORS[w], lw=5)

# ------------------------------------------------- stage D: 2D segmentation --
for w, col in enumerate(win_cols):
    tri_down(ax, cols[col], (y_proj - 0.7 + y_seg + 0.7) / 2, WIN_COLORS[w])
    tile(ax, cols[col], y_seg, WIN_NAMES[w], OBJ, WIN_COLORS[w], lw=5)

# -------------------------------------------------- stage E: decoupling ----
panel = rect(10.7, (dec_top + dec_bot) / 2, 8.0, dec_top - dec_bot)
ax.add_patch(Polygon(panel, facecolor="white", edgecolor="none", zorder=2))
for b in [8.3, 9.9, 11.5, 13.1]:
    ax.plot(
        [b, b],
        [dec_bot, dec_top],
        color=SLATE,
        lw=1,
        ls=(0, (4, 4)),
        alpha=0.5,
        zorder=2.2,
    )
outline(ax, panel, SLATE, 2.3, lw=1.4)

for w, col in enumerate(win_cols):  # fan-out: one copy per slice
    for c in range(w, w + 3):
        xe = cols[c] + (w - 1) * 0.3
        line(ax, [cols[col], xe], [y_seg - 0.71, dec_top + 0.14], WIN_COLORS[w], lw=1.8)
        tri_down(ax, xe, dec_top + 0.07, WIN_COLORS[w], s=0.1)
        tile(ax, cols[c], rows_dec[w], WIN_NAMES[w], OBJ, WIN_COLORS[w], lw=5)

# --------------------------------------- stage F: collapse into consensus ----
for i, x in enumerate(cols):
    collapse_funnel(
        ax, x, dec_bot, y_cons + 0.72, [w for w in range(3) if w <= i <= w + 2]
    )
    tile(ax, x, y_cons, f"z{i}", OBJ, GOLD, lw=5)

# ------------------------------------------------------------ row labels ----
rows = [
    (y_orig, "B", "Original\nslices"),
    (y_proj, "C", "Sliding-window\nmax projection"),
    (y_seg, "D", "2D segmentation\nof each window"),
    (7.37, "E", "Decouple: copy\neach window's\nmasks back to\nits three slices"),
    (3.55, "F", "Collapse:\nmax over each\nslice's copies\n→ consensus"),
]
for y, n, s in rows:
    badge(ax, x_badge, y, n)
    label(ax, x_label, y, s)

# ------------------------------------------- stage G: link across z (graph) --
gp = rect(16.85, 6.33, 3.3, 8.55)
ax.add_patch(Polygon(gp, facecolor=STRIPE, edgecolor="none", zorder=1))
outline(ax, gp, GOLD, 1.5, lw=1.8)
badge(ax, 15.65, 10.1, "G", GOLD)
label(ax, 16.08, 10.1, "Link across z", size=22, weight="bold")

gx = [16.25, 16.9, 17.55, 18.2]
gy = [y_cons + k * 1.63 for k in range(5)]  # z0 level with the consensus row
gc = [BLUE, ORANGE, TEAL, PINK]
for k in range(4):
    for a in range(4):
        for b in range(4):
            if a != b:
                ax.plot(
                    [gx[a], gx[b]],
                    [gy[k], gy[k + 1]],
                    color=RAW,
                    lw=0.9,
                    alpha=0.45,
                    zorder=3,
                )
        line(ax, [gx[a], gx[a]], [gy[k], gy[k + 1]], gc[a], z=4, lw=3.2)
for k, y in enumerate(gy):
    label(ax, 15.65, y, f"z{k}", ha="center", size=20, color=MUTED, weight="bold")
    for x, c in zip(gx, gc):
        gem(ax, x, y, c, r=0.13, z=8)

arrow_h(ax, 14.82, 15.17, y_cons, GOLD)  # consensus -> graph

# ---------------------------------------------------- stage H: 3D masks ----
mp = rect(20.38, 6.33, 2.85, 8.55)
ax.add_patch(Polygon(mp, facecolor=STRIPE, edgecolor="none", zorder=1))
outline(ax, mp, GOLD, 1.5, lw=1.8)
badge(ax, 19.33, 10.1, "H", GOLD)
label(ax, 19.75, 10.1, "3D masks", size=22, weight="bold")
for (x, y), c in zip(
    [(19.7, 7.6), (21.05, 7.6), (19.7, 5.4), (21.05, 5.4)], [BLUE, ORANGE, TEAL, PINK]
):
    sphere(ax, x, y, 0.5, c)
label(ax, 20.38, 3.3, "one volume\nper object", ha="center", size=20, color=MUTED)
arrow_h(ax, 18.55, 18.93, 6.5, GOLD)  # graph -> masks

# ------------------------------------------------------------------ legend --
lp = rect(18.5, 13.27, 6.6, 4.9)
ax.add_patch(Polygon(lp, facecolor="white", edgecolor=RAW, lw=1.3, zorder=2))
label(ax, 15.55, 15.3, "Key", size=24, weight="bold")
lx, tx = 15.65, 17.15
ey = [14.6, 13.9, 13.2, 12.5, 11.8, 11.1]
gem(ax, lx + 0.08, ey[0], RAW)
label(ax, tx, ey[0], "Raw signal (unlabelled)", size=21)
for i, c in enumerate(OBJ):
    gem(ax, lx + 0.08 + i * 0.38, ey[1], c)
label(ax, tx, ey[1], "Object labels", size=21)
for i, c in enumerate(WIN_COLORS):
    outline(ax, rect(lx + 0.15 + i * 0.5, ey[2], 0.38, 0.38), c, 5, lw=2.4)
label(ax, tx, ey[2], "Windows z0–2 · z1–3 · z2–4", size=21)
outline(ax, rect(lx + 0.15, ey[3], 0.38, 0.38), GOLD, 5, lw=2.4)
label(ax, tx, ey[3], "Consensus & 3D output", size=21)
ax.plot([lx - 0.05, lx + 1.25], [ey[4], ey[4]], color=RAW, lw=1, alpha=0.6, zorder=5)
label(ax, tx, ey[4], "Candidate link", size=21)
line(ax, [lx - 0.05, lx + 1.25], [ey[5], ey[5]], BLUE, z=5, lw=3.2)
label(ax, tx, ey[5], "Matched: same object", size=21)

fig.savefig(segmentation_pipeline_figure_file_path, dpi=600, facecolor="white")
plt.show()
