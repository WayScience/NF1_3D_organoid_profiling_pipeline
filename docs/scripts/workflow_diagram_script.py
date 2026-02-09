#!/usr/bin/env python
"""Generate workflow diagram for documentation."""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# Create figure
fig, ax = plt.subplots(1, 1, figsize=(14, 18))
ax.set_xlim(0, 10)
ax.set_ylim(0, 24)
ax.axis("off")

# Define colors
color_input = "#E8F4F8"
color_process = "#B3E5FC"
color_output = "#81D4FA"
color_analysis = "#4FC3F7"
color_decision = "#FFE082"


# Helper function to create boxes
def create_box(ax, x, y, width, height, text, color, fontsize=10):
    box = FancyBboxPatch(
        (x - width / 2, y - height / 2),
        width,
        height,
        boxstyle="round,pad=0.1",
        edgecolor="black",
        facecolor=color,
        linewidth=2,
    )
    ax.add_patch(box)
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        weight="bold",
        wrap=True,
    )


# Helper function to create arrows
def create_arrow(ax, x1, y1, x2, y2, label=""):
    arrow = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle="->",
        mutation_scale=30,
        linewidth=2,
        color="black",
    )
    ax.add_patch(arrow)
    if label:
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mid_x + 0.3, mid_y, label, fontsize=8, style="italic")


# Title
ax.text(
    5, 23, "NF1 3D Organoid Profiling Pipeline", ha="center", fontsize=16, weight="bold"
)

# Stage 0: Raw Data
create_box(ax, 5, 21.5, 3, 0.8, "Raw Microscopy Images", color_input)
create_arrow(ax, 5, 21.1, 5, 20.4)

# Stage 0: Preprocessing
create_box(
    ax,
    5,
    19.8,
    3.5,
    0.8,
    "Stage 0: Data Preprocessing\n(0.preprocessing_data)",
    color_process,
    fontsize=9,
)
create_arrow(ax, 5, 19.4, 5, 18.6)

create_box(ax, 3.7, 17.8, 2.5, 0.6, "Z-stack Creation", color_process, fontsize=8)
create_box(ax, 6.5, 17.8, 2.5, 0.6, "Deconvolution", color_process, fontsize=8)

ax.text(5, 17, "↓", fontsize=20, ha="center")
create_box(ax, 5, 16.3, 3, 0.8, "Deconvolved Z-Stack Images", color_output)
create_arrow(ax, 5, 15.9, 5, 15.2)

# Stage 1: QC
create_box(
    ax,
    5,
    14.6,
    3.5,
    0.8,
    "Stage 1: Image Quality Control\n(1.image_quality_control)",
    color_process,
    fontsize=9,
)
create_arrow(ax, 5, 14.2, 5, 13.4)

create_box(ax, 3.5, 12.6, 2.5, 0.6, "Blur Detection", color_process, fontsize=8)
create_box(ax, 6.5, 12.6, 2.5, 0.6, "Saturation Check", color_process, fontsize=8)

ax.text(5, 11.8, "↓", fontsize=20, ha="center")
create_box(ax, 5, 11.1, 3, 0.8, "QC Flags & Reports", color_decision)
create_arrow(ax, 5, 10.7, 5, 10)

# Stage 2: Segmentation
create_box(
    ax,
    5,
    9.4,
    3.5,
    0.8,
    "Stage 2: Image Segmentation\n(2.segment_images)",
    color_process,
    fontsize=9,
)
create_arrow(ax, 5, 9, 5, 8.2)

create_box(ax, 1.5, 7.4, 2.2, 0.6, "Nuclei Seg.", color_process, fontsize=8)
create_box(ax, 4, 7.4, 2.2, 0.6, "Cell Seg.", color_process, fontsize=8)
create_box(ax, 6.5, 7.4, 2.2, 0.6, "Organoid Seg.", color_process, fontsize=8)
create_box(ax, 9, 7.4, 1.5, 0.6, "Refinement", color_process, fontsize=7)

ax.text(5, 6.6, "↓", fontsize=20, ha="center")
create_box(ax, 5, 5.9, 3, 0.8, "3D Segmentation Masks", color_output)
create_arrow(ax, 5, 5.5, 5, 4.8)

# Stage 3: Feature Extraction
create_box(
    ax,
    5,
    4.2,
    3.5,
    0.8,
    "Stage 3: Feature Extraction\n(3.cellprofiling)",
    color_process,
    fontsize=9,
)
create_arrow(ax, 5, 3.8, 5, 3)

create_box(ax, 0.8, 2.2, 1.5, 0.5, "Area/Size", color_process, fontsize=7)
create_box(ax, 2.3, 2.2, 1.5, 0.5, "Intensity", color_process, fontsize=7)
create_box(ax, 3.8, 2.2, 1.5, 0.5, "Texture", color_process, fontsize=7)
create_box(ax, 5.3, 2.2, 1.5, 0.5, "Coloc.", color_process, fontsize=7)
create_box(ax, 6.8, 2.2, 1.5, 0.5, "Neighbors", color_process, fontsize=7)
create_box(ax, 8.3, 2.2, 1.5, 0.5, "DL Features", color_process, fontsize=7)

ax.text(5, 1.4, "↓", fontsize=20, ha="center")
create_box(ax, 5, 0.7, 3, 0.8, "Feature Matrices (Parquets)", color_output)

# Create legend
fig.text(0.12, 0.02, "© NF1 Organoid Profiling Pipeline", fontsize=8, style="italic")

plt.tight_layout()
plt.savefig(
    "/home/lippincm/Documents/NF1_3D_organoid_profiling_pipeline/docs/source/_static/workflow_diagram.png",
    dpi=600,
    bbox_inches="tight",
    facecolor="white",
)
print("Workflow diagram saved to docs/source/_static/workflow_diagram.png")
plt.close()
