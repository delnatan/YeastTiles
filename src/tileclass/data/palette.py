"""Distinct-color palette for category badges.

Trimmed from pyvistra's ``data/colors.py`` -- only the flat
index-into-palette lookup used for tagging categories with a stable
color, none of the label-adjacency graph-coloring machinery (that's for
segmentation-mask overlays, unrelated here).
"""

PALETTE = [
    (0.90, 0.10, 0.29),  # Red
    (0.23, 0.70, 0.29),  # Green
    (0.00, 0.51, 0.78),  # Blue
    (0.96, 0.51, 0.19),  # Orange
    (0.57, 0.12, 0.71),  # Purple
    (0.27, 0.94, 0.94),  # Cyan
    (0.94, 0.20, 0.90),  # Magenta
    (0.98, 0.75, 0.00),  # Yellow
    (0.00, 0.50, 0.50),  # Teal
    (0.86, 0.75, 1.00),  # Lavender
]


def rgb_to_hex(color):
    r, g, b = color[:3]
    return "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))


def category_color(index):
    """Hex color for the *index*-th distinct category (wraps around)."""
    return rgb_to_hex(PALETTE[index % len(PALETTE)])
