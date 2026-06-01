"""Shared plotting helpers so every baseline/presentation figure uses one
consistent visual format per *kind* of quantity.

Two formats, applied by what the value IS — not by which notebook draws it:

- **scalar-per-condition** (success rate, steps-to-source): a colored mean
  point+line, with the individual per-seed sub-aggregates overlaid as grey dots.
  Use :func:`point_line_with_seeds`.
- **distribution-per-group** (neuron overlap, oscillation fraction, anything
  with a spread of samples): a box plot. Use :func:`boxplot_by_group`.

All figures share :func:`apply_style`.
"""
from __future__ import annotations

import numpy as np


# Presentation palette (kept in sync with presentation_assets/).
BLUE = PRESENTATION_BLUE = "#1f77b4"
RED = PRESENTATION_RED = "#d62728"
ACCENT = BLUE
GREY = "0.6"


def apply_style():
    import matplotlib as mpl
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.titleweight": "normal",
        "axes.labelsize": 12,
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 3.0,
        "lines.markersize": 9,
        "legend.frameon": False,
        "figure.dpi": 110,
        "savefig.dpi": 150,
    })


def point_line_with_seeds(ax, x, seed_values, *, color=BLUE, label=None,
                          jitter=0.06, seed_alpha=0.5, connect=True):
    """Mean point+line (colored) with per-seed values overlaid as grey dots.

    Parameters
    ----------
    ax : matplotlib Axes
    x : sequence of length C — x position per condition.
    seed_values : list of length C; each entry is a 1-D array of per-seed
        sub-aggregate values for that condition. The mean over seeds is the
        colored point; each seed value is a grey dot (x-jittered).
    color : color of the mean point+line.
    label : legend label for the mean series.
    jitter : horizontal jitter (in x-units) for the grey seed dots.
    connect : draw the connecting line through the means.
    """
    x = np.asarray(x, dtype=float)
    means = np.array([np.nanmean(v) if len(v) else np.nan for v in seed_values])
    rng = np.random.default_rng(0)
    # Grey per-seed dots first (behind the mean).
    for xi, vals in zip(x, seed_values):
        vals = np.asarray(vals, dtype=float)
        if not len(vals):
            continue
        jx = xi + rng.uniform(-jitter, jitter, size=len(vals))
        ax.scatter(jx, vals, s=28, color=GREY, alpha=seed_alpha, zorder=2,
                   edgecolors="none")
    if connect:
        ax.plot(x, means, color=color, lw=3.0, zorder=3,
                label=label, marker="o", markersize=9)
    else:
        ax.scatter(x, means, color=color, s=90, zorder=3, label=label)
    return means


def boxplot_by_group(ax, group_values, group_labels, *, colors=None,
                     show_points=True, point_alpha=0.5):
    """Box plot of a distribution per group, with optional jittered points.

    Parameters
    ----------
    group_values : list of 1-D arrays — one distribution per group.
    group_labels : list of str — x tick labels.
    colors : optional list of box face colors (one per group).
    show_points : overlay the raw samples as jittered grey dots.
    """
    positions = np.arange(1, len(group_values) + 1)
    bp = ax.boxplot(
        [np.asarray(v, dtype=float) for v in group_values],
        positions=positions, widths=0.55, patch_artist=True,
        showfliers=False, medianprops=dict(color="black", lw=2),
        whiskerprops=dict(color="0.3"), capprops=dict(color="0.3"),
        boxprops=dict(edgecolor="0.3"),
    )
    if colors is not None:
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.55)
    else:
        for patch in bp["boxes"]:
            patch.set_facecolor(BLUE)
            patch.set_alpha(0.45)
    if show_points:
        rng = np.random.default_rng(0)
        for pos, vals in zip(positions, group_values):
            vals = np.asarray(vals, dtype=float)
            if not len(vals):
                continue
            jx = pos + rng.uniform(-0.13, 0.13, size=len(vals))
            ax.scatter(jx, vals, s=20, color=GREY, alpha=point_alpha,
                       zorder=3, edgecolors="none")
    ax.set_xticks(positions)
    ax.set_xticklabels(group_labels)
    return bp


def seed_subaggregates(per_episode_values, per_episode_seeds, reducer=np.mean):
    """Group per-episode values by seed and reduce each group.

    Returns a 1-D array of one sub-aggregate per distinct seed — the grey-dot
    distribution for :func:`point_line_with_seeds`.
    """
    vals = np.asarray(per_episode_values, dtype=float)
    seeds = np.asarray(per_episode_seeds)
    out = []
    for s in np.unique(seeds):
        m = seeds == s
        if m.any():
            out.append(float(reducer(vals[m])))
    return np.asarray(out, dtype=float)
