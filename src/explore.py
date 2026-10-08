"""Descriptive tables and figures for the data section of the report."""
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .data import user_cols

BLUE = "#2a78d6"
INK = "#52514e"


def wilson(clicks, n, z=1.96):
    """95% Wilson interval for a click rate. Better than the normal
    approximation when the rate is close to zero, as it is here."""
    clicks = np.asarray(clicks, float)
    n = np.asarray(n, float)
    p = clicks / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return np.clip(centre - half, 0, 1), np.clip(centre + half, 0, 1)


def ctr_table(df, by):
    g = df.groupby(by)["click"].agg(rows="size", clicks="sum")
    g["ctr"] = g["clicks"] / g["rows"]
    g["ci_lo"], g["ci_hi"] = wilson(g["clicks"], g["rows"])
    return g


def summary_row(name, df):
    return {
        "file": name,
        "rows": len(df),
        "items": df["item_id"].nunique(),
        "clicks": int(df["click"].sum()),
        "ctr": df["click"].mean(),
        "first": df["timestamp"].min().strftime("%Y-%m-%d %H:%M"),
        "last": df["timestamp"].max().strftime("%Y-%m-%d %H:%M"),
        "user_features": len(user_cols(df)),
    }


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(INK)
    ax.tick_params(colors=INK, labelsize=8)
    ax.grid(axis="y", color="#e6e5e1", linewidth=0.6)
    ax.set_axisbelow(True)


def plot_item_ctr(tab, overall, path):
    t = tab.sort_values("ctr", ascending=False)
    x = np.arange(len(t))
    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.vlines(x, t["ci_lo"] * 100, t["ci_hi"] * 100, color=BLUE, alpha=0.35, linewidth=2)
    ax.scatter(x, t["ctr"] * 100, s=14, color=BLUE, zorder=3)
    ax.axhline(overall * 100, color=INK, linewidth=1, linestyle="--")
    ax.text(len(t) - 1, overall * 100, " overall", va="bottom", ha="right", fontsize=8, color=INK)
    ax.set_xticks(x[::5])
    ax.set_xticklabels(t.index[::5], fontsize=7)
    ax.set_xlabel("item (sorted by CTR)", color=INK, fontsize=9)
    ax.set_ylabel("CTR (%)", color=INK, fontsize=9)
    ax.set_title("CTR per item with 95% Wilson interval", fontsize=10, loc="left")
    style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_daily_ctr(tab, path):
    x = np.arange(len(tab))
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.fill_between(x, tab["ci_lo"] * 100, tab["ci_hi"] * 100, color=BLUE, alpha=0.15, linewidth=0)
    ax.plot(x, tab["ctr"] * 100, color=BLUE, linewidth=2, marker="o", markersize=4)
    ax.set_xticks(x)
    ax.set_xticklabels([str(d) for d in tab.index], fontsize=8)
    ax.set_ylabel("CTR (%)", color=INK, fontsize=9)
    ax.set_title("Daily CTR with 95% Wilson interval", fontsize=10, loc="left")
    style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


ORANGE = "#eb6834"


def plot_learning(curves, oracle_ctr, path, skip=200):
    """Cumulative CTR against number of matched rows, one line per policy.
    The first `skip` matches are left out because a CTR over a handful of
    rows jumps around too much to read."""
    colors = {"TS": BLUE, "Random": INK, "UCB": ORANGE}
    fig, ax = plt.subplots(figsize=(7, 3.4))
    for name, c in curves.items():
        x = np.arange(1, len(c) + 1)[skip:]
        ax.plot(x, c[skip:] * 100, color=colors.get(name, BLUE), linewidth=2)
        ax.text(x[-1], c[-1] * 100, f" {name}", va="center", fontsize=8, color=colors.get(name, BLUE))
    ax.axhline(oracle_ctr * 100, color=INK, linewidth=1, linestyle="--")
    ax.text(skip, oracle_ctr * 100, "Oracle (best item in hindsight)", va="bottom", fontsize=8, color=INK)
    ax.set_xlabel("matched rows so far", color=INK, fontsize=9)
    ax.set_ylabel("cumulative CTR (%)", color=INK, fontsize=9)
    ax.set_title("Cumulative CTR during replay, mean over runs", fontsize=10, loc="left")
    style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
