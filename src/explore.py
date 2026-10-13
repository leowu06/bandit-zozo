"""Descriptive tables and all figures of the report."""
import warnings
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .data import user_cols

BLUE = "#2a78d6"
ORANGE = "#eb6834"
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
    """Rows, clicks, CTR and Wilson interval for each value of `by`."""
    g = df.groupby(by)["click"].agg(rows="size", clicks="sum")
    g["ctr"] = g["clicks"] / g["rows"]
    g["ci_lo"], g["ci_hi"] = wilson(g["clicks"], g["rows"])
    return g


def summary_row(name, df):
    """One line of Table 1 for one data file."""
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
    """Shared look of all figures: no top and right border, light grid."""
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(INK)
    ax.tick_params(colors=INK, labelsize=8)
    ax.grid(axis="y", color="#e6e5e1", linewidth=0.6)
    ax.set_axisbelow(True)


def plot_item_ctr(tab, overall, path):
    """Figure 1: CTR of every item, best first, with its interval."""
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
    """Figure 2: CTR per day with its interval."""
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


def plot_learning(curves, oracle_ctr, path, skip=200):
    """Figure 3: cumulative CTR against matched rows, one line per policy.
    The first `skip` matches are left out, since a CTR over a few rows
    jumps around too much to read."""
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


def plot_tuning(ts_tab, ucb_tab, random_ctr, path):
    """Figure 5: CTR for each TS prior (left) and each UCB bonus scale
    (right), on the same CTR scale."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
    panels = [(axes[0], ts_tab, "prior_strength_b", "TS: prior Beta(1, b), b on log scale", BLUE),
              (axes[1], ucb_tab, "c", "UCB: bonus scale c, log scale", ORANGE)]
    for ax, tab, xcol, title, colour in panels:
        x = tab[xcol].to_numpy(dtype=float)
        y = tab["ctr_mean"].to_numpy() * 100
        se = tab["se_binomial"].to_numpy() * 100
        ax.fill_between(x, y - 1.96 * se, y + 1.96 * se, color=colour, alpha=0.15, linewidth=0)
        ax.plot(x, y, color=colour, linewidth=2, marker="o", markersize=5)
        ax.axhline(random_ctr * 100, color=INK, linewidth=1, linestyle="--")
        ax.text(x.min(), random_ctr * 100, "Random", va="bottom", fontsize=8, color=INK)
        ax.set_xscale("log")
        ax.set_title(title, fontsize=9, loc="left")
        style(ax)
    axes[0].set_ylabel("CTR (%)", color=INK, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_batch(tab, random_ctr, path):
    """Figure 4: CTR against batch size, one line per policy. The band is
    plus or minus 1.96 binomial standard errors."""
    fig, ax = plt.subplots(figsize=(6.5, 3.3))
    for (label, part), colour in zip(tab.groupby("policy", sort=False), [BLUE, ORANGE]):
        x = part["batch_size"].to_numpy(dtype=float)
        y = part["ctr_mean"].to_numpy() * 100
        se = part["se_binomial"].to_numpy() * 100
        ax.fill_between(x, y - 1.96 * se, y + 1.96 * se, color=colour, alpha=0.15, linewidth=0)
        ax.plot(x, y, color=colour, linewidth=2, marker="o", markersize=5)
        ax.text(x[-1], y[-1], f" {label}", va="center", fontsize=8, color=colour)
    ax.axhline(random_ctr * 100, color=INK, linewidth=1, linestyle="--")
    ax.text(1, random_ctr * 100, "Random", va="bottom", fontsize=8, color=INK)
    ax.set_xscale("log")
    ax.set_xlabel("batch size (matched rows per update, log scale)", color=INK, fontsize=9)
    ax.set_ylabel("CTR (%)", color=INK, fontsize=9)
    ax.set_title("CTR when the policy learns in batches", fontsize=10, loc="left")
    style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_half_split(by_time, path):
    """Figure 7: each item's CTR in days 1 to 3 against days 4 to 7.
    A point on the diagonal did the same in both halves."""
    x = by_time["days 1-3"].to_numpy() * 100
    y = by_time["days 4-7"].to_numpy() * 100
    hi = max(x.max(), y.max()) * 1.05
    fig, ax = plt.subplots(figsize=(4.6, 4.3))
    ax.plot([0, hi], [0, hi], color=INK, linewidth=1, linestyle="--")
    ax.scatter(x, y, s=16, color=BLUE, alpha=0.8, edgecolor="white", linewidth=0.5)
    ax.set_xlim(0, hi)
    ax.set_ylim(0, hi)
    ax.set_xlabel("CTR days 1 to 3 (%)", color=INK, fontsize=9)
    ax.set_ylabel("CTR days 4 to 7 (%)", color=INK, fontsize=9)
    ax.set_title("Item CTR, first vs second half of the week", fontsize=10, loc="left")
    style(ax)
    ax.grid(axis="x", color="#e6e5e1", linewidth=0.6)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_shift(curves, switch_row, path, window=1000):
    """Figure 8: moving CTR over the last `window` matched rows on the log
    where the item ranking flips halfway. The dashed line marks the flip."""
    colours = [INK, BLUE, "#8a8984", ORANGE]
    fig, ax = plt.subplots(figsize=(7, 3.4))
    for (label, runs), colour in zip(curves.items(), colours):
        # Runs match different rows, so put them on a shared grid of log rows.
        grid = np.linspace(0, max(r["matched_rows"].max() for r in runs), 300)
        lines = []
        for r in runs:
            c = np.array(r["matched_clicks"], dtype=float)
            moving = pd.Series(c).rolling(window, min_periods=window // 2).mean().to_numpy()
            lines.append(np.interp(grid, r["matched_rows"], moving))
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore")   # the first points have no full window yet
            ax.plot(grid, np.nanmean(lines, axis=0) * 100, color=colour, linewidth=2, label=label)
    ax.axvline(switch_row, color=INK, linewidth=1, linestyle="--")
    ax.text(switch_row, ax.get_ylim()[1], " ranking flips", va="top", fontsize=8, color=INK)
    ax.set_xlabel("row of the log (time order)", color=INK, fontsize=9)
    ax.set_ylabel(f"CTR over last {window} matches (%)", color=INK, fontsize=9)
    ax.set_title("Who adapts when the best item changes", fontsize=10, loc="left")
    ax.legend(fontsize=8, frameon=False)
    style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_fairness(tab, path):
    """Figure 6: CTR against the Gini of item exposure, one point per policy.
    A lower Gini is fairer to items."""
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    x = tab["gini_exposure"].to_numpy()
    y = tab["ctr_mean"].to_numpy() * 100
    se = tab["se_binomial"].to_numpy() * 100
    ax.errorbar(x, y, yerr=1.96 * se, fmt="o", color=BLUE, ecolor=BLUE, alpha=0.9,
                elinewidth=1, capsize=0, markersize=6)
    # UCB with and without the weak floor almost overlap, so a point close to
    # one already labelled gets its label below. The wider x axis leaves room
    # for the labels.
    placed = []
    for xi, yi, label in zip(x, y, tab["policy"]):
        near = any(abs(xi - xp) < 0.06 and abs(yi - yp) < 0.05 for xp, yp in placed)
        ax.annotate(label, (xi, yi), textcoords="offset points", xytext=(6, -12 if near else 4),
                    fontsize=7, color=INK)
        placed.append((xi, yi))
    ax.set_xlim(-0.02, max(1.0, x.max() + 0.3))
    ax.set_xlabel("Gini of item exposure (0 = every item shown equally)", color=INK, fontsize=9)
    ax.set_ylabel("CTR (%)", color=INK, fontsize=9)
    ax.set_title("Clicks against fairness to items", fontsize=10, loc="left")
    style(ax)
    ax.grid(axis="x", color="#e6e5e1", linewidth=0.6)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
