"""Tests for question (f): do click rates stay the same over the week?

Three checks:
  A. Overall CTR per day: chi-square test of homogeneity.
  B. Item rankings in the first vs second half of the week: Spearman rank
     correlation, compared with the correlation between two random halves
     (the value we would expect from noise alone if nothing changed).
  C. Likelihood ratio test: do items change differently over time?
     Logistic model click ~ item + period against the model where every
     item has its own rate in every period (item x period).
"""
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf


def daily_chi_square(df):
    """Check A. Table of clicks and non clicks per day, and the chi-square
    test of 'every day has the same CTR'."""
    day = df["timestamp"].dt.date
    tab = df.groupby(day)["click"].agg(rows="size", clicks="sum")
    tab["non_clicks"] = tab["rows"] - tab["clicks"]
    tab["ctr"] = tab["clicks"] / tab["rows"]
    tab.index = [f"{d} ({pd.Timestamp(d).day_name()[:3]})" for d in tab.index]
    chi2, p, dof, expected = stats.chi2_contingency(tab[["clicks", "non_clicks"]].to_numpy())
    test = {"chi2": chi2, "df": dof, "p_value": p, "smallest_expected_count": expected.min()}
    return tab, test


def item_ctr_by_group(df, group):
    """CTR of every item within each value of `group` (an array of labels)."""
    g = df.groupby([df["item_id"], group])["click"].mean().unstack()
    return g


def half_split_correlation(df, n_random=20, seed=1):
    """Check B. Spearman correlation of item CTRs between the first three days
    and the last four, and the same correlation between random halves."""
    days = sorted(df["timestamp"].dt.date.unique())
    half = np.where(df["timestamp"].dt.date <= days[2], "days 1-3", "days 4-7")
    by_time = item_ctr_by_group(df, half)
    rho_time = stats.spearmanr(by_time["days 1-3"], by_time["days 4-7"]).statistic

    rng = np.random.default_rng(seed)
    rho_random = []
    for _ in range(n_random):
        random_half = rng.integers(0, 2, size=len(df))
        g = item_ctr_by_group(df, random_half)
        rho_random.append(stats.spearmanr(g[0], g[1]).statistic)
    rho_random = np.array(rho_random)

    top_time = len(set(by_time["days 1-3"].nlargest(10).index) & set(by_time["days 4-7"].nlargest(10).index))
    summary = {
        "spearman_first_vs_second_half": rho_time,
        "spearman_random_halves_mean": rho_random.mean(),
        "spearman_random_halves_min": rho_random.min(),
        "spearman_random_halves_max": rho_random.max(),
        "top10_items_in_both_halves": top_time,
    }
    return by_time, summary


def interaction_lr_test(df, period):
    """Check C. Likelihood ratio test of item x period interaction.

    The data is collapsed to one row per (item, period) with its clicks and
    shows. A binomial model on these counts has the same likelihood as a
    logistic regression on the individual rows, but is far faster.
    Restricted model:  logit P(click) = item effect + period effect
    Full model:        every (item, period) cell has its own rate (saturated)
    LR statistic = 2 * (loglik full - loglik restricted), chi-square with
    (items - 1) * (periods - 1) degrees of freedom under the null of no change
    in the item ranking.
    """
    cells = (df.assign(period=period)
               .groupby(["item_id", "period"])["click"]
               .agg(clicks="sum", shows="size").reset_index())
    cells["non_clicks"] = cells["shows"] - cells["clicks"]
    cells["item_id"] = cells["item_id"].astype(str)
    cells["period"] = cells["period"].astype(str)

    restricted = smf.glm("clicks + non_clicks ~ C(item_id) + C(period)",
                         data=cells, family=sm.families.Binomial()).fit()
    # The deviance of the restricted model is exactly 2 * (loglik saturated
    # minus loglik restricted), so it is the LR statistic we need.
    lr = restricted.deviance
    n_items, n_periods = cells["item_id"].nunique(), cells["period"].nunique()
    dof = (n_items - 1) * (n_periods - 1)
    expected = restricted.fittedvalues * cells["shows"]
    return {
        "periods": n_periods,
        "LR_statistic": lr,
        "df": dof,
        "p_value": stats.chi2.sf(lr, dof),
        "cells_with_expected_clicks_below_5_percent": round(100 * (expected < 5).mean(), 1),
    }


def hour_blocks(df):
    """Index of the calendar hour each row falls in (0 to 167 for one week)."""
    hour = df["timestamp"].dt.floor("h").astype("int64").to_numpy()
    return np.searchsorted(np.unique(hour), hour)


def block_permutation(df, n_groups, statistic, n_perm=200, seed=1):
    """Placebo distribution of a statistic when whole hours are dealt out at
    random into `n_groups` groups.

    Why: rows that are close in time are not independent (the log shows the
    same item many rows in a row), so textbook p-values are too small.
    Shuffling whole hours keeps that dependence but removes any real trend
    over the week. If the real split by day or half week gives a statistic far
    outside this placebo range, the change over time is real.
    """
    block = hour_blocks(df)
    n_blocks = block.max() + 1
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_perm):
        group_of_block = rng.integers(0, n_groups, size=n_blocks)
        values.append(statistic(group_of_block[block]))
    return np.array(values)


def chi_square_stat(df, groups):
    tab = pd.crosstab(groups, df["click"].to_numpy())
    return stats.chi2_contingency(tab.to_numpy())[0]


def spearman_stat(df, groups):
    m = df.groupby([df["item_id"].to_numpy(), groups])["click"].mean().unstack()
    return stats.spearmanr(m.iloc[:, 0], m.iloc[:, 1]).statistic


def position_permutation(df, statistic, n_perm=200, seed=1):
    """Placebo for 'does the best item depend on the position?'.

    Within every hour, the three position labels are randomly swapped between
    the three slots. Each slot keeps its own run of rows (so the streaks in the
    log stay intact), but any real link between position and item ranking is
    destroyed. The real statistic is then compared with these placebo values.
    """
    block = hour_blocks(df)
    pos = df["position"].to_numpy() - 1           # 0, 1, 2
    n_blocks = block.max() + 1
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_perm):
        perms = np.array([rng.permutation(3) for _ in range(n_blocks)])
        values.append(statistic(perms[block, pos]))
    return np.array(values)
