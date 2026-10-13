"""Assignment 2, Learning from Big Data 2026.

Running this file prints every table and saves every figure used in the
report. Each one carries the ID it has in the report.
"""
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src.data import load, user_cols
from src import explore as ex
from src.replay import RandomPolicy, FixedPolicy, run_many, summarise
from src.policies import (ThompsonSampling, UCB, DiscountedTS, SlidingWindowUCB, SegmentedTS,
                          ExposureFloor)
from src import fairness as fair
from src.report_io import save_table, start_log
from src import stationarity as st

random.seed(1)
np.random.seed(1)

OUT = Path(__file__).resolve().parent / "output"
OUT.mkdir(exist_ok=True)
pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 20)

N_RUNS = 10  # replay repetitions for stochastic policies
N_TUNE = 5   # repetitions per setting in the tuning grid (keeps run time down)

# Settings tried in question (c)
TS_PRIOR_B = [1, 30, 100, 300, 1000]        # prior Beta(1, b), prior mean 1/(1+b)
UCB_C = [1, 0.1, 0.03, 0.01, 0.003, 0.001]  # scale of the UCB bonus

# Settings tried in question (b)
BATCH_SIZES = [1, 10, 100, 500, 1000, 5000]  # matched rows collected before each update
UCB_C_FOR_B = 0.01  # UCB with c = 1 does not learn (see question c), so use a working c


def header(label):
    print(f"\n--- {label} ---")


def section_data(dfs):
    header("Table 1: Overview of the three data files")
    t1 = pd.DataFrame([ex.summary_row(k, d) for k, d in dfs.items()])
    print(t1.to_string(index=False))
    save_table(t1, "table_1_overview", index=False)

    df = dfs["80 items"]

    header("Table 2: Propensity scores in the 80 item file")
    t2 = df["propensity_score"].value_counts().rename("rows").to_frame()
    print(t2.to_string())
    save_table(t2, "table_2_propensity")
    print("Every row has propensity 1/80, so the logging policy was uniform random.")

    header("Table 3: CTR per position (80 items)")
    t3 = ex.ctr_table(df, "position").round(5)
    print(t3.to_string())
    save_table(t3, "table_3_position")

    items = ex.ctr_table(df, "item_id")
    header("Table 4: Ten highest and ten lowest CTR items (80 items)")
    s = items.sort_values("ctr", ascending=False).round(5)
    t4 = pd.concat([s.head(10), s.tail(10)])
    print(t4.to_string())
    save_table(t4, "table_4_items_top_bottom")
    save_table(s, "table_4_items_all")
    ex.plot_item_ctr(items, df["click"].mean(), OUT / "figure_1_item_ctr.png")
    print("\nFigure 1 saved: output/figure_1_item_ctr.png")

    header("Table 5: CTR per level of each user feature (80 items)")
    for c in user_cols(df):
        print(f"\n{c}")
        t5 = ex.ctr_table(df, c).round(5)
        print(t5.to_string())
        save_table(t5, f"table_5_{c}")

    daily = ex.ctr_table(df.assign(day=df["timestamp"].dt.date), "day")
    header("Table 6: CTR per day (80 items)")
    print(daily.round(5).to_string())
    save_table(daily.round(5), "table_6_daily")
    ex.plot_daily_ctr(daily, OUT / "figure_2_daily_ctr.png")
    print("\nFigure 2 saved: output/figure_2_daily_ctr.png")


def section_sanity(dfs):
    rows = []
    for name, df in dfs.items():
        K = df["item_id"].nunique()
        best = df.groupby("item_id")["click"].mean().idxmax()

        r = summarise(run_many(lambda rng: RandomPolicy(K, rng), df, n_runs=N_RUNS))
        o = summarise(run_many(lambda rng: FixedPolicy(K, rng, best), df, n_runs=1))
        for s in (r, o):
            s["file"] = name
            s["log_ctr"] = df["click"].mean()
            rows.append(s)

    header(f"Table 7: Replay sanity check, Random ({N_RUNS} runs) and hindsight Oracle")
    cols = ["file", "policy", "runs", "mean_matched", "mean_clicks", "ctr_mean",
            "ctr_sd_runs", "se_binomial", "log_ctr"]
    t7 = pd.DataFrame(rows)[cols].round(5)
    print(t7.to_string(index=False))
    save_table(t7, "table_7_sanity", index=False)
    print("Random should land close to log_ctr. Oracle is the best single item "
          "in hindsight and is an upper bound for a non contextual policy.")


def mean_curve(runs):
    """Average cumulative CTR across runs, cut to the shortest run."""
    n = min(len(r["cum_ctr"]) for r in runs)
    return np.mean([r["cum_ctr"][:n] for r in runs], axis=0)


def section_a(name, df):
    """Question (a): Thompson Sampling and UCB against Random and Oracle."""
    K = df["item_id"].nunique()
    true_ctr = df.groupby("item_id")["click"].mean()
    best = true_ctr.idxmax()

    runs = {
        "Random": run_many(lambda rng: RandomPolicy(K, rng), df, n_runs=N_RUNS),
        "TS": run_many(lambda rng: ThompsonSampling(K, rng), df, n_runs=N_RUNS),
        "UCB": run_many(lambda rng: UCB(K, rng), df, n_runs=N_RUNS),
        "Oracle": run_many(lambda rng: FixedPolicy(K, rng, best), df, n_runs=1),
    }

    header(f"Table 8 ({name}): CTR of TS and UCB vs Random and Oracle, {N_RUNS} runs")
    rows = [summarise(r) for r in runs.values()]
    cols = ["policy", "runs", "mean_matched", "mean_clicks", "ctr_mean",
            "ctr_sd_runs", "ctr_min_run", "ctr_max_run", "se_binomial"]
    t8 = pd.DataFrame(rows)[cols].round(5)
    print(t8.to_string(index=False))
    tag = name.replace(" ", "")
    save_table(t8, f"table_8_{tag}", index=False)

    header(f"Table 9 ({name}): How often TS and UCB showed each item vs its true CTR")
    t = pd.DataFrame({"true_ctr_percent": (true_ctr * 100).round(3)})
    for p in ("TS", "UCB"):
        shown = np.mean([r["shown"] for r in runs[p]], axis=0)
        t[f"{p}_share_shown_percent"] = (shown / shown.sum() * 100).round(1)
    t9 = t.sort_values("true_ctr_percent", ascending=False)
    print(t9.to_string())
    save_table(t9, f"table_9_{tag}")

    fig_path = OUT / f"figure_3_learning_{name.split()[0]}items.png"
    ex.plot_learning({k: mean_curve(v) for k, v in runs.items() if k != "Oracle"},
                     summarise(runs["Oracle"])["ctr_mean"], fig_path)
    print(f"\nFigure 3 ({name}) saved: output/{fig_path.name}")


def section_b(name, df):
    """Question (b): what happens when the policy only learns in batches?

    With batch size B the policy keeps its beliefs fixed for B matched rows,
    then receives all B results at once. B = 1 is learning after every row.
    """
    tag = name.replace(" ", "")
    K = df["item_id"].nunique()
    random_ctr = summarise(run_many(lambda rng: RandomPolicy(K, rng), df, n_runs=N_RUNS))["ctr_mean"]

    policies = {
        "TS Beta(1,1)": lambda rng: ThompsonSampling(K, rng),
        f"UCB c={UCB_C_FOR_B}": lambda rng: UCB(K, rng, c=UCB_C_FOR_B),
    }
    rows = []
    for batch in BATCH_SIZES:
        for label, make in policies.items():
            s = summarise(run_many(make, df, n_runs=N_TUNE, batch_size=batch))
            s["policy"], s["batch_size"] = label, batch
            rows.append(s)
    tab = pd.DataFrame(rows)

    header(f"Table 14 ({name}): CTR for different batch sizes, {N_TUNE} runs each")
    cols = ["policy", "batch_size", "mean_matched", "mean_clicks", "ctr_mean", "ctr_sd_runs", "se_binomial"]
    t14 = tab[cols].round(5)
    print(t14.to_string(index=False))
    print(f"Random policy on the same data: {random_ctr:.5f}")
    save_table(t14, f"table_14_{tag}", index=False)

    fig_path = OUT / f"figure_4_batch_{tag}.png"
    ex.plot_batch(tab, random_ctr, fig_path)
    print(f"\nFigure 4 ({name}) saved: output/{fig_path.name}")


def tune_grid(df, n_runs):
    """Run every TS prior and every UCB constant once through replay."""
    K = df["item_id"].nunique()
    ts_rows, ucb_rows = [], []
    for b in TS_PRIOR_B:
        s = summarise(run_many(lambda rng: ThompsonSampling(K, rng, prior_a=1, prior_b=b), df, n_runs=n_runs))
        s.update(setting=f"Beta(1,{b})", prior_strength_b=b, prior_mean_percent=round(100 / (1 + b), 3))
        ts_rows.append(s)
    for c in UCB_C:
        s = summarise(run_many(lambda rng: UCB(K, rng, c=c), df, n_runs=n_runs))
        s.update(setting=f"c={c}", c=c)
        ucb_rows.append(s)
    return pd.DataFrame(ts_rows), pd.DataFrame(ucb_rows)


def section_c(name, df):
    """Question (c): how much do the results depend on the tuning parameters?"""
    tag = name.replace(" ", "")
    K = df["item_id"].nunique()
    random_ctr = summarise(run_many(lambda rng: RandomPolicy(K, rng), df, n_runs=N_RUNS))["ctr_mean"]

    # 1. Sensitivity: every setting on the full week
    ts_tab, ucb_tab = tune_grid(df, N_TUNE)
    cols = ["mean_matched", "mean_clicks", "ctr_mean", "ctr_sd_runs", "se_binomial"]

    header(f"Table 15 ({name}): TS with different priors Beta(1, b), {N_TUNE} runs each")
    t15 = ts_tab[["setting", "prior_mean_percent"] + cols].round(5)
    print(t15.to_string(index=False))
    save_table(t15, f"table_15_{tag}", index=False)

    header(f"Table 16 ({name}): UCB with different bonus scales c, {N_TUNE} runs each")
    t16 = ucb_tab[["setting"] + cols].round(5)
    print(t16.to_string(index=False))
    save_table(t16, f"table_16_{tag}", index=False)

    fig_path = OUT / f"figure_5_tuning_{tag}.png"
    ex.plot_tuning(ts_tab, ucb_tab, random_ctr, fig_path)
    print(f"\nFigure 5 ({name}) saved: output/{fig_path.name}")

    # 2. Fair pick: choose the setting on days 1 and 2 only, then score it on
    #    days 3 to 7. This avoids choosing the setting on the same rows we report.
    days = sorted(df["timestamp"].dt.date.unique())
    tune_df = df[df["timestamp"].dt.date <= days[1]]
    test_df = df[df["timestamp"].dt.date > days[1]]
    ts_tune, ucb_tune = tune_grid(tune_df, N_TUNE)
    best_b = int(ts_tune.loc[ts_tune["ctr_mean"].idxmax(), "prior_strength_b"])
    best_c = float(ucb_tune.loc[ucb_tune["ctr_mean"].idxmax(), "c"])

    candidates = {
        "Random": lambda rng: RandomPolicy(K, rng),
        "TS default Beta(1,1)": lambda rng: ThompsonSampling(K, rng),
        f"TS tuned on days 1-2, Beta(1,{best_b})": lambda rng: ThompsonSampling(K, rng, 1, best_b),
        "UCB default c=1": lambda rng: UCB(K, rng, c=1),
        f"UCB tuned on days 1-2, c={best_c}": lambda rng: UCB(K, rng, c=best_c),
    }
    rows = []
    for label, make in candidates.items():
        s = summarise(run_many(make, test_df, n_runs=N_RUNS))
        s["policy"] = label
        rows.append(s)
    header(f"Table 17 ({name}): Settings chosen on days 1 and 2, scored on days 3 to 7, {N_RUNS} runs")
    t17 = pd.DataFrame(rows)[["policy"] + cols].round(5)
    print(t17.to_string(index=False))
    save_table(t17, f"table_17_{tag}", index=False)

    # 3. Does the order of the log matter? Same policies on shuffled rows.
    #    For UCB we also check the setting that did best over the whole week.
    week_best_c = float(ucb_tab.loc[ucb_tab["ctr_mean"].idxmax(), "c"])
    shuffled = df.sample(frac=1, random_state=1).reset_index(drop=True)
    rows = []
    for label, make in [("Random", lambda rng: RandomPolicy(K, rng)),
                        ("TS default Beta(1,1)", lambda rng: ThompsonSampling(K, rng)),
                        ("UCB default c=1", lambda rng: UCB(K, rng, c=1)),
                        (f"UCB best week setting c={week_best_c}", lambda rng: UCB(K, rng, c=week_best_c))]:
        for order, d in [("time order", df), ("shuffled", shuffled)]:
            s = summarise(run_many(make, d, n_runs=N_TUNE))
            s["policy"], s["row_order"] = label, order
            rows.append(s)
    header(f"Table 18 ({name}): Replay in time order vs shuffled rows, {N_TUNE} runs")
    t18 = pd.DataFrame(rows)[["policy", "row_order"] + cols].round(5)
    print(t18.to_string(index=False))
    save_table(t18, f"table_18_{tag}", index=False)


N_PERM = 200  # placebo splits for the block permutation tests in (f)

# Settings for question (d)
BORROW = 1000   # per group TS: borrow up to this many shows of evidence from the pooled model

# Settings for question (e)
FAIR_FLOORS = [0.25, 0.5]   # every item gets at least this share of a fair (uniform) share of shows
FAIR_FEATURE = "user_feature_0"  # user groups used for fairness to users
MIN_MATCHED = 500               # groups with fewer matched rows per run are too small to judge

# Settings for question (g) and the extension
GAMMAS = [0.9999, 0.999, 0.995]   # discounted TS, memory about 10000, 1000, 200 updates
WINDOWS = [5000, 2000, 1000]      # sliding window UCB, last W matched rows


def section_f(name, df):
    """Question (f): is the data stationary? Uses the log itself, no replay.

    Every test is done twice: the textbook version, which assumes rows are
    independent, and a block permutation version that keeps whole hours
    together (see src/stationarity.py for why).
    """
    tag = name.replace(" ", "")
    days = df["timestamp"].dt.date
    first_days = sorted(days.unique())[:3]
    halves = np.where(days.isin(first_days), 0, 1)

    # Check A: overall CTR per day
    daily, test_a = st.daily_chi_square(df)
    placebo = st.block_permutation(df, 7, lambda g: st.chi_square_stat(df, g), N_PERM)
    test_a.update(placebo_mean=placebo.mean(), placebo_95th_percentile=np.percentile(placebo, 95),
                  permutation_p_value=(placebo >= test_a["chi2"]).mean())
    header(f"Table 23 ({name}): Clicks per day and test of equal daily CTR")
    print(daily.round(5).to_string())
    print(pd.Series(test_a).round(6).to_string())
    save_table(daily.round(5), f"table_23_{tag}")
    save_table(pd.Series(test_a).round(6).to_frame("value"), f"table_23_test_{tag}")

    # Check B: does the item ranking change between the two halves of the week?
    by_time, test_b = st.half_split_correlation(df)
    placebo = st.block_permutation(df, 2, lambda g: st.spearman_stat(df, g), N_PERM)
    test_b.update(spearman_hour_block_halves_mean=placebo.mean(),
                  spearman_hour_block_halves_5th_percentile=np.percentile(placebo, 5),
                  permutation_p_value=(placebo <= test_b["spearman_first_vs_second_half"]).mean())
    header(f"Table 24 ({name}): Item CTR ranking, first vs second half of the week")
    print(pd.Series(test_b).round(4).to_string())
    print("A real change in the ranking would give a first vs second half correlation")
    print("below the hour block placebo values (small permutation p value).")
    save_table(pd.Series(test_b).round(4).to_frame("value"), f"table_24_{tag}")
    fig_path = OUT / f"figure_7_half_split_{tag}.png"
    ex.plot_half_split(by_time, fig_path)
    print(f"\nFigure 7 ({name}) saved: output/{fig_path.name}")

    # Check C: likelihood ratio test of item x period interaction
    rows = []
    for label, period, n_groups in [("by half week", halves, 2), ("by day", days.to_numpy(), 7)]:
        r = st.interaction_lr_test(df, period)
        placebo = st.block_permutation(
            df, n_groups, lambda g: st.interaction_lr_test(df, g)["LR_statistic"], N_PERM)
        r.update(split=label, placebo_mean=placebo.mean(),
                 placebo_95th_percentile=np.percentile(placebo, 95),
                 permutation_p_value=(placebo >= r["LR_statistic"]).mean())
        rows.append(r)
    header(f"Table 25 ({name}): Likelihood ratio test, do items change differently over time?")
    t25 = pd.DataFrame(rows)[["split", "periods", "LR_statistic", "df", "p_value", "placebo_mean",
                              "placebo_95th_percentile", "permutation_p_value",
                              "cells_with_expected_clicks_below_5_percent"]].round(5)
    print(t25.to_string(index=False))
    save_table(t25, f"table_25_{tag}", index=False)


def ctr_per_group(runs, groups):
    """Pool all runs and compute matched rows, clicks and CTR per user group.
    `groups` holds the group of every row of the log."""
    g = np.concatenate([groups[r["matched_rows"]] for r in runs])
    c = np.concatenate([np.array(r["matched_clicks"]) for r in runs])
    tab = pd.DataFrame({"group": g, "click": c}).groupby("group")["click"].agg(matched="size", clicks="sum")
    tab["matched"] = tab["matched"] / len(runs)   # per run, comparable with other tables
    tab["clicks"] = tab["clicks"] / len(runs)
    tab["ctr"] = tab["clicks"] / tab["matched"]
    return tab


def section_d(name, df):
    """Question (d): one model for everyone (aggregated) or one per user group
    (heterogeneous)? For every user feature we compare pooled TS, per group TS
    starting from zero and per group TS that borrows from the pooled model."""
    tag = name.replace(" ", "")
    K = df["item_id"].nunique()
    features = user_cols(df)
    random_runs = run_many(lambda rng: RandomPolicy(K, rng), df, n_runs=N_TUNE)
    pooled_runs = run_many(lambda rng: ThompsonSampling(K, rng), df, n_runs=N_TUNE)

    overview = []
    per_group_tables = {}
    for s, label in [(summarise(random_runs), "Random"), (summarise(pooled_runs), "Pooled TS (one model)")]:
        s.update(feature="all", policy=label)
        overview.append(s)

    for f in features:
        n_levels = df[f].max() + 1
        groups = df[f].to_numpy()
        variants = {
            "Per group TS, from zero": lambda rng: SegmentedTS(K, rng, f, n_levels, borrow=0),
            f"Per group TS, borrowing {BORROW}": lambda rng: SegmentedTS(K, rng, f, n_levels, borrow=BORROW),
        }
        per_group = pd.DataFrame({"rows_in_log": df[f].value_counts().sort_index()})
        per_group["random_ctr"] = ctr_per_group(random_runs, groups)["ctr"]
        per_group["pooled_ctr"] = ctr_per_group(pooled_runs, groups)["ctr"]
        for label, make in variants.items():
            runs = run_many(make, df, n_runs=N_TUNE, ctx_cols=(f,))
            s = summarise(runs)
            s.update(feature=f, policy=label)
            overview.append(s)
            short = "from_zero" if "zero" in label else "borrow"
            tab = ctr_per_group(runs, groups)
            per_group[f"{short}_ctr"] = tab["ctr"]
            per_group[f"{short}_matched"] = tab["matched"]
        # lift = CTR of the policy in this group divided by Random's CTR in the same group
        for col in ["pooled", "from_zero", "borrow"]:
            per_group[f"{col}_lift"] = per_group[f"{col}_ctr"] / per_group["random_ctr"]
        per_group_tables[f] = per_group

    header(f"Table 19 ({name}): One pooled model vs one model per user group, {N_TUNE} runs each")
    cols = ["feature", "policy", "mean_matched", "mean_clicks", "ctr_mean", "ctr_sd_runs", "se_binomial"]
    t19 = pd.DataFrame(overview)[cols].round(5)
    print(t19.to_string(index=False))
    save_table(t19, f"table_19_{tag}", index=False)

    for f, per_group in per_group_tables.items():
        header(f"Table 20 ({name}, {f}): CTR and lift over Random per user group, {N_TUNE} runs")
        print(per_group.round(5).to_string())
        save_table(per_group.round(5), f"table_20_{tag}_{f}")


def section_e(name, df):
    """Question (e): fairness to items (who gets shown) and to users (who gets
    good recommendations), plus the fairness extension (exposure floor)."""
    tag = name.replace(" ", "")
    K = df["item_id"].nunique()
    f = FAIR_FEATURE
    n_levels = df[f].max() + 1
    groups = df[f].to_numpy()
    true_ctr = df.groupby("item_id")["click"].mean().to_numpy()

    policies = {
        "Random": lambda rng: RandomPolicy(K, rng),
        "TS": lambda rng: ThompsonSampling(K, rng),
        f"UCB c={UCB_C_FOR_B}": lambda rng: UCB(K, rng, c=UCB_C_FOR_B),
    }
    # The floor is put on UCB: it is the policy that concentrates exposure on a
    # few items (TS already spreads its shows through its random draws).
    for fl in FAIR_FLOORS:
        policies[f"UCB c={UCB_C_FOR_B} + exposure floor {fl}"] = (
            lambda fl: lambda rng: ExposureFloor(UCB(K, rng, c=UCB_C_FOR_B), rng, floor=fl))(fl)
    policies[f"Per group TS ({f}), borrowing {BORROW}"] = \
        lambda rng: SegmentedTS(K, rng, f, n_levels, borrow=BORROW)

    runs = {p: run_many(make, df, n_runs=N_TUNE, ctx_cols=(f,)) for p, make in policies.items()}

    # fairness to users: CTR per group, divided by Random's CTR in the same group
    random_groups = ctr_per_group(runs["Random"], groups)
    reliable = random_groups.index[random_groups["matched"] >= MIN_MATCHED]
    per_group = pd.DataFrame({"rows_in_log": df[f].value_counts().sort_index(),
                              "random_ctr": random_groups["ctr"]}).loc[reliable]
    summary = []
    for p, r in runs.items():
        s = summarise(r)
        s["policy"] = p
        s.update(fair.item_fairness(r, true_ctr))
        g = ctr_per_group(r, groups).loc[reliable]
        lift = g["ctr"] / random_groups.loc[reliable, "ctr"]
        if p != "Random":
            per_group[f"{p} lift"] = lift
        s["worst_group_lift"], s["best_group_lift"] = lift.min(), lift.max()
        s["group_lift_gap"] = lift.max() - lift.min()
        summary.append(s)

    header(f"Table 21 ({name}): Clicks against fairness to items and to users, {N_TUNE} runs each")
    cols = ["policy", "ctr_mean", "se_binomial", "gini_exposure", "top3_share", "least_shown_item_share",
            "items_below_half_fair_share", "spearman_exposure_vs_ctr",
            "worst_group_lift", "best_group_lift", "group_lift_gap"]
    t21 = pd.DataFrame(summary)[cols].round(4)
    print(t21.to_string(index=False))
    print(f"Group lifts use {f}; groups with fewer than {MIN_MATCHED} matched rows per run are left out:")
    print(f"they are too small to judge (that is itself a fairness problem).")
    save_table(t21, f"table_21_{tag}", index=False)

    header(f"Table 22 ({name}): Lift over Random per user group ({f}), {N_TUNE} runs each")
    print(per_group.round(3).to_string())
    left_out = sorted(int(x) for x in set(df[f].unique()) - set(reliable))
    print(f"Groups left out as too small: {left_out}")
    save_table(per_group.round(3), f"table_22_{tag}")

    fig_path = OUT / f"figure_6_fairness_{tag}.png"
    ex.plot_fairness(t21, fig_path)
    print(f"\nFigure 6 ({name}) saved: output/{fig_path.name}")


def section_placement(name, df):
    """Question (a), second part: where to place the items. Does the best item
    depend on the slot (left, centre, right)?"""
    tag = name.replace(" ", "")
    K = df["item_id"].nunique()
    n_levels = df["position"].max() + 1

    # 1. Item CTR per position: is the ranking the same in every slot?
    by_pos = df.groupby(["item_id", "position"])["click"].mean().unstack()
    top = pd.DataFrame({f"position {p}": by_pos[p].sort_values(ascending=False).index[:5].to_numpy()
                        for p in by_pos.columns}, index=[f"rank {i}" for i in range(1, 6)])
    header(f"Table 10 ({name}): Five best items in each position (left = 1, centre = 2, right = 3)")
    print(top.to_string())
    save_table(top, f"table_10_{tag}")

    r = st.interaction_lr_test(df, df["position"].to_numpy())
    placebo = st.position_permutation(df, lambda g: st.interaction_lr_test(df, g)["LR_statistic"], N_PERM)
    r.update(placebo_mean=placebo.mean(), placebo_95th_percentile=np.percentile(placebo, 95),
             permutation_p_value=(placebo >= r["LR_statistic"]).mean())
    header(f"Table 11 ({name}): Likelihood ratio test, does the item ranking differ by position?")
    t11 = pd.Series(r).to_frame("value")
    print(t11.to_string())
    print("Placebo: within each hour the position labels are swapped between slots at random.")
    save_table(t11, f"table_11_{tag}")

    # 2. One model for all slots vs one model per slot
    policies = {
        "TS, one model for all positions": lambda rng: ThompsonSampling(K, rng),
        "TS per position, from zero": lambda rng: SegmentedTS(K, rng, "position", n_levels, borrow=0),
        f"TS per position, borrowing {BORROW}": lambda rng: SegmentedTS(K, rng, "position", n_levels, borrow=BORROW),
    }
    rows = []
    for label, make in policies.items():
        s = summarise(run_many(make, df, n_runs=N_TUNE))
        s["policy"] = label
        rows.append(s)
    header(f"Table 12 ({name}): One model for all positions vs one model per position, {N_TUNE} runs")
    t12 = pd.DataFrame(rows)[["policy", "mean_matched", "mean_clicks", "ctr_mean", "ctr_sd_runs", "se_binomial"]].round(5)
    print(t12.to_string(index=False))
    save_table(t12, f"table_12_{tag}", index=False)


def two_proportion_p(clicks_a, n_a, clicks_b, n_b):
    """Two sided p value of a two proportion z test (is CTR a different from CTR b?)."""
    pa, pb = clicks_a / n_a, clicks_b / n_b
    pooled = (clicks_a + clicks_b) / (n_a + n_b)
    se = np.sqrt(pooled * (1 - pooled) * (1 / n_a + 1 / n_b))
    return 2 * stats.norm.sf(abs(pa - pb) / se)


def section_final(name, df):
    """Question (a), final comparison: the default and the tuned versions side by
    side, with regret against the Oracle and a test against Random.
    The tuned settings come from question (c), so they are chosen in hindsight."""
    tag = name.replace(" ", "")
    K = df["item_id"].nunique()
    best = df.groupby("item_id")["click"].mean().idxmax()
    policies = {
        "Random (logging policy)": lambda rng: RandomPolicy(K, rng),
        "TS default Beta(1,1)": lambda rng: ThompsonSampling(K, rng),
        "TS informed prior Beta(1,300)": lambda rng: ThompsonSampling(K, rng, prior_a=1, prior_b=300),
        "UCB default c=1": lambda rng: UCB(K, rng, c=1),
        f"UCB tuned c={UCB_C_FOR_B}": lambda rng: UCB(K, rng, c=UCB_C_FOR_B),
    }
    if user_cols(df):
        n_levels = df[FAIR_FEATURE].max() + 1
        policies[f"Per group TS ({FAIR_FEATURE}), borrowing"] = \
            lambda rng: SegmentedTS(K, rng, FAIR_FEATURE, n_levels, borrow=BORROW)
    policies["Oracle (best item in hindsight)"] = lambda rng: FixedPolicy(K, rng, best)

    ctx = (FAIR_FEATURE,) if user_cols(df) else ()
    rows = []
    for label, make in policies.items():
        n = 1 if label.startswith("Oracle") else N_RUNS
        s = summarise(run_many(make, df, n_runs=n, ctx_cols=ctx))
        s["policy"] = label
        rows.append(s)
    t = pd.DataFrame(rows)
    rnd = t.iloc[0]
    oracle_ctr = t.iloc[-1]["ctr_mean"]
    t["lift_vs_random_percent"] = 100 * (t["ctr_mean"] / rnd["ctr_mean"] - 1)
    # regret: clicks lost against always showing the best item, per 10,000 matched rows
    t["regret_per_10000_rows"] = 10000 * (oracle_ctr - t["ctr_mean"])
    t["p_value_vs_random"] = [two_proportion_p(r.mean_clicks, r.mean_matched, rnd.mean_clicks, rnd.mean_matched)
                              for r in t.itertuples()]
    header(f"Table 13 ({name}): Final comparison of default and tuned policies, {N_RUNS} runs")
    cols = ["policy", "mean_matched", "mean_clicks", "ctr_mean", "se_binomial",
            "lift_vs_random_percent", "regret_per_10000_rows", "p_value_vs_random"]
    t13 = t[cols].round(5)
    print(t13.to_string(index=False))
    print("p values compare one run worth of matched rows with Random (two proportion z test).")
    print("Tuned settings come from question (c) and were chosen in hindsight on the same log.")
    save_table(t13, f"table_13_{tag}", index=False)


def make_shifted_log(df):
    """Same log, but halfway through the week the item ranking is turned upside
    down: in the second half, the rows of the worst item are relabelled as the
    best item, the second worst as the second best, and so on. The logging was
    uniform, so replay stays valid. This creates a sudden change in which item
    is best, the situation where forgetting old data should pay off."""
    ranked = df.groupby("item_id")["click"].mean().sort_values(ascending=False).index.to_numpy()
    flip = dict(zip(ranked, ranked[::-1]))     # best <-> worst, 2nd best <-> 2nd worst, ...
    switch_time = df["timestamp"].min() + (df["timestamp"].max() - df["timestamp"].min()) / 2
    shifted = df.copy()
    second_half = shifted["timestamp"] >= switch_time
    shifted.loc[second_half, "item_id"] = shifted.loc[second_half, "item_id"].map(flip)
    switch_row = int(second_half.to_numpy().argmax())
    return shifted, switch_row


def section_g(name, df):
    """Question (g) and extension: policies that forget, on the real log and on
    a log where the best item changes halfway."""
    tag = name.replace(" ", "")
    K = df["item_id"].nunique()
    policies = {"TS (never forgets)": lambda rng: ThompsonSampling(K, rng)}
    for g in GAMMAS:
        policies[f"Discounted TS gamma={g}"] = (lambda g: lambda rng: DiscountedTS(K, rng, gamma=g))(g)
    policies[f"UCB c={UCB_C_FOR_B} (never forgets)"] = lambda rng: UCB(K, rng, c=UCB_C_FOR_B)
    for w in WINDOWS:
        policies[f"Sliding window UCB W={w}"] = (lambda w: lambda rng: SlidingWindowUCB(K, rng, c=UCB_C_FOR_B, window=w))(w)

    shifted, switch_row = make_shifted_log(df)
    cols = ["policy", "mean_matched", "mean_clicks", "ctr_mean", "ctr_sd_runs", "se_binomial"]
    curves = {}
    for table_no, label, data in [(26, "real log", df), (27, "log with the ranking flipped halfway", shifted)]:
        rows = []
        for pname, make in policies.items():
            runs = run_many(make, data, n_runs=N_TUNE)
            s = summarise(runs)
            s["policy"] = pname
            # CTR in the first and second half of the log. On the flipped log the
            # second half starts at the switch, so comparing the two tables shows
            # how much the flip cost each policy.
            before = [np.mean(np.array(r["matched_clicks"])[r["matched_rows"] < switch_row]) for r in runs]
            after = [np.mean(np.array(r["matched_clicks"])[r["matched_rows"] >= switch_row]) for r in runs]
            s["ctr_first_half"], s["ctr_second_half"] = np.mean(before), np.mean(after)
            if table_no == 27:
                curves[pname] = runs
            rows.append(s)
        t = pd.DataFrame(rows)[cols + ["ctr_first_half", "ctr_second_half"]].round(5)
        if table_no == 26:
            second_half_real = t.set_index("policy")["ctr_second_half"]
        else:
            # cost of the flip = second half CTR on the real log minus on the flipped log
            t["cost_of_flip"] = (t["policy"].map(second_half_real) - t["ctr_second_half"]).round(5)
        header(f"Table {table_no} ({name}): Forgetting policies on the {label}, {N_TUNE} runs each")
        print(t.to_string(index=False))
        save_table(t, f"table_{table_no}_{tag}", index=False)

    mid_gamma, mid_window = GAMMAS[len(GAMMAS) // 2], WINDOWS[len(WINDOWS) // 2]
    keep = ["TS (never forgets)", f"Discounted TS gamma={mid_gamma}",
            f"UCB c={UCB_C_FOR_B} (never forgets)", f"Sliding window UCB W={mid_window}"]
    fig_path = OUT / f"figure_8_shift_{tag}.png"
    ex.plot_shift({k: curves[k] for k in keep}, switch_row, fig_path)
    print(f"\nFigure 8 ({name}) saved: output/{fig_path.name}")


def clean_output():
    """Remove tables and figures of earlier runs, so the output folder holds
    only what this full run produces."""
    for f in list((OUT / "tables").glob("*")) + list(OUT.glob("figure_*.png")):
        f.unlink()


if __name__ == "__main__":
    # Full run (for the report):   python main.py
    # Quick test on one file:       python main.py 10
    # One part only, quick:         python main.py 10 b
    # Parts, in report order: a (comparison), p (placement), h (head to head),
    # b (batch size), c (tuning), d (user groups), e (fairness),
    # f (stationarity), g (forgetting policies and extension)
    order = ["a", "p", "h", "b", "c", "d", "e", "f", "g"]
    args = sys.argv[1:]
    files = [a for a in args if a.isdigit()]
    sections = [a for a in args if not a.isdigit()] or order
    quick = bool(args)
    if not quick:
        clean_output()
    # The full run writes output/run_log.txt, a quick run writes run_log_quick.txt,
    # so a quick test never overwrites the log of the full run.
    start_log("run_log_quick.txt" if quick else "run_log.txt")

    if files:
        dfs = {f"{k} items": load(int(k)) for k in files}
    else:
        dfs = {f"{k} items": load(k) for k in (10, 20, 80)}
        section_data(dfs)          # Tables 1 to 6 use all three files
        del dfs["20 items"]        # the 20 item file is only described, not replayed

    run = {
        "a": section_a, "p": section_placement, "h": section_final, "b": section_b,
        "c": section_c, "d": section_d, "e": section_e, "f": section_f, "g": section_g,
    }
    needs_features = {"d", "e"}   # the 10 item file has no user features
    only_full_file = {"f"}        # stationarity is about the data, so use the 80 item file

    if "a" in sections:
        section_sanity(dfs)
    for key in order:
        if key not in sections:
            continue
        for name, df in dfs.items():
            if key in needs_features and not user_cols(df):
                continue
            if key in only_full_file and name != "80 items":
                continue
            run[key](name, df)
