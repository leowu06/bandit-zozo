"""Assignment 2, Learning from Big Data 2026.

Running this file prints every table and saves every figure used in the
report. Each one carries the ID it has in the report.
"""
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import load, user_cols
from src import explore as ex
from src.replay import RandomPolicy, FixedPolicy, run_many, summarise
from src.policies import ThompsonSampling, UCB
from src.report_io import save_table, start_log

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
    """Question (a), first part: Thompson Sampling against Random and Oracle."""
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

    header(f"Table 10 ({name}): TS with different priors Beta(1, b), {N_TUNE} runs each")
    t10 = ts_tab[["setting", "prior_mean_percent"] + cols].round(5)
    print(t10.to_string(index=False))
    save_table(t10, f"table_10_{tag}", index=False)

    header(f"Table 11 ({name}): UCB with different bonus scales c, {N_TUNE} runs each")
    t11 = ucb_tab[["setting"] + cols].round(5)
    print(t11.to_string(index=False))
    save_table(t11, f"table_11_{tag}", index=False)

    fig_path = OUT / f"figure_4_tuning_{tag}.png"
    ex.plot_tuning(ts_tab, ucb_tab, random_ctr, fig_path)
    print(f"\nFigure 4 ({name}) saved: output/{fig_path.name}")

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
    header(f"Table 12 ({name}): Settings chosen on days 1 and 2, scored on days 3 to 7, {N_RUNS} runs")
    t12 = pd.DataFrame(rows)[["policy"] + cols].round(5)
    print(t12.to_string(index=False))
    save_table(t12, f"table_12_{tag}", index=False)

    # 3. Does the order of the log matter? Same policies on shuffled rows.
    shuffled = df.sample(frac=1, random_state=1).reset_index(drop=True)
    rows = []
    for label, make in [("TS default Beta(1,1)", lambda rng: ThompsonSampling(K, rng)),
                        ("UCB default c=1", lambda rng: UCB(K, rng, c=1)),
                        (f"UCB c={best_c}", lambda rng: UCB(K, rng, c=best_c))]:
        for order, d in [("time order", df), ("shuffled", shuffled)]:
            s = summarise(run_many(make, d, n_runs=N_TUNE))
            s["policy"], s["row_order"] = label, order
            rows.append(s)
    header(f"Table 13 ({name}): Replay in time order vs shuffled rows, {N_TUNE} runs")
    t13 = pd.DataFrame(rows)[["policy", "row_order"] + cols].round(5)
    print(t13.to_string(index=False))
    save_table(t13, f"table_13_{tag}", index=False)


if __name__ == "__main__":
    # Full run (for the report): python main.py
    # Quick test on one file:      python main.py 10
    start_log()
    quick = sys.argv[1:]
    if quick:
        dfs = {f"{k} items": load(int(k)) for k in quick}
    else:
        dfs = {f"{k} items": load(k) for k in (10, 20, 80)}
        section_data(dfs)          # Tables 1 to 6 use all three files
        del dfs["20 items"]        # the 20 item file is only described, not replayed
    section_sanity(dfs)
    for name, df in dfs.items():
        section_a(name, df)
    for name, df in dfs.items():
        section_c(name, df)
