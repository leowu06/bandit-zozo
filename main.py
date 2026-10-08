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
from src.policies import ThompsonSampling

random.seed(1)
np.random.seed(1)

OUT = Path(__file__).resolve().parent / "output"
OUT.mkdir(exist_ok=True)
pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 20)

N_RUNS = 10  # replay repetitions for stochastic policies


def header(label):
    print(f"\n--- {label} ---")


def section_data(dfs):
    header("Table 1: Overview of the three data files")
    print(pd.DataFrame([ex.summary_row(k, d) for k, d in dfs.items()]).to_string(index=False))

    df = dfs["80 items"]

    header("Table 2: Propensity scores in the 80 item file")
    print(df["propensity_score"].value_counts().rename("rows").to_frame().to_string())
    print("Every row has propensity 1/80, so the logging policy was uniform random.")

    header("Table 3: CTR per position (80 items)")
    print(ex.ctr_table(df, "position").round(5).to_string())

    items = ex.ctr_table(df, "item_id")
    header("Table 4: Ten highest and ten lowest CTR items (80 items)")
    s = items.sort_values("ctr", ascending=False).round(5)
    print(pd.concat([s.head(10), s.tail(10)]).to_string())
    ex.plot_item_ctr(items, df["click"].mean(), OUT / "figure_1_item_ctr.png")
    print("\nFigure 1 saved: output/figure_1_item_ctr.png")

    header("Table 5: CTR per level of each user feature (80 items)")
    for c in user_cols(df):
        print(f"\n{c}")
        print(ex.ctr_table(df, c).round(5).to_string())

    daily = ex.ctr_table(df.assign(day=df["timestamp"].dt.date), "day")
    header("Table 6: CTR per day (80 items)")
    print(daily.round(5).to_string())
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
            "ctr_lo", "ctr_hi", "se_single_run", "log_ctr"]
    print(pd.DataFrame(rows)[cols].round(5).to_string(index=False))
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
        "Oracle": run_many(lambda rng: FixedPolicy(K, rng, best), df, n_runs=1),
    }

    header(f"Table 8 ({name}): CTR of Thompson Sampling vs Random and Oracle, {N_RUNS} runs")
    rows = [summarise(r) for r in runs.values()]
    cols = ["policy", "runs", "mean_matched", "mean_clicks", "ctr_mean", "ctr_lo", "ctr_hi"]
    print(pd.DataFrame(rows)[cols].round(5).to_string(index=False))

    header(f"Table 9 ({name}): How often TS showed each item vs its true CTR")
    shown = np.mean([r["shown"] for r in runs["TS"]], axis=0)
    t = pd.DataFrame({"true_ctr_percent": (true_ctr * 100).round(3),
                      "ts_share_shown_percent": (shown / shown.sum() * 100).round(1)})
    print(t.sort_values("true_ctr_percent", ascending=False).to_string())

    fig_path = OUT / f"figure_3_ts_learning_{name.split()[0]}items.png"
    ex.plot_learning({k: mean_curve(v) for k, v in runs.items() if k != "Oracle"},
                     summarise(runs["Oracle"])["ctr_mean"], fig_path)
    print(f"\nFigure 3 ({name}) saved: output/{fig_path.name}")


if __name__ == "__main__":
    which = sys.argv[1:] or ["10", "20", "80"]
    dfs = {f"{k} items": load(int(k)) for k in which}
    if "80 items" in dfs:
        section_data(dfs)
    section_sanity(dfs)
    for name, df in dfs.items():
        section_a(name, df)
