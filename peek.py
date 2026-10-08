"""Look at the data and the replay loop step by step in the terminal.

    python peek.py raw       first rows of the original file, untouched
    python peek.py clean     the same rows after load() cleaned them
    python peek.py codes     how the long user codes became small numbers
    python peek.py items     how many times each item was shown and clicked
    python peek.py replay    watch the replay loop row by row (Random policy)
    python peek.py ts        watch Thompson Sampling's beliefs change as it learns
"""
import sys

import numpy as np
import pandas as pd

from src.data import load, DATA_DIR, FILES

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)
pd.set_option("display.max_colwidth", 14)

N_ITEMS = 10  # small file, so it loads fast. Change to 80 for the main file.
what = sys.argv[1] if len(sys.argv) > 1 else "raw"

if what == "raw":
    raw = pd.read_csv(DATA_DIR / FILES[N_ITEMS], nrows=8)
    print(f"First 8 rows of {FILES[N_ITEMS]}, exactly as stored on disk:\n")
    print(raw.to_string(index=False))

elif what == "clean":
    df = load(N_ITEMS)
    print(f"First 8 rows after load({N_ITEMS}):\n")
    print(df.head(8).to_string(index=False))
    print(f"\nTotal rows: {len(df):,}")

elif what == "codes":
    raw = pd.read_csv(DATA_DIR / FILES[20], usecols=["user_feature_0"])
    df = load(20)
    pairs = pd.DataFrame({"original": raw["user_feature_0"].sort_values().values}).drop_duplicates()
    pairs["becomes"] = range(len(pairs))
    print("user_feature_0 in the 20 item file. Every distinct long code gets one small number:\n")
    print(pairs.to_string(index=False))
    print("\nRows per code after cleaning:")
    print(df["user_feature_0"].value_counts().sort_index().to_string())

elif what == "items":
    df = load(N_ITEMS)
    t = df.groupby("item_id")["click"].agg(times_shown="size", clicks="sum")
    t["ctr_percent"] = (t["clicks"] / t["times_shown"] * 100).round(3)
    print(f"Each item in the {N_ITEMS} item file:\n")
    print(t.to_string())

elif what == "replay":
    df = load(N_ITEMS)
    rng = np.random.default_rng(1)
    matches = clicks = 0
    print("Our policy here is Random. At each row it picks an item, then we compare.\n")
    print(f"{'row':>4} {'logged item':>12} {'we picked':>10} {'match?':>7} {'click':>6}   running total")
    for t in range(40):
        logged = df["item_id"].iat[t]
        click = df["click"].iat[t]
        pick = int(rng.integers(N_ITEMS))
        if pick == logged:
            matches += 1
            clicks += click
            note = f"counted: {matches} matches, {clicks} clicks"
            print(f"{t:>4} {logged:>12} {pick:>10} {'YES':>7} {click:>6}   {note}")
        else:
            print(f"{t:>4} {logged:>12} {pick:>10} {'no':>7} {'?':>6}   skipped")
    print("\nOnly YES rows count. On the skipped rows we never learn what our pick would have done.")

elif what == "ts":
    from src.policies import ThompsonSampling
    df = load(N_ITEMS)
    true_ctr = df.groupby("item_id")["click"].mean()
    ts = ThompsonSampling(N_ITEMS, np.random.default_rng(1))
    shown = np.zeros(N_ITEMS, dtype=int)
    items, clicks, pos = df["item_id"].to_numpy(), df["click"].to_numpy(), df["position"].to_numpy()
    checkpoints = {10, 100, 1000, 5000, 17000}
    n = 0
    for t in range(len(df)):
        a = ts.select({"position": pos[t]})
        if a != items[t]:
            continue
        ts.update(a, int(clicks[t]), None)
        shown[a] += 1
        n += 1
        if n in checkpoints:
            snap = pd.DataFrame({
                "true_ctr_%": (true_ctr * 100).round(3),
                "clicks(alpha-1)": (ts.alpha - 1).astype(int),
                "no_clicks(beta-1)": (ts.beta - 1).astype(int),
                "belief_mean_%": (ts.alpha / (ts.alpha + ts.beta) * 100).round(3),
                "times_shown_so_far": shown,
            })
            print(f"\n=== After {n} matched rows ===")
            print(snap.sort_values("true_ctr_%", ascending=False).to_string())
    print("\nbelief_mean = alpha / (alpha + beta), TS's current best guess of each item's CTR.")

else:
    print(__doc__)
