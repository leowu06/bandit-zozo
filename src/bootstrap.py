"""Hour-block bootstrap for the main comparison (code Table 13b).

The rows of the log are not independent: the same item often sits in the
same slot for many rows in a row. So we resample whole hours instead of
rows. One resample draws 168 hours with replacement and puts their rows one
after the other in the drawn order, keeping the order inside each hour.
Every policy is then replayed once on that resample and compared with Random
on the same resample. The spread of the lift over the resamples shows how
much it could move with another week of similar data.
"""
from multiprocessing import Pool

import numpy as np

from src.replay import RandomPolicy, run_many
from src.policies import ThompsonSampling, UCB, SegmentedTS

HOUR_SEED = 100000  # resample b draws its hours with seed HOUR_SEED + b

_df = None  # the log, set once in every worker process


def _init(df):
    """Give each worker process the log once, so it is not sent again for
    every resample."""
    global _df
    _df = df


def make_policies(K, feature, n_levels, ts_prior_b, ucb_c, borrow):
    """Policies compared in Table 13b, by label. Random comes first."""
    policies = {
        "Random (logging policy)": lambda rng: RandomPolicy(K, rng),
        f"TS informed prior Beta(1,{ts_prior_b})": lambda rng: ThompsonSampling(K, rng, 1, ts_prior_b),
        f"UCB tuned c={ucb_c}": lambda rng: UCB(K, rng, c=ucb_c),
    }
    if feature is not None:
        policies[f"Per group TS ({feature}), borrowing"] = \
            lambda rng: SegmentedTS(K, rng, feature, n_levels, borrow=borrow)
    return policies


def one_resample(args):
    """Draw one resample of hours, replay every policy once, return the CTRs."""
    b, settings = args
    df = _df
    hour = df["hour"].to_numpy()
    starts = np.flatnonzero(np.r_[True, hour[1:] != hour[:-1]])
    ends = np.r_[starts[1:], len(hour)]
    pick = np.random.default_rng(HOUR_SEED + b).integers(len(starts), size=len(starts))
    rows = np.concatenate([np.arange(starts[h], ends[h]) for h in pick])
    d = df.iloc[rows].reset_index(drop=True)

    ctx = (settings["feature"],) if settings["feature"] is not None else ()
    out = {}
    for label, make in make_policies(**settings).items():
        r = run_many(make, d, n_runs=1, seed=1 + b, ctx_cols=ctx)[0]
        out[label] = r["ctr"]
    return out


def hour_block_bootstrap(df, n_boot, n_cores, **settings):
    """CTR of every policy on n_boot hour-block resamples, one row per resample."""
    cols = ["item_id", "click", "position"] + ([settings["feature"]] if settings["feature"] else [])
    small = df[cols].copy()
    small["hour"] = df["timestamp"].dt.floor("h")
    with Pool(n_cores, initializer=_init, initargs=(small,)) as pool:
        return pool.map(one_resample, [(b, settings) for b in range(n_boot)])
