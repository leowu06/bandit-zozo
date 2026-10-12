"""Replay evaluation (Li et al., 2011) and two reference policies.

Every row in the log is one item shown in one slot, chosen uniformly at
random. The replay loop walks through the rows in time order. At each row
the policy picks an item for that slot. If the pick equals the item that was
logged, the click is revealed, the row counts towards the policy's CTR and the
policy gets to learn from it. Otherwise the row is skipped and the policy
learns nothing.

Because the logging policy is uniform, the rows that survive are an unbiased
sample of what would have happened had the policy been live.
"""
import numpy as np


class Policy:
    """Interface every policy follows.

    select(ctx) returns an item index. ctx is a dict with the slot position
    and the user feature codes of that row.
    update(item, reward, ctx) feeds back one revealed click (0 or 1).
    """

    name = "policy"

    def __init__(self, n_items, rng):
        self.K = n_items
        self.rng = rng

    def select(self, ctx):
        raise NotImplementedError

    def update(self, item, reward, ctx):
        pass


class RandomPolicy(Policy):
    """Uniform random choice. Same as the policy that generated the log."""

    name = "Random"

    def select(self, ctx):
        return int(self.rng.integers(self.K))


class FixedPolicy(Policy):
    """Always shows the same item. Used for the hindsight oracle."""

    def __init__(self, n_items, rng, item, name="Oracle"):
        super().__init__(n_items, rng)
        self.item = int(item)
        self.name = name

    def select(self, ctx):
        return self.item


def replay(policy, df, batch_size=1, ctx_cols=()):
    """Run one replay pass of `policy` over `df`.

    batch_size: the policy only receives feedback after this many matched
    rows have piled up, like a website that retrains every so often.
    With batch_size=1 it learns after every match.

    Returns a dict with the overall CTR, the number of matched rows and clicks,
    the cumulative CTR after each match, and how often each item was shown.
    """
    items = df["item_id"].to_numpy()
    clicks = df["click"].to_numpy()
    pos = df["position"].to_numpy()
    ctx_arr = {c: df[c].to_numpy() for c in ctx_cols}

    matched_clicks = []
    matched_rows = []
    shown = np.zeros(policy.K, dtype=int)
    buffer = []

    for t in range(len(items)):
        ctx = {"position": pos[t]}
        for c in ctx_cols:
            ctx[c] = ctx_arr[c][t]

        a = policy.select(ctx)
        if a != items[t]:
            continue

        r = int(clicks[t])
        matched_clicks.append(r)
        matched_rows.append(t)
        shown[a] += 1
        buffer.append((a, r, ctx))
        if len(buffer) >= batch_size:
            for b in buffer:
                policy.update(*b)
            buffer = []

    matched_clicks = np.array(matched_clicks)
    n = len(matched_clicks)
    cum_ctr = np.cumsum(matched_clicks) / np.arange(1, n + 1) if n else np.array([])
    return {
        "policy": policy.name,
        "n_matched": n,
        "clicks": int(matched_clicks.sum()),
        "ctr": matched_clicks.mean() if n else np.nan,
        "cum_ctr": cum_ctr,
        "matched_clicks": matched_clicks,
        "matched_rows": np.array(matched_rows),
        "shown": shown,
    }


def run_many(make_policy, df, n_runs=10, seed=1, **kw):
    """Repeat replay with a different random stream per run.

    make_policy(rng) must return a fresh policy. Run i uses seed + i, so the
    whole set is reproducible.
    """
    runs = []
    for i in range(n_runs):
        rng = np.random.default_rng(seed + i)
        runs.append(replay(make_policy(rng), df, **kw))
    return runs


def summarise(runs):
    """Summary of a set of replay runs.

    ctr_mean      average CTR over the runs
    ctr_sd_runs   how much the CTR moves between runs (policy randomness only,
                  the data stays the same)
    ctr_min_run, ctr_max_run   lowest and highest run
    se_binomial   standard error from the number of matched rows,
                  sqrt(p(1-p)/n). This is the sampling noise of one run and is
                  the better guide to whether two policies really differ.
    """
    ctrs = np.array([r["ctr"] for r in runs])
    n = np.mean([r["n_matched"] for r in runs])
    p = ctrs.mean()
    return {
        "policy": runs[0]["policy"],
        "runs": len(runs),
        "mean_matched": n,
        "mean_clicks": np.mean([r["clicks"] for r in runs]),
        "ctr_mean": p,
        "ctr_sd_runs": ctrs.std(ddof=1) if len(runs) > 1 else 0.0,
        "ctr_min_run": ctrs.min(),
        "ctr_max_run": ctrs.max(),
        "se_binomial": np.sqrt(p * (1 - p) / n) if n else np.nan,
    }
