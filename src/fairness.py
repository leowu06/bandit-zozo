"""Fairness measures for question (e)."""
import numpy as np
from scipy import stats


def gini(shares):
    """Gini coefficient of exposure. 0 means every item was shown equally
    often, values near 1 mean almost all exposure went to one item."""
    x = np.sort(np.asarray(shares, dtype=float))
    n = len(x)
    if x.sum() == 0:
        return np.nan
    ranks = np.arange(1, n + 1)
    return (2 * np.sum(ranks * x)) / (n * x.sum()) - (n + 1) / n


def item_fairness(runs, true_ctr):
    """Exposure measures, averaged over runs. `true_ctr` is each item's CTR in
    the whole log, used as its merit."""
    shown = np.mean([r["shown"] for r in runs], axis=0)
    share = shown / shown.sum()
    K = len(share)
    return {
        "gini_exposure": gini(share),
        "top3_share": np.sort(share)[-3:].sum(),
        "least_shown_item_share": share.min(),
        "items_below_half_fair_share": int((share < 0.5 / K).sum()),
        # do better items get more exposure? 1 = exposure perfectly ordered by merit
        "spearman_exposure_vs_ctr": stats.spearmanr(share, true_ctr).statistic,
    }
