"""Learning policies. Each one plugs into replay() like Random and Oracle do."""
import numpy as np

from .replay import Policy


class ThompsonSampling(Policy):
    """Bernoulli Thompson Sampling with a Beta prior per item.

    Belief about item i's click rate: Beta(alpha_i, beta_i)
        alpha_i = prior_a + clicks on i
        beta_i  = prior_b + non clicks on i
    Each round: draw one plausible click rate per item from its Beta,
    show the item with the highest draw, then add the result to that
    item's counters. Uncertain items sometimes draw high, which is how
    the policy keeps exploring without a separate exploration rule.
    """

    name = "TS"

    def __init__(self, n_items, rng, prior_a=1.0, prior_b=1.0):
        super().__init__(n_items, rng)
        self.alpha = np.full(n_items, prior_a, dtype=float)
        self.beta = np.full(n_items, prior_b, dtype=float)

    def select(self, ctx):
        draws = self.rng.beta(self.alpha, self.beta)
        return int(np.argmax(draws))

    def update(self, item, reward, ctx):
        self.alpha[item] += reward
        self.beta[item] += 1 - reward
