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


class UCB(Policy):
    """UCB1: show the item with the highest optimistic score.

    score = average CTR so far + c * sqrt(2 * ln(total shows) / shows of item)

    The first part rewards items that have done well. The second part is a
    bonus for items we have not seen much of: it is large when an item has few
    shows and shrinks as it gets more. Items never shown are tried first.
    c scales the bonus. c = 1 is the textbook version.
    """

    name = "UCB"

    def __init__(self, n_items, rng, c=1.0):
        super().__init__(n_items, rng)
        self.c = c
        self.shows = np.zeros(n_items)
        self.clicks = np.zeros(n_items)
        self.total_shows = 0

    def select(self, ctx):
        # Try every item once before using the formula,
        # because the bonus divides by the number of shows.
        never_shown = np.flatnonzero(self.shows == 0)
        if len(never_shown) > 0:
            return int(self.rng.choice(never_shown))

        average = self.clicks / self.shows
        bonus = self.c * np.sqrt(2 * np.log(self.total_shows) / self.shows)
        score = average + bonus

        # If several items share the top score, pick one of them at random.
        best = np.flatnonzero(score == score.max())
        return int(self.rng.choice(best))

    def update(self, item, reward, ctx):
        self.shows[item] += 1
        self.clicks[item] += reward
        self.total_shows += 1
