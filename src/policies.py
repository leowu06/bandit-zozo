"""Learning policies. Each one plugs into replay() like Random and Oracle do."""
from collections import deque

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


class DiscountedTS(Policy):
    """Thompson Sampling that slowly forgets old results (Raj and Kalyani, 2017).

    After every update all click and non click counts are multiplied by gamma
    (a number just below 1), then the new result is added. Old results fade,
    so the beliefs follow the recent past. A result that is n updates old
    still carries weight gamma ** n, which gives a memory of roughly
    1 / (1 - gamma) updates. gamma = 1 is plain Thompson Sampling.
    """

    def __init__(self, n_items, rng, gamma=0.999, prior_a=1.0, prior_b=1.0):
        super().__init__(n_items, rng)
        self.gamma = gamma
        self.prior_a, self.prior_b = prior_a, prior_b
        self.clicks = np.zeros(n_items)
        self.non_clicks = np.zeros(n_items)
        self.name = f"Discounted TS gamma={gamma}"

    def select(self, ctx):
        draws = self.rng.beta(self.prior_a + self.clicks, self.prior_b + self.non_clicks)
        return int(np.argmax(draws))

    def update(self, item, reward, ctx):
        self.clicks *= self.gamma
        self.non_clicks *= self.gamma
        self.clicks[item] += reward
        self.non_clicks[item] += 1 - reward


class SlidingWindowUCB(Policy):
    """UCB that only looks at the last `window` results (Garivier and Moulines, 2011).

    Same score as UCB, but averages and counts use only the most recent
    `window` matched rows. Anything older is dropped completely.
    """

    def __init__(self, n_items, rng, c=0.01, window=2000):
        super().__init__(n_items, rng)
        self.c = c
        self.window = window
        self.history = deque()
        self.shows = np.zeros(n_items)
        self.clicks = np.zeros(n_items)
        self.name = f"Sliding window UCB W={window}"

    def select(self, ctx):
        never_shown = np.flatnonzero(self.shows == 0)
        if len(never_shown) > 0:
            return int(self.rng.choice(never_shown))
        average = self.clicks / self.shows
        bonus = self.c * np.sqrt(2 * np.log(len(self.history)) / self.shows)
        score = average + bonus
        best = np.flatnonzero(score == score.max())
        return int(self.rng.choice(best))

    def update(self, item, reward, ctx):
        self.history.append((item, reward))
        self.shows[item] += 1
        self.clicks[item] += reward
        if len(self.history) > self.window:
            old_item, old_reward = self.history.popleft()
            self.shows[old_item] -= 1
            self.clicks[old_item] -= old_reward


class SegmentedTS(Policy):
    """Thompson Sampling with one model per user group.

    The group is the value of one user feature (for example user_feature_0).
    Every group keeps its own click and non click counts per item.

    borrow = 0: each group starts from Beta(1, 1), so a small group knows
    nothing and learns only from its own few rows.

    borrow > 0: each group starts from what the pooled model (all groups
    together) has learned so far, counted as up to `borrow` shows of evidence.
    A small group then leans on the crowd, while a large group soon outweighs
    the borrowed evidence with its own data. This is shrinkage towards the
    pooled estimate.
    """

    def __init__(self, n_items, rng, feature, n_levels, borrow=0):
        super().__init__(n_items, rng)
        self.feature = feature
        self.borrow = borrow
        self.group_clicks = np.zeros((n_levels, n_items))
        self.group_shows = np.zeros((n_levels, n_items))
        self.pooled_clicks = np.zeros(n_items)
        self.pooled_shows = np.zeros(n_items)
        self.name = f"Per group TS ({feature}, borrow={borrow})"

    def prior(self):
        """Starting belief for any group: flat, or borrowed from the pooled counts."""
        if self.borrow == 0:
            return np.ones(self.K), np.ones(self.K)
        # borrow at most as much evidence as the pooled model really has
        weight = np.minimum(self.borrow, self.pooled_shows)
        pooled_rate = np.divide(self.pooled_clicks, self.pooled_shows,
                                out=np.zeros(self.K), where=self.pooled_shows > 0)
        return 1 + weight * pooled_rate, 1 + weight * (1 - pooled_rate)

    def select(self, ctx):
        g = ctx[self.feature]
        a0, b0 = self.prior()
        clicks = self.group_clicks[g]
        non_clicks = self.group_shows[g] - clicks
        draws = self.rng.beta(a0 + clicks, b0 + non_clicks)
        return int(np.argmax(draws))

    def update(self, item, reward, ctx):
        g = ctx[self.feature]
        self.group_clicks[g, item] += reward
        self.group_shows[g, item] += 1
        self.pooled_clicks[item] += reward
        self.pooled_shows[item] += 1


class ExposureFloor(Policy):
    """Minimum exposure guarantee for items, wrapped around any policy.

    Every item is promised at least `floor` times its fair share of the shows
    so far (the fair share is total shows / number of items, what uniform
    random serving would give). Whenever some item is below that line, the
    wrapper shows the most neglected one; otherwise the base policy chooses.
    The base policy still learns from every result.
    floor = 0 gives back the base policy, floor = 1 forces equal exposure.
    """

    def __init__(self, base, rng, floor=0.5):
        super().__init__(base.K, rng)
        self.base = base
        self.floor = floor
        self.shows = np.zeros(base.K)
        self.name = f"{base.name} + exposure floor {floor}"

    def select(self, ctx):
        total = self.shows.sum()
        if total >= self.K:   # let every item be seen once before the rule applies
            below = np.flatnonzero(self.shows < self.floor * total / self.K)
            if len(below) > 0:
                least = below[self.shows[below] == self.shows[below].min()]
                return int(self.rng.choice(least))
        return self.base.select(ctx)

    def update(self, item, reward, ctx):
        self.base.update(item, reward, ctx)
        self.shows[item] += 1
