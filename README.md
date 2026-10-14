# bandit-zozo

Thompson Sampling and UCB recommenders tested on logged data from ZOZOTOWN, a Japanese fashion site. Written for Assignment 2 of Learning from Big Data (Erasmus University Rotterdam, 2026).

## The problem

ZOZOTOWN shows each visitor three items in a row (left, centre, right). Out of about 80 candidates, which ones should go in those slots to get the most clicks? I treat this as a multi-armed bandit: each item is an arm, and a click is the reward.

I can't show anything to real users, so I evaluate offline with the replay method (Li et al., 2011). The log is stepped through in time order. A row counts only when my policy picks the same item that was actually shown. This is unbiased here because every logged item was chosen uniformly at random (propensity 1/80 on every row).

## Data

The data is a sample of the Open Bandit Dataset (Saito et al., 2021). The files are not included in this folder because of their size. To run the code, put them in `data/`:

| File | Rows | Items | CTR |
|---|---|---|---|
| `zozo_Context_80items.csv.zip` | 1,356,670 | 79 | 0.35% |
| `zozo_Context_20items.csv` | 345,469 | 20 | 0.25% |
| `zozo_noContext_10items.csv` | 172,400 | 10 | 0.28% |

Columns: `timestamp`, `item_id`, `position` (1 = left, 2 = centre, 3 = right), `click`, `propensity_score` and `user_feature_0` to `user_feature_3` (hashed user attributes). The 10 item file has no user features. The zip also holds a `__MACOSX` file, so the loader opens the CSV inside it by name.

## What the code answers

| Question | What we do | Tables | Figures |
|---|---|---|---|
| Data | descriptives, propensity check, CTR by item, position, user group and day | 1 to 6 | 1, 2 |
| (a) | TS and UCB against Random and the hindsight Oracle; placement by position; final comparison with regret and hour-block bootstrap intervals | 7 to 13, 13b | 3 |
| (b) | learning in batches of 1 to 5,000 matched rows | 14 | 4 |
| (c) | TS prior and UCB bonus scale; settings chosen on days 1 and 2 and scored on days 3 to 7; replay in time order vs shuffled | 15 to 18 | 5 |
| (d) | one pooled model vs one model per user group, with and without borrowing from the pooled model | 19, 20 | |
| (e) | fairness to items (exposure, Gini) and to users (lift per group, all four user features); UCB with a minimum exposure guarantee | 21, 22 | 6 |
| (f) | daily CTR, item ranking across the week, likelihood ratio test, all checked against an hour block permutation placebo | 23 to 25 | 7 |
| (g) | discounted TS and sliding window UCB on the real log and on a log where the ranking flips halfway | 26, 27 | 8 |

The main results use the 80 item file. The 10 item file is the easier setting (fewer items to learn about) and is run next to it. The 20 item file is only described.

## Layout

```
src/data.py          loader
src/replay.py        replay evaluator, Random and fixed item policies
src/policies.py      TS, UCB, discounted TS, sliding window UCB, per group TS, exposure floor
src/explore.py       descriptive tables and all figures
src/stationarity.py  tests for (f) and the placement test
src/fairness.py      exposure measures for (e)
src/bootstrap.py     hour-block bootstrap for the lifts in Table 13b
src/report_io.py     saves every table as csv and tex, and the console log
main.py              runs everything in report order
peek.py              small tool to look at the data and the replay step by step
data/                input files (not included)
output/              tables, figures and the log of the full run, as written by main.py
```

## Running it

```
pip install -r requirements.txt
python main.py           # full run, about 2.5 hours, writes output/run_log.txt
python main.py 10 b      # one part on one file, for quick checks (log in output/run_log_quick.txt)
```

On a Mac, `caffeinate -i python main.py` keeps the computer awake during the full run. The full run first clears old tables and figures from `output/`.

The seed is fixed at 1 and every replay run uses its own seed, so every run gives the same numbers. The bootstrap for Table 13b runs on 4 worker processes, and each resample has its own fixed seed, so its results do not depend on the number of cores. Each table and figure prints with an ID (Table 8, Figure 3 and so on), and the report cites these IDs as "code Table 8".

## References

- Li, L., Chu, W., Langford, J. and Wang, X. (2011). Unbiased offline evaluation of contextual bandit based news article recommendation algorithms. WSDM.
- Saito, Y., Aihara, S., Matsutani, M. and Narita, Y. (2021). Open Bandit Dataset and Pipeline: Towards Realistic and Reproducible Off-Policy Evaluation. NeurIPS Datasets and Benchmarks.
