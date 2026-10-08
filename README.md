# bandit-zozo

Thompson Sampling and UCB recommenders tested on logged data from ZOZOTOWN, a Japanese fashion site. Written for Assignment 2 of Learning from Big Data (Erasmus University Rotterdam, 2026).

## The problem

ZOZOTOWN shows each visitor three items in a row (left, centre, right). Out of about 80 candidates, which ones should go in those slots to get the most clicks? I treat this as a multi-armed bandit: each item is an arm, and a click is the reward.

I can't show anything to real users, so I evaluate offline with the replay method (Li et al., 2011). The log is stepped through in time order. A row counts only when my policy picks the same item that was actually shown. This is unbiased here because every logged item was chosen uniformly at random (propensity 1/80 on every row).

## Data

The data is a sample of the Open Bandit Dataset (Saito et al., 2021). It is not in the repo because the main file is about 270 MB. To run the code, put these files in `data/`:

| File | Rows | Items | CTR |
|---|---|---|---|
| `zozo_Context_80items.csv.zip` | 1,356,670 | 79 | 0.35% |
| `zozo_Context_20items.csv` | 345,469 | 20 | 0.25% |
| `zozo_noContext_10items.csv` | 172,400 | 10 | 0.28% |

Columns: `timestamp`, `item_id`, `position` (1 = left, 2 = centre, 3 = right), `click`, `propensity_score` and `user_feature_0` to `user_feature_3` (hashed user attributes). The 10 item file has no user features. The zip also holds a `__MACOSX` file, so the loader opens the CSV inside it by name.

## What the code answers

- (a) two policies, one built on TS and one on UCB, compared with each other and with the random logging policy
- (b) sensitivity to batch size
- (c) sensitivity to parameter tuning
- (d) one pooled model vs separate models per user group
- (e) fairness to users and to items, plus a fairer variant
- (f) whether click rates are stationary over the 7 days
- (g) how the models would change if they are not

## Layout

```
src/       loader, replay evaluator, policies, fairness checks
data/      input files (gitignored)
output/    tables and figures written by the code (gitignored)
report/    LaTeX report
main.py    runs everything in order
```

## Running it

```
pip install -r requirements.txt
python main.py
```

The seed is fixed at 1, so every run gives the same numbers. Each table and figure prints with an ID (Table 1, Figure 2 and so on) that matches the report.

## References

- Li, L., Chu, W., Langford, J. and Wang, X. (2011). Unbiased offline evaluation of contextual bandit based news article recommendation algorithms. WSDM.
- Saito, Y., Aihara, S., Matsutani, M. and Narita, Y. (2021). Open Bandit Dataset and Pipeline: Towards Realistic and Reproducible Off-Policy Evaluation. NeurIPS Datasets and Benchmarks.
