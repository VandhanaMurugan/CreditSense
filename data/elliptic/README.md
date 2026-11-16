# Elliptic data set (not committed)

Used by `phase2/p2_step18_elliptic_transaction_risk.py` to test the decision layer on crypto transaction screening.

Place here: `elliptic_txs_features.csv` and `elliptic_txs_classes.csv`.

- Source: the Elliptic Data Set (Kaggle: "Elliptic Data Set"), released with
  Weber et al. (2019), "Anti-Money Laundering in Bitcoin: Experimenting with Graph Convolutional Networks for Financial Forensics", arXiv:1908.02591.
- `elliptic_txs_features.csv`: 689,683,771 bytes, SHA-256 `fd7f83573443c9e302e371d3f110e3b6224160f5d1ed8a287757936127800ff0`
- `elliptic_txs_classes.csv`: 3,305,144 bytes, SHA-256 `93e2e7b2405c735ba752bf6ba06b947561deddd1f5a8fc91e46f6a4c0e439493`
- 203,769 transactions in 49 time steps; 46,564 are labelled licit or illicit. Features are anonymised.
- Check the licence and terms on the dataset page before any use beyond research and coursework.
