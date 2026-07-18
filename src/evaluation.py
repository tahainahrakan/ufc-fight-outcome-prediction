"""Scoring & weekly metrics logging.

`ece` is the size-weighted expected calibration error used in notebook 03 §10 —
the honesty metric: how far stated probabilities drift from observed win rates.
Every weekly retrain appends one row to models/metrics_history.csv so drift is
visible over time.
"""
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

from .config import METRICS_HISTORY_CSV


def scores(y, p) -> dict:
    """Binary-model scores on the holdout: log-loss, Brier, accuracy, ECE."""
    return {"log_loss": round(log_loss(y, p), 4),
            "brier": round(brier_score_loss(y, p), 4),
            "accuracy": round(accuracy_score(y, p > 0.5), 4),
            "ece": round(ece(y, p), 4)}


def ece(y, p, n_bins: int = 10) -> float:
    """Size-weighted expected calibration error over quantile bins."""
    y, p = np.asarray(y), np.asarray(p)
    edges = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    b = np.digitize(p, edges[1:-1])
    total = 0.0
    for k in np.unique(b):
        m = b == k
        total += m.mean() * abs(y[m].mean() - p[m].mean())
    return total


def append_metrics(row: dict) -> None:
    """Append one weekly-run row to the metrics history CSV (header on first write)."""
    METRICS_HISTORY_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(METRICS_HISTORY_CSV, mode="a",
                               header=not METRICS_HISTORY_CSV.exists(), index=False)
