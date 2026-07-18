"""Inference helpers: odds conversion + the combined `predict_fight` call.

This is the deployment-facing surface (ported from notebook 03 Part 2 §6):
load the two saved bundles once, then feed `predict_fight` one row of the
processed feature table to get win probability, both display odds, and the
finish-method distribution.
"""
import joblib
import numpy as np
import pandas as pd

from .config import FINISH_MODEL_PATH, WIN_MODEL_PATH


def decimal_odds(p):
    """Decimal odds = total return per unit staked."""
    return 1.0 / np.asarray(p, dtype=float)


def american_odds(p):
    """American odds: negative for favorites (p >= 0.5), positive for underdogs."""
    p = np.asarray(p, dtype=float)
    fav = p >= 0.5
    out = np.where(fav, -100 * p / (1 - p), 100 * (1 - p) / p)
    return np.round(out).astype(int)


def load_models() -> tuple[dict, dict]:
    """Load (win_bundle, finish_bundle) from models/."""
    return joblib.load(WIN_MODEL_PATH), joblib.load(FINISH_MODEL_PATH)


def predict_fight(row: pd.Series, win_bundle: dict | None = None,
                  finish_bundle: dict | None = None) -> dict:
    """row: one fight from the processed feature table (needs debut_a/b derivable).

    Returns win probability, favorite's display odds, and the finish-method
    distribution — the full output the serving app quotes.
    """
    if win_bundle is None or finish_bundle is None:
        win_bundle, finish_bundle = load_models()

    # win probability (calibrated LR on symmetric diffs + debut flags).
    # The raw LR is antisymmetric in the diffs, but isotonic calibration is fit
    # on finite data and is NOT — so we score both corner orderings and average,
    # which restores exact corner-invariance.
    row = row.copy()
    row["debut_a"] = int(row["is_debut_a"])
    row["debut_b"] = int(row["is_debut_b"])
    xw = row[win_bundle["features"]].to_frame().T.astype(float)
    med = win_bundle["medians"]
    xw[win_bundle["median_impute"]] = xw[win_bundle["median_impute"]].fillna(med)
    xw = xw.fillna(0)

    xw_swap = xw.copy()  # the same bout with the corners exchanged
    diff_cols = [c for c in win_bundle["features"] if c.endswith("_diff")]
    xw_swap[diff_cols] = -xw_swap[diff_cols]
    xw_swap[["debut_a", "debut_b"]] = xw[["debut_b", "debut_a"]].to_numpy()

    p_fwd = float(win_bundle["model"].predict_proba(xw)[0, 1])
    p_rev = float(win_bundle["model"].predict_proba(xw_swap)[0, 1])
    p_a = (p_fwd + (1.0 - p_rev)) / 2.0
    fav_p = max(p_a, 1 - p_a)

    # finish-method distribution (corner-independent profile + division)
    xf = pd.DataFrame(0.0, index=[0], columns=finish_bundle["features"])
    for col in finish_bundle["num_features"]:
        if col in row.index and pd.notna(row[col]):
            xf.at[0, col] = row[col]
    for col, m in finish_bundle["age_medians"].items():  # missing age -> training median
        if col not in row.index or pd.isna(row[col]):
            xf.at[0, col] = m
    divcol = f"div_{row['division']}"
    if divcol in xf.columns:
        xf.at[0, divcol] = 1.0
    proba = finish_bundle["model"].predict_proba(xf)[0]
    labels = (finish_bundle["classes"] if finish_bundle["is_xgb"]
              else list(finish_bundle["model"].classes_))
    fm = dict(zip(labels, proba))

    return {
        "p_fighter_a": round(p_a, 3),
        "favorite": "A" if p_a >= 0.5 else "B",
        "fav_win_prob": round(fav_p, 3),
        "fav_american_odds": int(american_odds(fav_p)),
        "fav_decimal_odds": round(float(decimal_odds(fav_p)), 2),
        "finish_method": {k: round(float(v), 3)
                          for k, v in sorted(fm.items(), key=lambda x: -x[1])},
    }
