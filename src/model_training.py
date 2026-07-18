"""Model training — the src/ port of notebook 03 (both parts).

Win-probability model: isotonic-calibrated Logistic Regression (the notebook's
verdict). Each retrain (a) benchmarks on the rolling temporal holdout for the
weekly health metric, then (b) refits on ALL data so the shipped model has
learned from the newest fights. Imputation values are always computed on the
rows the model is fit on — never on future data relative to evaluation.

Finish-method model: LR vs XGBoost head-to-head on the holdout, ship whichever
wins, refit on all data — same policy as the notebook, re-decided every week.
"""
import joblib
import numpy as np
import pandas as pd

from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from .config import (FINISH_CLASSES, FINISH_MODEL_PATH, METHOD_MAP, MODELS_DIR,
                     RNG_SEED, WIN_MODEL_PATH)
from .evaluation import scores

MEDIAN_IMPUTE = ["age_at_fight_diff", "height_in_diff"]  # real-but-missing attributes

# finish-model per-fighter profile (corner-independent — a KO is a KO either way)
PROFILE = ["career_finish_rate", "career_avg_sig_str_landed", "career_avg_sig_str_att",
           "career_avg_takedowns_landed", "career_avg_takedowns_att",
           "career_avg_sub_att", "career_avg_control_time_sec",
           "age_at_fight", "career_fights"]


def _win_features(df: pd.DataFrame) -> list[str]:
    diff_cols = [c for c in df.columns if c.endswith("_diff")]
    return diff_cols + ["debut_a", "debut_b"]


def _prep_win(df: pd.DataFrame, features: list[str], medians: pd.Series) -> pd.DataFrame:
    """Debut-driven NaNs -> 0 ('no evidence'); real-but-missing attributes -> median."""
    X = df[features].copy()
    X[MEDIAN_IMPUTE] = X[MEDIAN_IMPUTE].fillna(medians)
    return X.fillna(0)


def _new_calibrated_lr() -> CalibratedClassifierCV:
    return CalibratedClassifierCV(
        Pipeline([("scale", StandardScaler()),
                  ("clf", LogisticRegression(max_iter=1000, random_state=RNG_SEED))]),
        method="isotonic", cv=5)


def train_win_model(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """Benchmark on the rolling holdout, then refit on ALL rows and save. Returns metrics."""
    for df in (train, test):
        df["debut_a"] = df["is_debut_a"].astype(int)
        df["debut_b"] = df["is_debut_b"].astype(int)
    features = _win_features(train)

    # (a) health benchmark: fit on train only, score the untouched holdout
    med_train = train[MEDIAN_IMPUTE].median()
    bench = _new_calibrated_lr()
    bench.fit(_prep_win(train, features, med_train), train["target"].to_numpy())
    p = bench.predict_proba(_prep_win(test, features, med_train))[:, 1]
    metrics = scores(test["target"].to_numpy(), p)

    # (b) shipped model: refit on everything so it has learned from the newest fights
    full = pd.concat([train, test], ignore_index=True)
    med_full = full[MEDIAN_IMPUTE].median()
    final = _new_calibrated_lr()
    final.fit(_prep_win(full, features, med_full), full["target"].to_numpy())

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump({
        "model": final,
        "features": features,
        "median_impute": MEDIAN_IMPUTE,
        "medians": med_full,
        "calibrated": True,
        "trained_through": str(full["date"].max().date()),
    }, WIN_MODEL_PATH)
    return metrics


def _finish_design(model_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Feature matrix + 3-class label for the finish model (nb03 Part 2 §2–3)."""
    y = model_df["method"].map(METHOD_MAP)
    assert y.notna().all(), "every kept fight must map to a finish class"
    num_features = [f"{c}_{s}" for c in PROFILE for s in ("a", "b")] + ["title_fight"]
    X = pd.concat([model_df[num_features],
                   pd.get_dummies(model_df["division"], prefix="div")], axis=1)
    return X, y, num_features


def _impute_finish(X: pd.DataFrame, num_features: list[str], age_medians: pd.Series) -> pd.DataFrame:
    X = X.copy()
    age_cols = ["age_at_fight_a", "age_at_fight_b"]
    X[age_cols] = X[age_cols].fillna(age_medians)
    X[num_features] = X[num_features].fillna(0)
    return X


def train_finish_model(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """LR vs XGBoost on the holdout; ship the winner refit on ALL rows. Returns metrics."""
    full = pd.concat([train, test], ignore_index=True)
    X_all, y_all, num_features = _finish_design(full)
    is_test = np.zeros(len(full), dtype=bool)
    is_test[len(train):] = True
    age_cols = ["age_at_fight_a", "age_at_fight_b"]

    med_train = X_all.loc[~is_test, age_cols].median()
    X_tr = _impute_finish(X_all[~is_test], num_features, med_train)
    X_te = _impute_finish(X_all[is_test], num_features, med_train)
    y_tr, y_te = y_all[~is_test], y_all[is_test]

    lr = Pipeline([("scale", StandardScaler()),
                   ("clf", LogisticRegression(max_iter=2000, random_state=RNG_SEED))])
    lr.fit(X_tr, y_tr)
    lr_ll = log_loss(y_te, lr.predict_proba(X_te), labels=list(lr.classes_))

    cls_idx = {c: i for i, c in enumerate(FINISH_CLASSES)}
    xgb = XGBClassifier(n_estimators=500, learning_rate=0.03, max_depth=4,
                        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                        reg_lambda=1.0, objective="multi:softprob", num_class=3,
                        eval_metric="mlogloss", random_state=RNG_SEED)
    xgb.fit(X_tr, y_tr.map(cls_idx))
    xgb_ll = log_loss(y_te.map(cls_idx), xgb.predict_proba(X_te), labels=[0, 1, 2])

    best_is_xgb = xgb_ll < lr_ll

    # refit the winner on everything, with all-data imputation medians
    med_full = X_all[age_cols].median()
    X_full = _impute_finish(X_all, num_features, med_full)
    if best_is_xgb:
        final = XGBClassifier(**xgb.get_params())
        final.fit(X_full, y_all.map(cls_idx))
    else:
        final = Pipeline([("scale", StandardScaler()),
                          ("clf", LogisticRegression(max_iter=2000, random_state=RNG_SEED))])
        final.fit(X_full, y_all)

    joblib.dump({
        "model": final,
        "features": list(X_full.columns),
        "num_features": num_features,
        "classes": FINISH_CLASSES,
        "is_xgb": best_is_xgb,
        "age_medians": med_full.to_dict(),
    }, FINISH_MODEL_PATH)
    return {"finish_model": "XGBoost" if best_is_xgb else "LogReg",
            "finish_log_loss": round(min(xgb_ll, lr_ll), 4)}
