"""Point-in-time features & temporal split — the src/ port of notebook 02 §6–8.

Every feature for a fight is built from fights STRICTLY BEFORE that fight's
date (shift/cumsum patterns + a chronological Elo walk), then made symmetric
via a seeded random corner swap and a−b difference columns. The same leakage
asserts as the notebook run on every rebuild — a bad refactor fails loudly.

The split is a ROLLING temporal cutoff: the most recent `HOLDOUT_MONTHS` of
data is the holdout, so the boundary moves forward as new fights arrive.
"""
from collections import defaultdict

import numpy as np
import pandas as pd

from .config import (DIVISIONS, ELO_BASE, ELO_K, FEATURES_CSV, FINISH_METHODS,
                     HOLDOUT_MONTHS, PROCESSED_DIR, RNG_SEED, SNAPSHOTS_CSV,
                     TEST_CSV, TRAIN_CSV)

FEAT_COLS = [
    "age_at_fight", "height_in", "reach_in", "reach_imputed",
    "career_fights", "career_wins", "career_losses", "career_win_rate",
    "career_win_streak", "career_finish_rate", "is_debut",
    "career_avg_sig_str_landed", "career_avg_sig_str_att",
    "career_avg_takedowns_landed", "career_avg_takedowns_att",
    "career_avg_sub_att", "career_avg_control_time_sec",
    "days_since_last", "recent_win_rate_3", "recent_finish_rate_3", "elo",
]
# per-corner flags (reach_imputed / is_debut) and elo's shared 1500 start stay un-differenced
DIFF_BASES = [c for c in FEAT_COLS if c not in ("reach_imputed", "is_debut")]


def _prior_win_streak(won):
    """Streak entering the fight: append BEFORE updating with the current result."""
    streak, out = 0, []
    for w in won:
        out.append(streak)
        streak = streak + 1 if w == 1 else 0
    return pd.Series(out, index=won.index)


def _extract_division(wc):
    if pd.isna(wc):
        return "Unknown"
    # substring, not startswith: title bouts read "UFC Women's Bantamweight Title Bout"
    women = "Women's" in wc
    for d in sorted(DIVISIONS, key=len, reverse=True):  # longest first: 'Light Heavyweight'
        if d in wc:
            return ("Women's " if women else "") + d
    return "Other"


def _elo_walk(fights: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Walk all fights chronologically. Returns (pre-fight Elo per fighter per fight,
    final post-fight ratings) — the first feeds training, the second feeds snapshots."""
    elo = defaultdict(lambda: ELO_BASE)
    rows = []
    cols = ["fight_id", "fighter_a_id", "fighter_b_id", "winner_id", "date"]
    for f in fights[cols].sort_values(["date", "fight_id"]).itertuples(index=False):
        ra, rb = elo[f.fighter_a_id], elo[f.fighter_b_id]
        rows.append((f.fight_id, f.fighter_a_id, ra))
        rows.append((f.fight_id, f.fighter_b_id, rb))
        sa = 1.0 if f.winner_id == f.fighter_a_id else 0.0
        ea = 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))
        elo[f.fighter_a_id] = ra + ELO_K * (sa - ea)
        elo[f.fighter_b_id] = rb + ELO_K * ((1.0 - sa) - (1.0 - ea))
    return pd.DataFrame(rows, columns=["fight_id", "fighter_id", "elo"]), dict(elo)


def build_fighter_snapshots(fights: pd.DataFrame, stats: pd.DataFrame,
                            fighters: pd.DataFrame) -> pd.DataFrame:
    """Each fighter's CURRENT state — career INCLUDING their latest fight, final Elo,
    last-3 form, last division. This is what a hypothetical matchup 'as of today'
    needs; unlike the training features there is nothing to hold out, so no shift.
    Date-dependent fields (age, days_since_last) are left for the caller to compute
    from `dob` / `last_fight_date` at prediction time.
    """
    base_cols = ["fight_id", "date", "winner_id", "method", "weight_class"]
    long = pd.concat([
        fights[base_cols + ["fighter_a_id"]].rename(columns={"fighter_a_id": "fighter_id"}),
        fights[base_cols + ["fighter_b_id"]].rename(columns={"fighter_b_id": "fighter_id"}),
    ], ignore_index=True)
    long["won"] = (long["fighter_id"] == long["winner_id"]).astype(int)
    long["won_by_finish"] = ((long["won"] == 1) & long["method"].isin(FINISH_METHODS)).astype(int)
    long = long.merge(
        stats[["fight_id", "fighter_id", "sig_str_landed", "sig_str_att",
               "takedowns_landed", "takedowns_att", "sub_att", "control_time_sec"]],
        on=["fight_id", "fighter_id"], how="left")
    long = long.sort_values(["fighter_id", "date", "fight_id"])
    g = long.groupby("fighter_id", sort=False)

    snap = pd.DataFrame({"career_fights": g.size(), "career_wins": g["won"].sum()})
    snap["career_losses"] = snap["career_fights"] - snap["career_wins"]
    snap["career_win_rate"] = snap["career_wins"] / snap["career_fights"]
    snap["career_finish_rate"] = (g["won_by_finish"].sum()
                                  / snap["career_wins"].replace(0, np.nan)).fillna(0)
    for col in ["sig_str_landed", "sig_str_att", "takedowns_landed", "takedowns_att",
                "sub_att", "control_time_sec"]:
        snap[f"career_avg_{col}"] = g[col].mean()
    # trailing win streak: reversed cumprod of 0/1 results stays 1 until the last loss
    snap["career_win_streak"] = g["won"].agg(lambda s: int(s[::-1].cumprod().sum()))
    snap["recent_win_rate_3"] = g["won"].agg(lambda s: s.tail(3).mean())
    snap["recent_finish_rate_3"] = g["won_by_finish"].agg(lambda s: s.tail(3).mean())
    snap["last_fight_date"] = g["date"].max()
    snap["last_division"] = g["weight_class"].agg("last").map(_extract_division)
    snap["is_debut"] = 0  # everyone here has at least one completed fight

    _, elo_final = _elo_walk(fights)
    snap["elo"] = snap.index.map(elo_final)

    # pro_* are the full-career W-L-D for display only — never differenced into features
    snap = snap.reset_index().merge(
        fighters[["fighter_id", "name", "dob", "height_in", "reach_in", "reach_imputed",
                  "pro_wins", "pro_losses", "pro_draws"]],
        on="fighter_id", how="left")
    assert snap["elo"].notna().all()
    return snap


def build_per_fighter(fights: pd.DataFrame, stats: pd.DataFrame,
                      fighters: pd.DataFrame) -> pd.DataFrame:
    """One row per fighter per fight with all strictly-pre-fight features (nb02 §6/6b)."""
    base_cols = ["fight_id", "date", "winner_id", "method"]
    long = pd.concat([
        fights[base_cols + ["fighter_a_id"]].rename(columns={"fighter_a_id": "fighter_id"}),
        fights[base_cols + ["fighter_b_id"]].rename(columns={"fighter_b_id": "fighter_id"}),
    ], ignore_index=True)
    long["won"] = (long["fighter_id"] == long["winner_id"]).astype(int)
    long["won_by_finish"] = ((long["won"] == 1) & long["method"].isin(FINISH_METHODS)).astype(int)

    long = long.merge(
        stats[["fight_id", "fighter_id", "sig_str_landed", "sig_str_att",
               "takedowns_landed", "takedowns_att", "sub_att", "control_time_sec"]],
        on=["fight_id", "fighter_id"], how="left")
    long = long.merge(fighters[["fighter_id", "dob", "height_in", "reach_in", "reach_imputed"]],
                      on="fighter_id", how="left")

    long = long.sort_values(["fighter_id", "date", "fight_id"]).reset_index(drop=True)
    g = long.groupby("fighter_id", sort=False)

    # counts: cumulative-minus-current == strictly-prior totals
    long["career_fights"] = g.cumcount()
    long["career_wins"] = g["won"].cumsum() - long["won"]
    long["career_losses"] = long["career_fights"] - long["career_wins"]
    long["career_win_rate"] = long["career_wins"] / long["career_fights"].replace(0, np.nan)
    prior_finishes = g["won_by_finish"].cumsum() - long["won_by_finish"]
    long["career_finish_rate"] = (prior_finishes / long["career_wins"].replace(0, np.nan)).fillna(0)

    # averages: shift(1) pushes the current fight out of its own window
    for col in ["sig_str_landed", "sig_str_att", "takedowns_landed", "takedowns_att",
                "sub_att", "control_time_sec"]:
        long[f"career_avg_{col}"] = g[col].transform(lambda s: s.shift(1).expanding().mean())

    long["career_win_streak"] = g["won"].transform(_prior_win_streak)
    long["days_since_last"] = (long["date"] - g["date"].shift(1)).dt.days
    long["age_at_fight"] = (long["date"] - long["dob"]).dt.days / 365.25

    # 6b: debut flag + last-3 rolling form
    long["is_debut"] = (long["career_fights"] == 0).astype(int)
    long["recent_win_rate_3"] = g["won"].transform(
        lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    long["recent_finish_rate_3"] = g["won_by_finish"].transform(
        lambda s: s.shift(1).rolling(3, min_periods=1).mean())

    # 6b: pre-fight Elo, walked in chronological order
    elo_pre, _ = _elo_walk(fights)
    long = long.merge(elo_pre, on=["fight_id", "fighter_id"], how="left", validate="1:1")

    # leakage asserts — every rebuild must pass these, not just the notebook run
    debuts = long.groupby("fighter_id").head(1)  # head(1)=literal first row (first() skips NaN)
    assert (debuts["career_fights"] == 0).all()
    assert (debuts["is_debut"] == 1).all()
    assert debuts["days_since_last"].isna().all()
    assert debuts["recent_win_rate_3"].isna().all()
    assert (debuts["elo"] == ELO_BASE).all() and long["elo"].notna().all()
    chk = long.sort_values(["fighter_id", "date"])
    lag = chk.groupby(chk["fighter_id"])["won"].cumsum().groupby(chk["fighter_id"]).shift(1).fillna(0)
    assert (chk["career_wins"] == lag).all(), "career_wins must lag results by one fight"

    return long


def build_model_df(fights: pd.DataFrame, stats: pd.DataFrame,
                   fighters: pd.DataFrame) -> pd.DataFrame:
    """Symmetric per-fight feature table: corner swap + a−b diffs (nb02 §7)."""
    long = build_per_fighter(fights, stats, fighters)
    per_fighter = long[["fight_id", "fighter_id"] + FEAT_COLS]

    model_df = fights[["fight_id", "date", "weight_class", "fighter_a_id", "fighter_b_id",
                       "winner_id", "method"]].copy()
    for side in ["a", "b"]:
        side_df = per_fighter.rename(
            columns={"fighter_id": f"fighter_{side}_id", **{c: f"{c}_{side}" for c in FEAT_COLS}})
        model_df = model_df.merge(side_df, on=["fight_id", f"fighter_{side}_id"],
                                  how="left", validate="1:1")

    # seeded random swap of the a/b roles — kills the "winner listed first" leak
    rng = np.random.default_rng(RNG_SEED)
    swap = rng.random(len(model_df)) < 0.5
    swap_pairs = [("fighter_a_id", "fighter_b_id")] + [(f"{c}_a", f"{c}_b") for c in FEAT_COLS]
    for col_a, col_b in swap_pairs:
        model_df.loc[swap, [col_a, col_b]] = model_df.loc[swap, [col_b, col_a]].to_numpy()

    model_df["target"] = (model_df["winner_id"] == model_df["fighter_a_id"]).astype(int)
    for base in DIFF_BASES:
        model_df[f"{base}_diff"] = model_df[f"{base}_a"] - model_df[f"{base}_b"]

    # division + title flag (finish-method model features, from nb03 Part 2)
    model_df["division"] = model_df["weight_class"].apply(_extract_division)
    model_df["title_fight"] = model_df["weight_class"].str.contains("Title", na=False).astype(int)

    target_rate = model_df["target"].mean()
    assert 0.45 < target_rate < 0.55, f"corner swap broken: target mean {target_rate:.3f}"
    return model_df.sort_values("date").reset_index(drop=True)


def rolling_split(model_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """Rolling temporal split: holdout = most recent HOLDOUT_MONTHS of the data."""
    cutoff = model_df["date"].max() - pd.DateOffset(months=HOLDOUT_MONTHS)
    train = model_df[model_df["date"] < cutoff]
    test = model_df[model_df["date"] >= cutoff]
    assert train["date"].max() < test["date"].min(), "temporal split must not overlap"
    return train, test, cutoff


def rebuild_features(fights, stats, fighters) -> dict:
    """Full rebuild: features + rolling split, written to data/processed/. Returns summary."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    model_df = build_model_df(fights, stats, fighters)
    train, test, cutoff = rolling_split(model_df)

    # fixed float format so the CSVs serialize identically on any Python/numpy
    # build (pandas' default shortest-repr drifts across versions) — a rebuild
    # with no new fights then produces a byte-identical file, i.e. no spurious diff
    fmt = "%.10g"
    model_df.to_csv(FEATURES_CSV, index=False, float_format=fmt)
    train.to_csv(TRAIN_CSV, index=False, float_format=fmt)
    test.to_csv(TEST_CSV, index=False, float_format=fmt)

    # current per-fighter snapshots for the serving app's hypothetical matchups
    snapshots = build_fighter_snapshots(fights, stats, fighters)
    snapshots.to_csv(SNAPSHOTS_CSV, index=False, float_format=fmt)

    return {"n_fights": len(model_df), "n_train": len(train), "n_test": len(test),
            "n_fighters": len(snapshots), "cutoff": cutoff.date(),
            "latest_fight": model_df["date"].max().date()}
