"""Hypothetical-matchup assembly for the serving app.

The training features describe a fighter *entering a past fight*; the app needs
each fighter *as of today*. `matchup_row` takes two rows of the weekly-refreshed
fighter_snapshots.csv, computes the date-dependent fields (age, layoff) at
prediction time, builds the symmetric diffs, and returns a Series shaped exactly
like a processed-feature row — so `src.utils.predict_fight` serves it unchanged.
"""
from datetime import datetime

import pandas as pd

from src.config import FEATURES_CSV, SNAPSHOTS_CSV
from src.feature_engineering import DIFF_BASES, FEAT_COLS


def load_snapshots() -> pd.DataFrame:
    """Fighter snapshots indexed by fighter_id (refreshed by the weekly update)."""
    snaps = pd.read_csv(SNAPSHOTS_CSV, parse_dates=["last_fight_date", "dob"])
    return snaps.set_index("fighter_id")


def load_fights() -> pd.DataFrame:
    """Completed fights, trimmed to the columns head-to-head needs (refreshed weekly)."""
    cols = ["date", "fighter_a_id", "fighter_b_id", "winner_id", "method"]
    return pd.read_csv(FEATURES_CSV, usecols=cols, parse_dates=["date"])


def head_to_head(fights: pd.DataFrame, id_a: str, id_b: str) -> dict:
    """Prior meetings between two fighters, newest first, from fighter A's point of view.

    Each meeting's winner is reported as corner "A" or "B" (the two ids passed in), so the
    caller can colour it with the current matchup's palette. `summary` reads "A leads 2–1",
    "Series tied 1–1", or "" when they have never met.
    """
    pair = (((fights["fighter_a_id"] == id_a) & (fights["fighter_b_id"] == id_b))
            | ((fights["fighter_a_id"] == id_b) & (fights["fighter_b_id"] == id_a)))
    hits = fights[pair].sort_values("date", ascending=False)

    meetings, wins_a, wins_b = [], 0, 0
    for _, r in hits.iterrows():
        winner = "A" if r["winner_id"] == id_a else "B"
        wins_a += winner == "A"
        wins_b += winner == "B"
        meetings.append({"date": str(r["date"].date()),
                         "winner": winner, "method": r["method"]})

    if not meetings:
        summary = ""
    elif wins_a == wins_b:
        summary = f"Series tied {wins_a}–{wins_b}"
    else:
        lead = "A" if wins_a > wins_b else "B"
        summary = f"{lead} leads {max(wins_a, wins_b)}–{min(wins_a, wins_b)}"
    return {"meetings": meetings, "summary": summary}


def fmt_height(inches) -> str:
    """72 -> \"6'0\\\"\".  Returns '—' when height is unknown."""
    if pd.isna(inches):
        return "—"
    feet, rem = divmod(int(round(inches)), 12)
    return f"{feet}'{rem}\""


def pro_record(s: pd.Series) -> str:
    """Full pro record 'W-L' (or 'W-L-D' with draws), from the ufcstats page header.

    Falls back to the UFC-only career record when the pro record wasn't scraped.
    """
    if pd.notna(s.get("pro_wins")) and pd.notna(s.get("pro_losses")):
        w, l, d = int(s["pro_wins"]), int(s["pro_losses"]), int(s.get("pro_draws") or 0)
        return f"{w}-{l}-{d}" if d else f"{w}-{l}"
    return f"{int(s['career_wins'])}-{int(s['career_losses'])}"


def fighter_label(s: pd.Series) -> str:
    """Display label: 'Name (W-L, Division)' using the full pro record."""
    return f"{s['name']} ({pro_record(s)}, {s['last_division']})"


def matchup_row(snap_a: pd.Series, snap_b: pd.Series, division: str,
                title_fight: bool, as_of=None) -> pd.Series:
    """Build one predict_fight-ready row for a hypothetical A-vs-B bout."""
    as_of = pd.Timestamp(as_of if as_of is not None else datetime.now().date())

    row: dict = {}
    for side, s in (("a", snap_a), ("b", snap_b)):
        vals = {c: s[c] for c in FEAT_COLS if c in s.index}
        # date-dependent fields are computed now, not at snapshot-build time
        vals["age_at_fight"] = ((as_of - s["dob"]).days / 365.25
                                if pd.notna(s["dob"]) else float("nan"))
        vals["days_since_last"] = float((as_of - s["last_fight_date"]).days)
        for c, v in vals.items():
            row[f"{c}_{side}"] = v

    for base in DIFF_BASES:
        row[f"{base}_diff"] = row[f"{base}_a"] - row[f"{base}_b"]

    row["division"] = division
    row["title_fight"] = int(bool(title_fight))
    return pd.Series(row)
