"""Raw-CSV loading & cleaning — the src/ port of notebook 02 §1–5.

Turns the four raw ufcstats scrapes into three clean frames:
`fights` (dated, filtered to clean win/loss), `stats` (per-fighter per-fight,
control time in seconds), `fighters` (heights/reaches in inches, reach imputed).

The notebook's wide per-fight stat pivot is intentionally NOT ported: it was
diagnostic-only — nothing downstream of it feeds the model (a fight's own stats
would be leakage; they only fuel *career history* for later fights).
"""
import re

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from .config import EVENTS_CSV, FIGHTERS_CSV, FIGHTS_CSV, STATS_CSV, VALID_METHODS

NA = ["--"]  # ufcstats' universal "no data" placeholder


def _parse_record(rec):
    """'27-1-0' or '27-1-0 (1 NC)' -> (pro_wins, pro_losses, pro_draws) as floats.

    This is the fighter's full professional W-L-D from the ufcstats page header
    (display-only). Returns (nan, nan, nan) when the record is missing/unparseable.
    """
    if pd.isna(rec):
        return (np.nan, np.nan, np.nan)
    m = re.match(r"\s*(\d+)\s*-\s*(\d+)\s*-\s*(\d+)", str(rec))
    return (float(m.group(1)), float(m.group(2)), float(m.group(3))) if m else (np.nan, np.nan, np.nan)


def _height_to_inches(h):
    if pd.isna(h):
        return np.nan
    feet, inches = h.replace('"', "").split("'")
    return int(feet) * 12 + int(inches)


def _reach_to_inches(r):
    return np.nan if pd.isna(r) else float(str(r).replace('"', ""))


def _mmss_to_seconds(t):
    if pd.isna(t):
        return np.nan
    m, s = str(t).split(":")
    return int(m) * 60 + int(s)


def load_clean() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load, type-cast, date, filter, and impute. Returns (fights, stats, fighters)."""
    events = pd.read_csv(EVENTS_CSV, na_values=NA)
    fights = pd.read_csv(FIGHTS_CSV, na_values=NA)
    stats = pd.read_csv(STATS_CSV, na_values=NA)
    fighters = pd.read_csv(FIGHTERS_CSV, na_values=NA)

    # type-casting (nb02 §1)
    fighters["height_in"] = fighters["height"].map(_height_to_inches)
    fighters["reach_in"] = fighters["reach"].map(_reach_to_inches)
    fighters["dob"] = pd.to_datetime(fighters["dob"], format="%b %d, %Y")
    rec_col = fighters["record"] if "record" in fighters.columns else pd.Series(np.nan, index=fighters.index)
    fighters[["pro_wins", "pro_losses", "pro_draws"]] = [_parse_record(r) for r in rec_col]
    events["date"] = pd.to_datetime(events["date"], format="%B %d, %Y")
    stats["control_time_sec"] = stats["control_time"].map(_mmss_to_seconds)

    # merge event date onto fights (nb02 §2)
    fights = fights.merge(events[["event_id", "date"]], on="event_id",
                          how="left", validate="m:1")
    assert fights["date"].notna().all(), "every fight must have a date"

    # filter to clean win/loss outcomes (nb02 §3)
    keep = fights["winner_id"].notna() & fights["method"].isin(VALID_METHODS)
    fights = fights[keep].copy()
    assert fights["fight_id"].is_unique

    # impute missing reach from height (nb02 §4); flag records which are estimates
    have_both = fighters["height_in"].notna() & fighters["reach_in"].notna()
    reg = LinearRegression().fit(fighters.loc[have_both, ["height_in"]],
                                 fighters.loc[have_both, "reach_in"])
    fighters["reach_imputed"] = fighters["reach_in"].isna()
    fill = fighters["reach_in"].isna() & fighters["height_in"].notna()
    fighters.loc[fill, "reach_in"] = reg.predict(fighters.loc[fill, ["height_in"]])
    fighters["reach_in"] = fighters["reach_in"].fillna(fighters["reach_in"].median())
    assert fighters["reach_in"].notna().all()

    return fights, stats, fighters
