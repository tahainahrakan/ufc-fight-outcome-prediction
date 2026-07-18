"""Central paths & constants for the UFC fight-outcome pipeline.

Everything is resolved relative to the project root so the modules work the same
from a notebook, the CLI (`python -m src.update`), or the weekly launchd job.

Tunables (scrape delays, Elo params, holdout window, RNG seed, app host/port) are
read from `config.yaml` at the project root; anything missing there falls back to
the defaults passed to `_cfg()` below. Paths and the method/division vocabularies
are structural and stay in this file.
"""
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── Tunables from config.yaml (with safe fallbacks) ──────────────
_CONFIG_YAML = PROJECT_ROOT / "config.yaml"
try:
    _YAML = yaml.safe_load(_CONFIG_YAML.read_text()) or {}
except FileNotFoundError:
    _YAML = {}


def _cfg(section: str, key: str, default):
    """Read config.yaml[section][key], falling back to `default` if absent."""
    return (_YAML.get(section) or {}).get(key, default)

# ── Data locations ───────────────────────────────────────────────
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CACHE_DIR = DATA_DIR / "_html_cache"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"

EVENTS_CSV = RAW_DIR / "raw_events.csv"
FIGHTS_CSV = RAW_DIR / "raw_fights.csv"
STATS_CSV = RAW_DIR / "raw_fight_stats.csv"
FIGHTERS_CSV = RAW_DIR / "raw_fighters.csv"

FEATURES_CSV = PROCESSED_DIR / "fights_features.csv"
TRAIN_CSV = PROCESSED_DIR / "train.csv"
TEST_CSV = PROCESSED_DIR / "test.csv"
SNAPSHOTS_CSV = PROCESSED_DIR / "fighter_snapshots.csv"

WIN_MODEL_PATH = MODELS_DIR / "win_probability_lr.joblib"
FINISH_MODEL_PATH = MODELS_DIR / "finish_method.joblib"
METRICS_HISTORY_CSV = MODELS_DIR / "metrics_history.csv"

# ── Scraping ─────────────────────────────────────────────────────
BASE = _cfg("scraping", "base_url", "http://ufcstats.com")
MIN_DELAY = _cfg("scraping", "min_delay", 1.0)  # polite pause between live requests (seconds)
MAX_DELAY = _cfg("scraping", "max_delay", 2.0)

# ── Modeling ─────────────────────────────────────────────────────
RNG_SEED = _cfg("model", "rng_seed", 42)
ELO_BASE = _cfg("model", "elo_base", 1500.0)
ELO_K = _cfg("model", "elo_k", 32.0)
HOLDOUT_MONTHS = _cfg("model", "holdout_months", 12)  # rolling holdout: most recent N months

# ── App serving ──────────────────────────────────────────────────
APP_HOST = _cfg("app", "host", "127.0.0.1")
APP_PORT = _cfg("app", "port", 5001)

VALID_METHODS = [
    "Decision - Unanimous", "Decision - Split", "Decision - Majority",
    "KO/TKO", "Submission", "TKO - Doctor's Stoppage",
]
FINISH_METHODS = ["KO/TKO", "Submission", "TKO - Doctor's Stoppage"]
METHOD_MAP = {
    "Decision - Unanimous": "Decision", "Decision - Split": "Decision",
    "Decision - Majority": "Decision",
    "KO/TKO": "KO/TKO", "TKO - Doctor's Stoppage": "KO/TKO",
    "Submission": "Submission",
}
FINISH_CLASSES = ["Decision", "KO/TKO", "Submission"]

DIVISIONS = ["Strawweight", "Flyweight", "Bantamweight", "Featherweight", "Lightweight",
             "Welterweight", "Middleweight", "Light Heavyweight", "Heavyweight",
             "Catch Weight", "Open Weight"]
