"""Weekly self-update: scrape new fights -> rebuild features -> retrain -> log.

Run manually:            python -m src.update
Skip the network step:   python -m src.update --skip-scrape
Scheduled: the launchd agent com.rakantahineh.ufc-weekly-update runs this every
Monday morning (see logs/weekly_update.log for each run's output).

Elo and career features are rebuilt from the full history every run, so a new
result automatically shifts both fighters' ratings for all their later fights.
"""
import argparse
import sys
import time
from datetime import datetime

import pandas as pd

from .config import TEST_CSV, TRAIN_CSV
from .data_preprocessing import load_clean
from .evaluation import append_metrics
from .feature_engineering import rebuild_features
from .model_training import train_finish_model, train_win_model
from .scraping import update_raw_data


def main() -> int:
    ap = argparse.ArgumentParser(description="Weekly UFC data update + retrain")
    ap.add_argument("--skip-scrape", action="store_true",
                    help="retrain from existing raw CSVs without hitting ufcstats.com")
    ap.add_argument("--max-events", type=int, default=None,
                    help="cap how many newest events are considered (debugging)")
    args = ap.parse_args()

    t0 = time.time()
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    print(f"=== UFC weekly update — {stamp} ===")

    # 1) incremental scrape
    if args.skip_scrape:
        scrape = {"new_events": 0, "new_fights": 0, "upcoming_skipped": 0,
                  "new_fighters": 0, "failed": []}
        print("[1/3] scrape skipped (--skip-scrape)")
    else:
        print("[1/3] scraping new events from ufcstats.com …")
        scrape = update_raw_data(max_events=args.max_events)
        print(f"      +{scrape['new_events']} events, +{scrape['new_fights']} fights, "
              f"+{scrape['new_fighters']} fighter bios, "
              f"{scrape.get('records_refreshed', 0)} records refreshed "
              f"({scrape['upcoming_skipped']} upcoming bouts left for next week)")
        for url, why in scrape["failed"][:5]:
            print(f"      FAILED {why} -> {url}")

    # 2) rebuild leak-free features with the rolling temporal split
    print("[2/3] rebuilding features …")
    fights, stats, fighters = load_clean()
    feat = rebuild_features(fights, stats, fighters)
    print(f"      {feat['n_fights']} fights (latest {feat['latest_fight']}) — "
          f"train {feat['n_train']} / holdout {feat['n_test']} (cutoff {feat['cutoff']})")

    # 3) retrain: benchmark on the holdout, then refit shipped models on ALL data
    print("[3/3] retraining models …")
    train = pd.read_csv(TRAIN_CSV, parse_dates=["date"])
    test = pd.read_csv(TEST_CSV, parse_dates=["date"])
    win_metrics = train_win_model(train, test)
    finish_metrics = train_finish_model(train, test)
    print(f"      win model (holdout): log-loss {win_metrics['log_loss']}  "
          f"brier {win_metrics['brier']}  acc {win_metrics['accuracy']}  ece {win_metrics['ece']}")
    print(f"      finish model: {finish_metrics['finish_model']} "
          f"(holdout log-loss {finish_metrics['finish_log_loss']})")

    append_metrics({
        "run_date": datetime.now().strftime("%Y-%m-%d"),
        "n_fights": feat["n_fights"],
        "new_events": scrape["new_events"],
        "new_fights": scrape["new_fights"],
        "cutoff": str(feat["cutoff"]),
        **win_metrics,
        **finish_metrics,
    })
    print(f"done in {time.time() - t0:.0f}s — models refit through {feat['latest_fight']}, "
          "metrics appended to models/metrics_history.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
