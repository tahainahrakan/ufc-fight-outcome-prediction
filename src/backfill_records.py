"""One-time backfill: add each fighter's full pro record to raw_fighters.csv.

The record (page-header 'W-L-D') is new information the original scrape never
kept. This re-fetches every fighter page whose `record` is still blank and fills
it in, flushing to disk every 25 fighters so a crash loses almost nothing — just
re-run to resume. Records already present are skipped unless --force is passed.

    python -m src.backfill_records            # fill blanks (resumable)
    python -m src.backfill_records --force     # re-fetch every fighter (refresh)
    python -m src.backfill_records --limit 20  # smoke test on the first 20
"""
import argparse
import sys

import pandas as pd

from .config import FIGHTERS_CSV
from .scraping import parse_fighter


def backfill(force: bool = False, limit: int | None = None) -> dict:
    df = pd.read_csv(FIGHTERS_CSV, dtype={"fighter_id": str})
    if "record" not in df.columns:
        df["record"] = pd.NA

    todo = df if force else df[df["record"].isna()]
    ids = todo["fighter_id"].tolist()
    if limit is not None:
        ids = ids[:limit]
    print(f"backfilling records for {len(ids)} fighters (force={force})")

    idx = df.set_index("fighter_id")
    filled = failed = 0
    for i, fid in enumerate(ids, 1):
        try:
            # cache disabled so a fighter who has fought since the last scrape
            # gets their up-to-date record, not a stale cached page
            idx.at[fid, "record"] = parse_fighter(fid, use_cache=False).get("record")
            filled += 1
        except Exception as e:  # noqa: BLE001 — one bad page must not kill the run
            failed += 1
            print(f"  FAILED {fid}: {e}")
        if i % 25 == 0 or i == len(ids):
            idx.reset_index().to_csv(FIGHTERS_CSV, index=False)
            print(f"  {i}/{len(ids)} processed ({filled} filled, {failed} failed)")

    return {"processed": len(ids), "filled": filled, "failed": failed}


def main() -> int:
    ap = argparse.ArgumentParser(description="Backfill full pro records into raw_fighters.csv")
    ap.add_argument("--force", action="store_true", help="re-fetch even fighters that already have a record")
    ap.add_argument("--limit", type=int, default=None, help="only process the first N fighters (smoke test)")
    args = ap.parse_args()
    summary = backfill(force=args.force, limit=args.limit)
    print(f"done: {summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
