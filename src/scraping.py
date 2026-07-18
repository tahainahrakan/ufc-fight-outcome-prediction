"""Incremental ufcstats.com scraper — the src/ port of notebook 01.

The anti-bot wall is a SHA-256 proof-of-work, not a browser check: solve
`sha256(f"{nonce}:{n}")` until the hex digest has the demanded leading zeros,
POST it, and the session cookie unlocks every page (no Playwright needed).

`update_raw_data()` is the weekly entrypoint: it re-reads the completed-events
index and scrapes only events/fights/fighters not already in the raw CSVs.
New event & fight pages are fetched with the cache DISABLED — a fight page
cached while the bout was still upcoming would otherwise never show its result.
"""
import hashlib
import random
import re
import time

import pandas as pd
import requests
from bs4 import BeautifulSoup

from .config import (BASE, CACHE_DIR, EVENTS_CSV, FIGHTERS_CSV, FIGHTS_CSV,
                     MAX_DELAY, MIN_DELAY, RAW_DIR, STATS_CSV)

session = requests.Session()
session.headers.update({"User-Agent": "Mozilla/5.0 (educational research scraper)"})


def _solve_pow(challenge_html: str) -> None:
    """Solve the SHA-256 proof-of-work and POST it so the session gets the access cookie."""
    nonce = re.search(r'nonce="([0-9a-f]+)"', challenge_html).group(1)
    zeros = int(re.search(r'target=new Array\((\d+)\+1\)', challenge_html).group(1))
    target = "0" * zeros
    n = 0
    while not hashlib.sha256(f"{nonce}:{n}".encode()).hexdigest().startswith(target):
        n += 1
    session.post(f"{BASE}/__c", data={"nonce": nonce, "n": n}, timeout=30)


def _get_html(url: str, tries: int = 4) -> str:
    """Fetch rendered HTML, solving the PoW challenge if shown and retrying transient
    network errors with exponential backoff."""
    solved = False
    for attempt in range(tries):
        time.sleep(random.uniform(MIN_DELAY, MAX_DELAY))  # be polite
        try:
            resp = session.get(url, timeout=30)
        except requests.RequestException:
            if attempt == tries - 1:
                raise
            time.sleep(2 ** attempt)
            continue
        if "Checking your browser" in resp.text and not solved:
            _solve_pow(resp.text)  # get the access cookie…
            solved = True
            continue               # …and refetch, now cookied
        return resp.text
    raise RuntimeError(f"failed to fetch after {tries} tries: {url}")


def get_soup(url: str, use_cache: bool = True) -> BeautifulSoup:
    """Return parsed HTML for a URL, caching the raw HTML on disk for fast, polite reruns."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / (re.sub(r"[^a-z0-9]+", "_", url.lower()).strip("_")[-120:] + ".html")
    if use_cache and cache_file.exists():
        return BeautifulSoup(cache_file.read_text(encoding="utf-8"), "lxml")
    html = _get_html(url)
    if use_cache:
        cache_file.write_text(html, encoding="utf-8")
    return BeautifulSoup(html, "lxml")


# ── Parsing helpers ──────────────────────────────────────────────

def url_id(url: str) -> str:
    """Last path segment of a ufcstats URL — the stable hex id we use as a primary key."""
    return url.rstrip("/").split("/")[-1]


def split_landed_attempted(text: str):
    """'15 of 25' -> (15, 25). Returns (None, None) for '---' or anything unparseable."""
    m = re.match(r"(\d+)\s+of\s+(\d+)", text.strip())
    return (int(m.group(1)), int(m.group(2))) if m else (None, None)


def _cell(cells, col, fighter_idx):
    """Text of fighter `fighter_idx`'s value in column `col` (each cell has one <p> per fighter)."""
    ps = cells[col].select("p")
    return ps[fighter_idx].get_text(strip=True) if len(ps) > fighter_idx else ""


# ── Page parsers (verbatim from notebook 01) ─────────────────────

def get_event_links() -> list[str]:
    """All completed-event URLs, newest first (always fetched live, never cached)."""
    soup = get_soup(f"{BASE}/statistics/events/completed?page=all", use_cache=False)
    seen, links = set(), []
    for a in soup.select("a.b-link"):
        href = a.get("href") or ""
        if "event-details" in href and href not in seen:
            seen.add(href)
            links.append(href)
    return links


def parse_event(url: str, use_cache: bool = True):
    """Return (event_dict, [fight_urls]) for one event page."""
    soup = get_soup(url, use_cache=use_cache)
    name = soup.select_one("h2.b-content__title").get_text(strip=True)
    date = location = None
    for li in soup.select("li.b-list__box-list-item"):
        t = li.get_text(" ", strip=True)
        if t.startswith("Date:"):
            date = t.split("Date:", 1)[1].strip()
        elif t.startswith("Location:"):
            location = t.split("Location:", 1)[1].strip()
    fight_urls = [r.get("data-link")
                  for r in soup.select("tr.b-fight-details__table-row[data-link]")]
    event = {"event_id": url_id(url), "name": name, "date": date, "location": location}
    return event, fight_urls


def parse_fight(url: str, event_id: str, use_cache: bool = True):
    """Return (fight_dict, [stat_rows]) for one fight; (None, []) for an unfought bout."""
    soup = get_soup(url, use_cache=use_cache)

    fighters = []
    for person in soup.select("div.b-fight-details__person"):
        a = person.select_one("a.b-link")
        status = person.select_one("i.b-fight-details__person-status").get_text(strip=True)
        fighters.append({"id": url_id(a.get("href")), "status": status})
    winner_id = next((f["id"] for f in fighters if f["status"] == "W"), None)  # None = draw/NC

    # Upcoming/unfought bouts have no result line — signal 'skip cleanly'.
    info_el = soup.select_one("p.b-fight-details__text")
    if info_el is None:
        return None, []
    info = info_el.get_text(" ", strip=True)

    def grab(pattern):
        m = re.search(pattern, info)
        return m.group(1).strip() if m else None

    fight = {
        "fight_id": url_id(url),
        "event_id": event_id,
        "weight_class": soup.select_one("i.b-fight-details__fight-title").get_text(" ", strip=True),
        "fighter_a_id": fighters[0]["id"],
        "fighter_b_id": fighters[1]["id"],
        "winner_id": winner_id,
        "method": grab(r"Method:\s*(.+?)\s+Round:"),
        "round": grab(r"Round:\s*(\d+)"),
        "time": grab(r"Time:\s*(\d+:\d+)"),
        "referee": grab(r"Referee:\s*(.+)$"),
    }

    # per-fighter stat tables (full-fight totals = row 0)
    stat_rows = []
    tables = soup.select("table.b-fight-details__table")
    if len(tables) >= 2:
        tot = tables[0].select("tbody tr")[0].select("td")
        sig = tables[1].select("tbody tr")[0].select("td")
        for i, f in enumerate(fighters):
            sig_l, sig_a = split_landed_attempted(_cell(tot, 2, i))
            tot_l, tot_a = split_landed_attempted(_cell(tot, 4, i))
            td_l, td_a = split_landed_attempted(_cell(tot, 5, i))
            head_l, head_a = split_landed_attempted(_cell(sig, 3, i))
            body_l, body_a = split_landed_attempted(_cell(sig, 4, i))
            leg_l, leg_a = split_landed_attempted(_cell(sig, 5, i))
            dist_l, dist_a = split_landed_attempted(_cell(sig, 6, i))
            clinch_l, clinch_a = split_landed_attempted(_cell(sig, 7, i))
            grnd_l, grnd_a = split_landed_attempted(_cell(sig, 8, i))
            stat_rows.append({
                "fight_id": fight["fight_id"], "fighter_id": f["id"],
                "knockdowns": _cell(tot, 1, i),
                "sig_str_landed": sig_l, "sig_str_att": sig_a,
                "total_str_landed": tot_l, "total_str_att": tot_a,
                "takedowns_landed": td_l, "takedowns_att": td_a,
                "sub_att": _cell(tot, 7, i), "reversals": _cell(tot, 8, i),
                "control_time": _cell(tot, 9, i),
                "sig_head_landed": head_l, "sig_head_att": head_a,
                "sig_body_landed": body_l, "sig_body_att": body_a,
                "sig_leg_landed": leg_l, "sig_leg_att": leg_a,
                "sig_distance_landed": dist_l, "sig_distance_att": dist_a,
                "sig_clinch_landed": clinch_l, "sig_clinch_att": clinch_a,
                "sig_ground_landed": grnd_l, "sig_ground_att": grnd_a,
            })
    return fight, stat_rows


def parse_fighter(fighter_id: str, use_cache: bool = True):
    """Return a bio dict for one fighter id.

    `record` is the page-header W-L-D — the fighter's FULL professional record,
    including pre-UFC bouts that never appear as fight rows on ufcstats. It is
    display-only; the model's career features are still built from UFC fights.
    """
    soup = get_soup(f"{BASE}/fighter-details/{fighter_id}", use_cache=use_cache)
    rec_el = soup.select_one("span.b-content__title-record")
    record = rec_el.get_text(strip=True).split("Record:", 1)[-1].strip() if rec_el else None
    bio = {
        "fighter_id": fighter_id,
        "name": soup.select_one("span.b-content__title-highlight").get_text(strip=True),
        "height": None, "reach": None, "stance": None, "dob": None,
        "record": record,
    }
    labels = {"Height:": "height", "Reach:": "reach", "STANCE:": "stance", "DOB:": "dob"}
    for li in soup.select("li.b-list__box-list-item"):
        t = li.get_text(" ", strip=True)
        for lbl, key in labels.items():
            if t.startswith(lbl):
                bio[key] = t.split(lbl, 1)[1].strip()
    return bio


# ── Incremental persistence ──────────────────────────────────────

def append_rows(path, rows: list[dict]) -> None:
    """Append dict rows to a CSV, writing the header only when the file is new."""
    if not rows:
        return
    pd.DataFrame(rows).to_csv(path, mode="a", header=not path.exists(), index=False)


def done_ids(path, column: str) -> set[str]:
    """Set of already-saved ids in `column`, or empty set if the file doesn't exist yet."""
    if path.exists():
        return set(pd.read_csv(path, usecols=[column])[column].astype(str))
    return set()


# ── Weekly entrypoint ────────────────────────────────────────────

def update_raw_data(max_events: int | None = None) -> dict:
    """Scrape events/fights/fighters not yet in the raw CSVs. Returns a summary dict.

    Only genuinely-new pages are fetched, so a typical weekly run costs one index
    request plus ~a dozen pages per new event. Already-scraped events are skipped
    via `done_ids`; an all-upcoming card writes nothing and is retried next run.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    event_links = get_event_links()
    todo = event_links if max_events is None else event_links[:max_events]
    done_events = done_ids(FIGHTS_CSV, "event_id")

    new_events = new_fights = upcoming = 0
    skipped: list[tuple[str, str]] = []
    touched: set[str] = set()  # fighters in new fights — their pro records changed

    for ev_url in todo:
        eid = url_id(ev_url)
        if eid in done_events:
            continue
        try:
            # cache disabled: these pages are new, and results must be fresh
            event, fight_urls = parse_event(ev_url, use_cache=False)
        except Exception as e:  # noqa: BLE001 — one bad page must not kill the run
            skipped.append((ev_url, f"event parse: {e}"))
            continue

        fights, stats = [], []
        for f_url in fight_urls:
            try:
                fight, stat_rows = parse_fight(f_url, eid, use_cache=False)
                if fight is None:  # upcoming/unfought bout — nothing to record
                    upcoming += 1
                    continue
                fights.append(fight)
                stats.extend(stat_rows)
            except Exception as e:  # noqa: BLE001
                skipped.append((f_url, f"fight parse: {e}"))

        # Commit the event atomically, but only once it has >=1 completed fight.
        if fights:
            append_rows(FIGHTS_CSV, fights)
            append_rows(STATS_CSV, stats)
            append_rows(EVENTS_CSV, [event])
            done_events.add(eid)
            new_events += 1
            new_fights += len(fights)
            for f in fights:
                touched.update((str(f["fighter_a_id"]), str(f["fighter_b_id"])))
            print(f"  scraped event {event['name']} ({event['date']}): {len(fights)} fights")

    # bios for any fighter referenced in fights but not yet saved (debuts, mostly)
    fights_df = pd.read_csv(FIGHTS_CSV)
    known = done_ids(FIGHTERS_CSV, "fighter_id")
    needed = (set(fights_df["fighter_a_id"].astype(str))
              | set(fights_df["fighter_b_id"].astype(str))) - known

    bios = []
    for fid in sorted(needed):
        try:
            bios.append(parse_fighter(fid, use_cache=False))
        except Exception as e:  # noqa: BLE001
            skipped.append((fid, f"fighter parse: {e}"))
        if len(bios) >= 25:  # flush periodically so progress survives a crash
            append_rows(FIGHTERS_CSV, bios)
            bios = []
    append_rows(FIGHTERS_CSV, bios)

    # refresh the pro record of returning fighters (already-known ones who just
    # fought) so the header W-L-D the app shows doesn't drift a fight stale
    refresh = sorted((touched & known) - needed)
    refreshed = 0
    if refresh:
        fighters_df = pd.read_csv(FIGHTERS_CSV, dtype={"fighter_id": str}).set_index("fighter_id")
        if "record" not in fighters_df.columns:
            fighters_df["record"] = pd.NA
        for fid in refresh:
            try:
                fighters_df.at[fid, "record"] = parse_fighter(fid, use_cache=False).get("record")
                refreshed += 1
            except Exception as e:  # noqa: BLE001
                skipped.append((fid, f"record refresh: {e}"))
        fighters_df.reset_index().to_csv(FIGHTERS_CSV, index=False)

    return {"new_events": new_events, "new_fights": new_fights,
            "upcoming_skipped": upcoming, "new_fighters": len(needed),
            "records_refreshed": refreshed, "failed": skipped}
