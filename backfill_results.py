"""
backfill_results.py - fallback results ingest when the primary feed stalls.
==========================================================================
Rugby League Project is the primary source for match results. On 2026-09-01
it was found to have stopped publishing scores after Round 25: fixtures for
Round 26 were listed with empty score cells six days after those games were
played. The pipeline correctly treated the round as unplayed, which stalled
grading and stopped the public archive advancing.

This script reads the same results from the Wikipedia season-results page,
which is structured, well maintained during the season, and independent of
RLP. It writes ONLY rows that are missing from nrl_results.csv and never
edits an existing row, so running it when the primary source is healthy is
a no-op.

Usage:
    python3 backfill_results.py            # report what is missing
    python3 backfill_results.py --write    # append missing rows

Verify the numbers before writing. This is a manual recovery tool, not part
of the scheduled run.
"""
import json
import os
import re
import sys
import urllib.request

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "nrl_results.csv")
SEASON = 2026
API = ("https://en.wikipedia.org/w/api.php?action=parse&page="
       "{season}_NRL_season_results&prop=wikitext&format=json&formatversion=2")

# Wikipedia article titles -> the slugs used throughout this project.
TEAM = {
    "Brisbane Broncos": "brisbane-broncos",
    "Melbourne Storm": "melbourne-storm",
    "Manly Warringah Sea Eagles": "manly-warringah-sea-eagles",
    "St. George Illawarra Dragons": "st-george-illawarra-dragons",
    "Penrith Panthers": "penrith-panthers",
    "Canterbury-Bankstown Bulldogs": "canterbury-bankstown-bulldogs",
    "Gold Coast Titans": "gold-coast-titans",
    "South Sydney Rabbitohs": "south-sydney-rabbitohs",
    "Sydney Roosters": "sydney-roosters",
    "Dolphins (NRL)": "dolphins",
    "Dolphins": "dolphins",
    "North Queensland Cowboys": "north-queensland-cowboys",
    "Wests Tigers": "wests-tigers",
    "New Zealand Warriors": "warriors",
    "Newcastle Knights": "newcastle-knights",
    "Parramatta Eels": "parramatta-eels",
    "Cronulla-Sutherland Sharks": "cronulla-sutherland-sharks",
    "Canberra Raiders": "canberra-raiders",
}
MONTHS = {m: i + 1 for i, m in enumerate(
    ["January", "February", "March", "April", "May", "June", "July",
     "August", "September", "October", "November", "December"])}
DAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
        "Saturday", "Sunday")


def wikitext(season):
    req = urllib.request.Request(API.format(season=season),
                                 headers={"User-Agent": "nrl-model/1.0 research"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)["parse"]["wikitext"]


def parse_round(seg, month_range, season):
    """One '=== Round N ===' section into result rows."""
    out = []
    for b in seg.split("|- style=")[1:]:
        teams = re.findall(r"\[\[([^\]|]+)(?:\|[^\]]*)?\]\]", b)
        score = re.search(r"\|(\d+)[–-](\d+)\*?\n", b)
        day = re.search(r"\|((?:%s))[^\n]*" % "|".join(DAYS), b)
        if not (score and day and len(teams) >= 2):
            continue
        home, away = TEAM.get(teams[0]), TEAM.get(teams[1])
        if not (home and away):
            print(f"  ! unmapped team: {teams[0]} / {teams[1]}")
            continue
        # section headers read "August 27–30" or "August 27 – September 2"
        wd = DAYS.index(day.group(1))
        start_m, start_d = month_range
        base = pd.Timestamp(year=season, month=start_m, day=start_d)
        date = base + pd.Timedelta(days=(wd - base.dayofweek) % 7)
        out.append(dict(season=season, date=date.date().isoformat(),
                        home_team=home, away_team=away,
                        home_score=int(score.group(1)),
                        away_score=int(score.group(2))))
    return out


def scrape(season, only_round=None):
    w = wikitext(season)
    rows = []
    for m in re.finditer(r"===\s*(Round \d+)\s*===\s*\n'''([A-Z][a-z]+) (\d+)",
                         w):
        name, month, day = m.group(1), m.group(2), int(m.group(3))
        if only_round and name != only_round:
            continue
        nxt = w.find("===", m.end())
        seg = w[m.start():nxt if nxt > 0 else len(w)]
        for r in parse_round(seg, (MONTHS[month], day), season):
            r["round"] = name
            rows.append(r)
    return pd.DataFrame(rows)


def main():
    write = "--write" in sys.argv
    rnd = next((a.split("=")[1] for a in sys.argv if a.startswith("--round=")),
               None)
    new = scrape(SEASON, rnd)
    if new.empty:
        print("nothing parsed")
        return

    cur = pd.read_csv(RESULTS, parse_dates=["date"])
    cur["date"] = cur["date"].dt.date.astype(str)
    key = ["date", "home_team", "away_team"]
    missing = new[~new.set_index(key).index.isin(cur.set_index(key).index)]

    print(f"parsed {len(new)} matches, {len(missing)} missing from "
          f"{os.path.basename(RESULTS)}\n")
    for _, r in missing.iterrows():
        print(f"  {r['date']}  {r['round']:10s} {r.home_team:30s} "
              f"{r.home_score:3d}-{r.away_score:<3d} {r.away_team}")

    if not len(missing):
        return
    if not write:
        print("\ndry run. re-run with --write to append.")
        return

    out = pd.concat([cur, missing], ignore_index=True)
    out["date"] = pd.to_datetime(out["date"])
    out = out.sort_values("date").reset_index(drop=True)
    out.to_csv(RESULTS, index=False)
    print(f"\nappended {len(missing)} rows -> {os.path.basename(RESULTS)}")


if __name__ == "__main__":
    main()
