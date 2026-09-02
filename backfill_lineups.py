"""
backfill_lineups.py - recover lineup history when the RLP scrape is dead.
=========================================================================
`scrape_players.py` reads fielded 17s from rugbyleagueproject.org match
pages. Those pages began returning zero bytes in 2026, so lineups.jsonl
stopped at 2026-07-12 while results continued to 2026-08-30. Rounds 22-26
have no lineup records at all.

That gap silently degrades three of the six model layers:

  * spine continuity - last_spine[team] holds each side's spine from the
    last match WITH a lineup, so a Round 27 spine is compared against a
    seven-week-old one, never matches, and resets to 1. This is why every
    fixture logged "spine run 1 v 1".
  * player ratings   - fitted from lineups.jsonl, so no player has accrued
    an appearance since July.
  * return windows   - in_return_window() reads the same appearance dates.

This script rebuilds the missing records from the NRL's own match centre
API, which predict_teamlists.py already calls every week for team lists.
No new upstream dependency is introduced.

The important part is identity. lineups.jsonl is keyed on RLP player ids
and the ratings model indexes on them, so NRL playerIds cannot simply be
written in their place. Every named player is resolved through the same
match_player() name index the live forecast uses. A player who cannot be
resolved is written with a null id and counted, never guessed, because a
wrong id silently corrupts a career rating and that is worse than a
missing appearance.

Usage:
    python3 backfill_lineups.py --rounds 22-26           # dry run
    python3 backfill_lineups.py --rounds 22-26 --write
    python3 backfill_lineups.py --round 27 --write       # after a round
"""
import json
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from predict_teamlists import (  # noqa: E402
    fetch_round_teamlists, build_name_index, match_player)
from nrl_player_model import SPINE, load_lineups  # noqa: E402

LINEUPS = os.path.join(HERE, "lineups.jsonl")
RESULTS = os.path.join(HERE, "nrl_results.csv")
SEASON = 2026

# The NRL match centre and RLP label positions differently. Only the four
# spine positions have to agree for spine continuity to work, but the rest
# are mapped so bench/forward composition stays comparable across sources.
POS = {
    "Five-Eighth": "Five-eighth",
    "Winger": "Wing",
    "Prop": "Front row",
    "2nd Row": "Second row",
    "Interchange": "Bench",
}


def parse_range(spec):
    if "-" in spec:
        a, b = spec.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(spec)]


def existing_keys():
    keys = set()
    if not os.path.exists(LINEUPS):
        return keys
    with open(LINEUPS) as f:
        for line in f:
            if not line.strip():
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue
            keys.add((str(d.get("date"))[:10], d.get("home_team"),
                      d.get("away_team")))
    return keys


def synth_match_id(date, home, away):
    """Stable negative id, so backfilled records never collide with RLP's."""
    return -abs(hash(f"{date}|{home}|{away}")) % 10_000_000 - 900_000_000


def build(rounds, write):
    lineups = load_lineups()
    results = pd.read_csv(RESULTS, parse_dates=["date"])
    name_index = build_name_index(lineups, results)
    have = existing_keys()

    print(f"name index: {len(name_index):,} known players")
    print(f"lineups.jsonl: {len(have):,} existing records\n")

    new_records, unresolved, skipped = [], [], 0

    for rnd in rounds:
        try:
            fixtures = fetch_round_teamlists(SEASON, rnd)
        except Exception as e:
            print(f"Round {rnd}: fetch failed ({e})")
            continue

        for fx in fixtures:
            date = str(fx.get("kickoff"))[:10]
            h, a = fx["home"], fx["away"]
            if (date, h, a) in have:
                skipped += 1
                continue
            if fx.get("state") not in ("FullTime", "PostGame"):
                print(f"  Round {rnd}: {h} v {a} not complete "
                      f"({fx.get('state')}), skipping")
                continue

            players, miss = [], 0
            for key, team in (("home", h), ("away", a)):
                named = fx["teams"].get(key, [])
                if len(named) < 13:
                    miss = 99
                    break
                for first, last, pos, _nrl in named:
                    pid = match_player(first, last, team, name_index)
                    if pid is None:
                        miss += 1
                        unresolved.append(f"{team}: {first} {last}")
                    players.append(dict(
                        match_id=synth_match_id(date, h, a),
                        side=key, player_id=pid,
                        player=f"{first} {last}",
                        position=POS.get(pos, pos)))

            if miss >= 99 or not players:
                print(f"  Round {rnd}: {h} v {a} incomplete team list, skipped")
                continue

            spine = sum(1 for p in players if p["position"] in SPINE)
            new_records.append(dict(
                match_id=synth_match_id(date, h, a), season=SEASON,
                date=date, home_team=h, away_team=a,
                players=players, n_players=len(players)))
            print(f"  Round {rnd}: {date}  {h[:26]:26s} v {a[:26]:26s}  "
                  f"{len(players)} players, {spine} spine, "
                  f"{miss} unresolved")

    print(f"\n{len(new_records)} new record(s), {skipped} already present")
    if unresolved:
        print(f"{len(unresolved)} unresolved player(s) written with null id:")
        for u in unresolved[:15]:
            print(f"    {u}")
        if len(unresolved) > 15:
            print(f"    ... and {len(unresolved) - 15} more")

    if not new_records:
        return
    if not write:
        print("\ndry run. re-run with --write to append.")
        return

    with open(LINEUPS, "a") as f:
        for rec in new_records:
            f.write(json.dumps(rec) + "\n")
    print(f"\nappended {len(new_records)} record(s) -> "
          f"{os.path.basename(LINEUPS)}")


def main():
    args = sys.argv[1:]
    write = "--write" in args
    rounds = None
    for i, a in enumerate(args):
        if a in ("--rounds", "--round") and i + 1 < len(args):
            rounds = parse_range(args[i + 1])
    if not rounds:
        print(__doc__)
        return
    build(rounds, write)


if __name__ == "__main__":
    main()
