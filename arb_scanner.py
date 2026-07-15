"""
Bookmaker scanner: arbitrage + best-price value + price history
================================================================
Pulls NRL head-to-head odds from Australian bookmakers via The Odds API,
then reports three things:

  1. ARBITRAGE  - combined best-price implied prob < 100% across books
  2. VALUE      - best available price vs the model's probability
                  (reads latest model probs from model_comparison.csv)
  3. LOGGING    - appends every book's price to odds_history.csv, so we can
                  measure whether early prices beat the close over the season

Setup (one-time):
  1. Get a free API key at https://the-odds-api.com
  2. Create a file named odds_api_key.txt in this folder containing the key

Usage:
  python arb_scanner.py            # scan now
Run it a few times per week: after team lists (Tue night), Thursday, and
just before each game day - late scratchings are when discrepancies appear.
"""
import json, os, sys, urllib.request
from datetime import datetime
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
KEY_FILE = os.path.join(HERE, "odds_api_key.txt")
HISTORY = os.path.join(HERE, "odds_history.csv")
COMPARE = os.path.join(HERE, "model_comparison.csv")

ODDS_TEAM = {"Cronulla Sutherland Sharks":"cronulla-sutherland-sharks",
 "Manly Warringah Sea Eagles":"manly-warringah-sea-eagles",
 "St. George Illawarra Dragons":"st-george-illawarra-dragons",
 "Canterbury-Bankstown Bulldogs":"canterbury-bankstown-bulldogs",
 "Brisbane Broncos":"brisbane-broncos","Canberra Raiders":"canberra-raiders",
 "Canterbury Bulldogs":"canterbury-bankstown-bulldogs","Cronulla Sharks":"cronulla-sutherland-sharks",
 "Cronulla-Sutherland Sharks":"cronulla-sutherland-sharks","Dolphins":"dolphins",
 "Gold Coast Titans":"gold-coast-titans","Manly Sea Eagles":"manly-warringah-sea-eagles",
 "Manly-Warringah Sea Eagles":"manly-warringah-sea-eagles","Melbourne Storm":"melbourne-storm",
 "Newcastle Knights":"newcastle-knights","North Queensland Cowboys":"north-queensland-cowboys",
 "Parramatta Eels":"parramatta-eels","Penrith Panthers":"penrith-panthers",
 "South Sydney Rabbitohs":"south-sydney-rabbitohs","St George Illawarra Dragons":"st-george-illawarra-dragons",
 "Sydney Roosters":"sydney-roosters","New Zealand Warriors":"warriors","Wests Tigers":"wests-tigers"}


def main():
    if not os.path.exists(KEY_FILE):
        print("No API key found. Get a free key at https://the-odds-api.com and save it")
        print(f"as a single line in: {KEY_FILE}")
        return
    key = open(KEY_FILE).read().strip()
    url = (f"https://api.the-odds-api.com/v4/sports/rugbyleague_nrl/odds/"
           f"?apiKey={key}&regions=au&markets=h2h&oddsFormat=decimal")
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            events = json.loads(r.read())
    except Exception as e:
        print(f"API call failed: {e}")
        return
    print(f"{len(events)} upcoming NRL games, scanning...\n")

    # model probs from the latest trial log entries
    model = {}
    if os.path.exists(COMPARE):
        cl = pd.read_csv(COMPARE)
        for r in cl.itertuples():
            model[(r.home_team, r.away_team)] = getattr(r, "p_base", None)

    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    hist_rows = []
    for ev in events:
        home_raw, away_raw = ev["home_team"], ev["away_team"]
        h = ODDS_TEAM.get(home_raw, home_raw)
        a = ODDS_TEAM.get(away_raw, away_raw)
        best = {h: (0, ""), a: (0, "")}
        for bk in ev.get("bookmakers", []):
            for mkt in bk.get("markets", []):
                if mkt["key"] != "h2h":
                    continue
                for oc in mkt["outcomes"]:
                    team = ODDS_TEAM.get(oc["name"], oc["name"])
                    if team in best and oc["price"] > best[team][0]:
                        best[team] = (oc["price"], bk["title"])
                    hist_rows.append(dict(scan_time=stamp, kickoff=ev["commence_time"][:10],
                                          home=h, away=a, book=bk["title"],
                                          team=team, odds=oc["price"]))
        bh, ba = best[h], best[a]
        if bh[0] <= 1 or ba[0] <= 1:
            continue
        total_implied = 1/bh[0] + 1/ba[0]
        print(f"{h} v {a}")
        print(f"  best: {h} @ {bh[0]:.2f} ({bh[1]})  |  {a} @ {ba[0]:.2f} ({ba[1]})")
        print(f"  combined implied: {total_implied:.1%}", end="")
        if total_implied < 1.0:
            margin = (1 - total_implied) * 100
            s1 = 100 / bh[0] / (100/bh[0] + 100/ba[0])
            print(f"   *** ARBITRAGE {margin:.2f}% *** "
                  f"(stake split {s1:.0%} / {1-s1:.0%})")
        else:
            print()
        mp = model.get((h, a))
        if mp is not None and mp == mp:
            import numpy as np
            ih, ia = 1/bh[0], 1/ba[0]
            mkt = ih / (ih + ia)                       # live de-vigged market
            lg = lambda q: np.log(np.clip(q, 1e-6, 1-1e-6) / (1 - np.clip(q, 1e-6, 1-1e-6)))
            mp = 1 / (1 + np.exp(-0.5 * (lg(float(mp)) + lg(mkt))))   # fresh blend
            ev_h = float(mp) * bh[0] - 1
            ev_a = (1 - float(mp)) * ba[0] - 1
            side, evv, price, bk = (h, ev_h, bh[0], bh[1]) if ev_h > ev_a else (a, ev_a, ba[0], ba[1])
            flag = "  >>> VALUE" if evv > 0.04 else ""
            print(f"  live blend {float(mp):.0%} home -> best-price EV: {side} {evv:+.1%} "
                  f"@ {price:.2f} ({bk}){flag}")
        print()

    if hist_rows:
        newH = pd.DataFrame(hist_rows)
        if os.path.exists(HISTORY):
            newH = pd.concat([pd.read_csv(HISTORY), newH], ignore_index=True)
        newH.to_csv(HISTORY, index=False)
        print(f"Price snapshot logged: {len(hist_rows)} quotes -> odds_history.csv")


if __name__ == "__main__":
    main()
