"""
Ledger odds auto-filler
=======================
Fetches anytime-try-scorer, line (spreads) and totals odds from The Odds API
per-event endpoint and fills the blank book_odds cells in sgm_ledger.csv.

Notes:
  - Uses the per-event endpoint, which costs more credits than the h2h scan
    (roughly [markets x regions] per event). A full 8-game round with 3
    markets is ~24 credits. The script prints your remaining quota.
  - Records the BEST price across AU books for each leg, and which book.
  - SGM combo rows cannot be auto-filled (no API exposes correlated combo
    prices) - fill those manually in-app or leave blank.

Usage:  python fill_ledger_odds.py
"""
import json, os, re, unicodedata, urllib.request
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
KEY_FILE = os.path.join(HERE, "odds_api_key.txt")
LEDGER = os.path.join(HERE, "sgm_ledger.csv")

ODDS_TEAM = {"Cronulla Sutherland Sharks":"cronulla-sutherland-sharks",
 "Manly Warringah Sea Eagles":"manly-warringah-sea-eagles",
 "Brisbane Broncos":"brisbane-broncos","Canberra Raiders":"canberra-raiders",
 "Canterbury Bulldogs":"canterbury-bankstown-bulldogs",
 "Canterbury-Bankstown Bulldogs":"canterbury-bankstown-bulldogs",
 "Cronulla Sharks":"cronulla-sutherland-sharks","Dolphins":"dolphins",
 "Gold Coast Titans":"gold-coast-titans","Manly Sea Eagles":"manly-warringah-sea-eagles",
 "Melbourne Storm":"melbourne-storm","Newcastle Knights":"newcastle-knights",
 "North Queensland Cowboys":"north-queensland-cowboys","Parramatta Eels":"parramatta-eels",
 "Penrith Panthers":"penrith-panthers","South Sydney Rabbitohs":"south-sydney-rabbitohs",
 "St George Illawarra Dragons":"st-george-illawarra-dragons",
 "St. George Illawarra Dragons":"st-george-illawarra-dragons",
 "Sydney Roosters":"sydney-roosters","New Zealand Warriors":"warriors",
 "Wests Tigers":"wests-tigers"}

def norm_name(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z]", "", s.lower())

def get(url):
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=30) as r:
        remaining = r.headers.get("x-requests-remaining")
        return json.loads(r.read()), remaining

def main():
    if not os.path.exists(KEY_FILE) or not os.path.exists(LEDGER):
        print("Need odds_api_key.txt and sgm_ledger.csv in this folder.")
        return
    key = open(KEY_FILE).read().strip()
    L = pd.read_csv(LEDGER)
    blank = L.book_odds.isna() | (L.book_odds.astype(str).str.strip() == "")
    todo = L[blank & (L.actual.isna() | (L.actual.astype(str) == ""))]
    if not len(todo):
        print("No blank pre-game rows to fill.")
        return
    games = todo[["home", "away"]].drop_duplicates()
    print(f"{len(todo)} blank legs across {len(games)} games")

    events, remaining = get(f"https://api.the-odds-api.com/v4/sports/rugbyleague_nrl/events?apiKey={key}")
    ev_map = {}
    for ev in events:
        h = ODDS_TEAM.get(ev["home_team"], ev["home_team"])
        a = ODDS_TEAM.get(ev["away_team"], ev["away_team"])
        ev_map[(h, a)] = ev["id"]

    filled = 0
    for _, g in games.iterrows():
        eid = ev_map.get((g.home, g.away))
        if not eid:
            print(f"  no upcoming event for {g.home} v {g.away} (already started?)")
            continue
        url = (f"https://api.the-odds-api.com/v4/sports/rugbyleague_nrl/events/{eid}/odds"
               f"?apiKey={key}&regions=au&oddsFormat=decimal"
               f"&markets=player_try_scorer_anytime,spreads,totals")
        try:
            data, remaining = get(url)
        except urllib.error.HTTPError as e:
            print(f"  {g.home} v {g.away}: API refused ({e.code}) - "
                  f"player props may need a paid tier; trying core markets only")
            try:
                data, remaining = get(url.replace("player_try_scorer_anytime,", ""))
            except Exception as e2:
                print(f"    core markets also failed: {e2}")
                continue
        # best price per (market, selection-ish key)
        best = {}
        for bk in data.get("bookmakers", []):
            for mkt in bk.get("markets", []):
                for oc in mkt["outcomes"]:
                    if mkt["key"] == "player_try_scorer_anytime":
                        k = ("anytime_try", norm_name(oc.get("description", oc["name"])))
                    elif mkt["key"] == "spreads":
                        team = ODDS_TEAM.get(oc["name"], oc["name"])
                        k = ("line", team, oc.get("point"))
                    elif mkt["key"] == "totals":
                        k = ("total", oc["name"].lower(), oc.get("point"))
                    else:
                        continue
                    if oc["price"] > best.get(k, (0, ""))[0]:
                        best[k] = (oc["price"], bk["title"])
        # fill ledger rows for this game
        gi = L[(L.home == g.home) & (L.away == g.away) & blank].index
        for i in gi:
            row = L.loc[i]
            k = None
            if row.market == "anytime_try":
                k = ("anytime_try", norm_name(row.selection))
            elif row.market == "line":
                team = row.selection.rsplit(" ", 1)[0]
                pt = float(row.selection.rsplit(" ", 1)[1].replace("+", ""))
                k = ("line", team, pt)
            elif row.market == "total":
                k = ("total", "over", float(row.selection.split()[-1]))
            if k and k in best:
                L.loc[i, "book_odds"] = best[k][0]
                filled += 1
    L.to_csv(LEDGER, index=False)
    print(f"\nFilled {filled} legs -> sgm_ledger.csv")
    print(f"API credits remaining this month: {remaining}")
    still = L[(L.book_odds.isna() | (L.book_odds.astype(str).str.strip() == "")) &
              (L.actual.isna() | (L.actual.astype(str) == ""))]
    if len(still):
        print(f"{len(still)} legs still blank (combos + anything the books don't list) - "
              f"fill manually or leave.")

if __name__ == "__main__":
    main()
