"""
Origin backup study (research run)
==================================
Question: do players backing up from State of Origin underperform in their
next club game - and how often are they rested entirely?

Method:
  1. Scrape Origin team lists 2021-2026 from rugbyleagueproject.org
  2. Match Origin players to our nrl.com stats identities by name
  3. Per-game "value" = the stats model's stage-1 score for that game
  4. Compare each player's club game within 6 days after an Origin match
     against their same-season baseline (windows around Origin excluded)

Run in the cloud: force_day = 99. Output lands in automation.log.
"""
import json, os, re, sys, time, unicodedata, urllib.request
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
UA = {"User-Agent": "Mozilla/5.0 (research; contact via site form)"}
MONTHS = {m: i+1 for i, m in enumerate(["January","February","March","April","May","June",
          "July","August","September","October","November","December"])}

def norm(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii","ignore").decode()
    return re.sub(r"[^a-z]", "", s.lower())

def fetch(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30) as r:
        return r.read().decode("utf-8", errors="ignore")

def scrape_origin(years=range(2021, 2027)):
    """-> list of (date, winner_pids, loser_pids) via nrl.com (competition 116)"""
    games = []
    for y in years:
        try:
            d = json.loads(fetch(f"https://www.nrl.com/draw/data?competition=116&season={y}"))
        except Exception as e:
            print(f"  {y}: draw fetch failed ({e})"); continue
        n = 0
        for f in d.get("fixtures", []):
            if f.get("type") != "Match" or f.get("matchState") not in ("FullTime", "PostGame"):
                continue
            hs, as_ = f["homeTeam"].get("score"), f["awayTeam"].get("score")
            if hs is None or as_ is None or hs == as_:
                continue
            url = "https://www.nrl.com" + f["matchCentreUrl"].rstrip("/") + "/data"
            try:
                mc = json.loads(fetch(url))
            except Exception:
                continue
            ko = mc.get("startTime") or mc.get("kickOffTime")
            if not ko:
                continue
            date = pd.to_datetime(ko).tz_localize(None).normalize()
            sides = {}
            for side in ("homeTeam", "awayTeam"):
                sides[side] = {p["playerId"] for p in (mc.get(side, {}).get("players") or [])
                               if p.get("playerId")}
            win_side = "homeTeam" if hs > as_ else "awayTeam"
            lose_side = "awayTeam" if hs > as_ else "homeTeam"
            if len(sides[win_side]) + len(sides[lose_side]) >= 30:
                games.append((date, sides[win_side], sides[lose_side]))
                n += 1
            time.sleep(0.4)
        print(f"  {y}: {n} Origin matches")
    return games


def main():
    print("Origin backup study")
    print("Scraping Origin team lists...")
    origin = scrape_origin()
    print(f"Total Origin matches: {len(origin)}")
    if len(origin) < 10:
        print("Too few Origin matches scraped - aborting")
        return

    from stats_ratings import _load_player_games, _stage1_weights
    PG, stat_cols = _load_player_games()
    sc, w = _stage1_weights(PG, stat_cols)
    PG = PG.copy()
    PG["value"] = sc.transform(PG[stat_cols].fillna(0)) @ w

    # discover schema rather than assume it
    id_col = next((c for c in ("playerId", "player_id", "pid", "id") if c in PG.columns), None)
    date_col = next((c for c in ("date", "kickoff", "game_date", "match_date") if c in PG.columns), None)
    if id_col is None or date_col is None:
        print("SCHEMA MISMATCH - PG columns are:", list(PG.columns))
        print("Edit origin_study.py candidates to match, rerun.")
        return
    print(f"(using id column '{id_col}', date column '{date_col}')")
    dt = pd.to_datetime(PG[date_col])
    try:
        dt = dt.dt.tz_localize(None)
    except TypeError:
        pass
    PG["gdate"] = dt.dt.normalize()

    all_origin_ids = set().union(*[w | l for _, w, l in origin])
    known = set(PG[id_col].unique())
    matched = all_origin_ids & known
    print(f"Origin players matched to stats identities: {len(matched)} "
          f"of {len(all_origin_ids)}")

    deltas, gaps, wons, backed, rested = [], [], [], 0, 0
    for odate, win_p, lose_p in origin:
        for pid in (win_p | lose_p) & known:
            won_origin = pid in win_p
            mine = PG[PG[id_col] == pid]
            season = mine[mine.gdate.dt.year == odate.year]
            if len(season) < 6:
                continue
            post = season[(season.gdate > odate) & (season.gdate <= odate + pd.Timedelta(days=6))]
            # baseline: same season, outside +/-7d of ANY origin date that year
            odates = [d for d, _, _ in origin if d.year == odate.year]
            base_mask = np.ones(len(season), dtype=bool)
            for od in odates:
                base_mask &= ~((season.gdate > od - pd.Timedelta(days=7)) &
                               (season.gdate <= od + pd.Timedelta(days=7)))
            base = season[base_mask]
            if not len(base):
                continue
            if len(post):
                backed += 1
                deltas.append(post.value.iloc[0] - base.value.mean())
                gaps.append(int((post.gdate.iloc[0] - odate).days))
                wons.append(won_origin)
            else:
                rested += 1

    d = np.array(deltas)
    print(f"\nBack-up appearances: {backed} | rested/absent next club game: {rested} "
          f"({rested/(backed+rested):.0%} rest rate)")
    if len(d) < 30:
        print("Too few back-up games for a verdict")
        return
    rng = np.random.default_rng(5)
    boots = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(3000)]
    lo, hi = np.percentile(boots, [5, 95])
    print(f"Mean next-club-game value delta: {d.mean():+.3f} "
          f"(90% CI [{lo:+.3f}, {hi:+.3f}]) across {len(d)} games")
    print(f"  (player per-game value SD for scale: {PG.value.std():.2f})")
    G = pd.DataFrame({"delta": d, "gap_days": gaps})
    print("\nBy days between Origin and club game:")
    print(G.groupby("gap_days").agg(n=("delta","size"), mean_delta=("delta","mean"))
           .round(3).to_string())
    W = np.array(wons)
    dw, dl = d[W], d[~W]
    print(f"\nBy Origin result:")
    print(f"  after a WIN : n={len(dw)}, mean delta {dw.mean():+.3f}")
    print(f"  after a LOSS: n={len(dl)}, mean delta {dl.mean():+.3f}")
    diff_boots = [dw[rng.integers(0, len(dw), len(dw))].mean() -
                  dl[rng.integers(0, len(dl), len(dl))].mean() for _ in range(3000)]
    lo2, hi2 = np.percentile(diff_boots, [5, 95])
    print(f"  win-minus-loss difference: {dw.mean()-dl.mean():+.3f} "
          f"(90% CI [{lo2:+.3f}, {hi2:+.3f}])")
    print("\nInterpretation guide: CI below zero = backing up genuinely hurts "
          "performance (candidate ORIGIN flag for 2027); CI straddling zero = "
          "the rest-rate number above is the real effect (absence, not fatigue).")

if __name__ == "__main__":
    main()
