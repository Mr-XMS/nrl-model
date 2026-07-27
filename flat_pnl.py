"""
Flat-stake picks P&L
====================
What if we bet $100 on the model's predicted winner in EVERY 2026 game?

Walk-forward model (trained on 2021-2025) picks each 2026 game; the pick is
"backed" at the cleaned Betfair best-back price (first bounce), 5% commission.
Output: flat_bets.csv for the dashboard + a season summary in the log.

Honest framing: this is the "picks are not profits" exhibit. Expect a slow
bleed - accuracy without price-edge pays the bookmaker's margin every game.
Games before the system went live (~Round 18) are retrospective walk-forward,
not live predictions; the CSV marks the live segment.

Run: force_day = 98 (or weekly from the Monday block).
"""
import glob, os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
LIVE_FROM = pd.Timestamp("2026-07-08")   # first live-logged predictions

def clean_price(b, l):
    return (1.01 < b <= 15) and (b <= l) and ((l - b) / b <= 0.25)

def main():
    odds_file = os.path.join(HERE, "odds", "nrl_2026.csv")
    if not os.path.exists(odds_file):
        import urllib.request
        os.makedirs(os.path.join(HERE, "odds"), exist_ok=True)
        url = "https://betfair-datascientists.github.io/data/assets/NRL_2026_Match_Odds.csv"
        print(f"Flat P&L: downloading Betfair 2026 odds from source...")
        try:
            urllib.request.urlretrieve(url, odds_file)
            print(f"  saved -> odds/nrl_2026.csv")
        except Exception as e:
            print(f"Flat P&L: download failed ({e}) - aborting")
            return
    BF_TEAM = {
        "Brisbane Broncos": "brisbane-broncos", "Canberra Raiders": "canberra-raiders",
        "Canterbury": "canterbury-bankstown-bulldogs",
        "Canterbury Bulldogs": "canterbury-bankstown-bulldogs",
        "Cronulla Sharks": "cronulla-sutherland-sharks", "Dolphins": "dolphins",
        "Gold Coast": "gold-coast-titans", "Gold Coast Titans": "gold-coast-titans",
        "Manly Sea Eagles": "manly-warringah-sea-eagles",
        "Melbourne Storm": "melbourne-storm", "New Zealand Warriors": "warriors",
        "Newcastle Knights": "newcastle-knights",
        "North Qld Cowboys": "north-queensland-cowboys",
        "North Queensland Cowboys": "north-queensland-cowboys",
        "Parramatta Eels": "parramatta-eels", "Penrith Panthers": "penrith-panthers",
        "South Sydney Rabbitohs": "south-sydney-rabbitohs",
        "St George Illawarra Dra": "st-george-illawarra-dragons",
        "St George/Illa Dragons": "st-george-illawarra-dragons",
        "Sydney": "sydney-roosters", "Sydney Roosters": "sydney-roosters",
        "Wests Tigers": "wests-tigers",
    }
    from nrl_model import build_features
    from nrl_player_model import load_lineups, attach_lineups, build_player_features
    from stats_ratings import build_stats_feature
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    results = pd.read_csv(os.path.join(HERE, "nrl_results.csv"), parse_dates=["date"])
    dfx, X, y, mask, _ = build_features(results)
    X = X.reset_index(drop=True)
    X = pd.concat([X, build_player_features(results, attach_lineups(results, load_lineups()))], axis=1)
    X["stats_strength_diff"], _ = build_stats_feature(results)
    cols = list(X.columns)
    ok = X.notna().all(axis=1).values
    tr = mask.values & ok & (dfx.season >= 2021) & (dfx.season < 2026)
    te = mask.values & ok & (dfx.season == 2026)
    sc = StandardScaler().fit(X.loc[tr, cols])
    clf = LogisticRegression(C=0.1, max_iter=2000).fit(sc.transform(X.loc[tr, cols]), y[tr])
    p26 = pd.Series(np.nan, index=X.index)
    p26[te] = clf.predict_proba(sc.transform(X.loc[te, cols]))[:, 1]

    # Betfair prices for 2026
    M = {}
    df = pd.read_csv(odds_file)
    df = df[df.MARKET_TYPE == "MATCH_ODDS"]
    for (eid, mid), g in df.groupby(["EVENT_ID", "MARKET_ID"]):
        if len(g) != 2: continue
        ht, at = g.iloc[0].HOME_TEAM, g.iloc[0].AWAY_TEAM
        if ht not in BF_TEAM or at not in BF_TEAM: continue
        hr = g[g.RUNNER_NAME.apply(lambda r: str(r).split()[0] in ht)]
        ar = g[g.RUNNER_NAME.apply(lambda r: str(r).split()[0] in at)]
        if len(hr) != 1 or len(ar) != 1: continue
        hr, ar = hr.iloc[0], ar.iloc[0]
        try:
            hb, hl = float(hr.BEST_BACK_FIRST_BOUNCE), float(hr.BEST_LAY_FIRST_BOUNCE)
            ab, al = float(ar.BEST_BACK_FIRST_BOUNCE), float(ar.BEST_LAY_FIRST_BOUNCE)
        except (TypeError, ValueError):
            continue
        if not (clean_price(hb, hl) and clean_price(ab, al)): continue
        M[(pd.to_datetime(str(hr.EVENT_DATE)[:10]), BF_TEAM[ht], BF_TEAM[at])] = (hb, ab)

    # live-era prices: best scanned book price strictly BEFORE kickoff
    live_px = {}
    hp = os.path.join(HERE, "odds_history.csv")
    fp = os.path.join(HERE, "forecast_history.csv")
    if os.path.exists(hp) and os.path.exists(fp):
        H = pd.read_csv(hp)
        H["scan_time"] = pd.to_datetime(H.scan_time)
        FH = pd.read_csv(fp)
        ko = {}
        for fr in FH.dropna(subset=["kickoff"]).itertuples():
            try:
                t = pd.to_datetime(fr.kickoff)
                t = t.tz_convert("Australia/Sydney") if t.tzinfo else t.tz_localize("UTC").tz_convert("Australia/Sydney")
                ko[(fr.home_team, fr.away_team)] = t.tz_localize(None)
            except Exception:
                continue
        for (h_, a_), g in H.groupby(["home", "away"]):
            k = ko.get((h_, a_))
            if k is None:
                continue
            pre = g[g.scan_time < k]
            pre = pre[(pre.odds > 1.01) & (pre.odds <= 15)]
            if len(pre):
                bh = pre[pre.team == h_].odds.max()
                ba = pre[pre.team == a_].odds.max()
                if bh == bh and ba == ba:
                    live_px[(h_, a_)] = (float(bh), float(ba))
        print(f"Flat P&L: {len(live_px)} live games priced from scanner history "
              f"(best-of-books, pre-kickoff)")

    rows = []
    for i, r in enumerate(results.itertuples()):
        if p26[i] != p26[i] or r.home_score == r.away_score: continue
        odds = None
        for dd in (0, 1, -1, 2, -2):
            k = (r.date + pd.Timedelta(days=dd), r.home_team, r.away_team)
            if k in M: odds = M[k]; break
        if odds is None: continue
        src_live = (r.home_team, r.away_team) in live_px and r.date >= LIVE_FROM
        if src_live:
            odds = live_px[(r.home_team, r.away_team)]
        pick_home = p26[i] > 0.5
        pick = r.home_team if pick_home else r.away_team
        price = odds[0] if pick_home else odds[1]
        won = (r.home_score > r.away_score) == pick_home
        comm = 1.0 if src_live else 0.95      # books pay face value; Betfair nets 5%
        pl = 100 * comm * (price - 1) if won else -100.0
        rows.append(dict(date=r.date, home=r.home_team, away=r.away_team,
                         pick=pick, prob=round(float(max(p26[i], 1-p26[i])), 3),
                         odds=price, won=int(won), pl=round(pl, 2),
                         segment="live" if r.date >= LIVE_FROM else "backtest"))
    F = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    F["cum_pl"] = F.pl.cumsum().round(2)
    F.to_csv(os.path.join(HERE, "flat_bets.csv"), index=False)

    for seg in ("backtest", "live", None):
        s = F if seg is None else F[F.segment == seg]
        if not len(s): continue
        label = seg or "FULL SEASON"
        print(f"Flat P&L [{label}]: {len(s)} bets, {s.won.mean():.0%} won, "
              f"staked ${len(s)*100:,}, P&L ${s.pl.sum():+,.0f} "
              f"({s.pl.sum()/(len(s)*100):+.1%} ROI)")
    print(f"Season equity range: best ${F.cum_pl.max():+,.0f}, "
          f"worst ${F.cum_pl.min():+,.0f} -> flat_bets.csv")

if __name__ == "__main__":
    main()
