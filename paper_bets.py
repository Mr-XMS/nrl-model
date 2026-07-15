"""
Paper betting ledger
====================
Each Tuesday run: builds the round's value slate (live blend vs best scanned
price, EV > 6%, quarter-Kelly stakes from a fixed $100 paper bankroll) and
records it to bets_ledger.csv BEFORE kickoff. Each run also grades any
completed bets. This is the season's official paper P&L.

Run automatically from nrl_auto.sh. No real money is involved.
"""
import os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
LEDGER = os.path.join(HERE, "bets_ledger.csv")
BANKROLL = 100.0
MIN_EV = 0.06            # supported by the 2022-26 threshold backtest
KELLY_FRACTION = 0.25


KEY = ["date_key", "home", "away", "side"]

def _load_ledger():
    L = pd.read_csv(LEDGER, parse_dates=["date"])
    L["date_key"] = L.date.dt.strftime("%Y-%m-%d")
    before = len(L)
    L = L.drop_duplicates(subset=KEY, keep="first").reset_index(drop=True)
    if len(L) < before:
        print(f"Paper bets: removed {before - len(L)} duplicate rows (self-heal)")
    return L

def grade():
    if not os.path.exists(LEDGER):
        return
    L = _load_ledger()
    results = pd.read_csv(os.path.join(HERE, "nrl_results.csv"), parse_dates=["date"])
    n = 0
    for i in L[L.result.isna()].index:
        r = L.loc[i]
        m = results[(results.home_team == r.home) & (results.away_team == r.away) &
                    ((results.date - r.date).abs() <= pd.Timedelta(days=2))]
        if not len(m):
            continue
        g = m.iloc[-1]
        if g.home_score == g.away_score:
            L.loc[i, ["result", "profit"]] = ["push", 0.0]
        else:
            winner = g.home_team if g.home_score > g.away_score else g.away_team
            won = (r.side == winner)
            L.loc[i, "result"] = "win" if won else "loss"
            L.loc[i, "profit"] = round(r.stake * (r.odds - 1), 2) if won else -r.stake
        n += 1
    L.drop(columns="date_key").to_csv(LEDGER, index=False)
    done = L.dropna(subset=["profit"])
    if len(done):
        print(f"Paper bets: graded {n} new | season: {len(done)} bets, "
              f"staked ${done.stake.sum():.0f}, P&L ${done.profit.sum():+.0f} "
              f"({done.profit.sum()/done.stake.sum():+.1%} ROI)")


def place():
    comp = pd.read_csv(os.path.join(HERE, "model_comparison.csv"), parse_dates=["date"])
    hpath = os.path.join(HERE, "odds_history.csv")
    if not os.path.exists(hpath):
        return print("Paper bets: no odds history yet - nothing placed")
    H = pd.read_csv(hpath)
    H["scan_time"] = pd.to_datetime(H.scan_time)
    latest = H[H.scan_time == H.scan_time.max()]
    up = comp[comp.actual_home_win.isna()]
    lg = lambda q: np.log(np.clip(q, 1e-6, 1-1e-6) / (1 - np.clip(q, 1e-6, 1-1e-6)))
    cands = []
    for r in up.itertuples():
        gh = latest[(latest.home == r.home_team) & (latest.away == r.away_team)]
        if not len(gh):
            continue
        bh, ba = gh[gh.team == r.home_team].odds.max(), gh[gh.team == r.away_team].odds.max()
        if not (1.01 < bh <= 15 and 1.01 < ba <= 15):
            continue
        bkh = gh[(gh.team == r.home_team) & (gh.odds == bh)].book.iloc[0]
        bka = gh[(gh.team == r.away_team) & (gh.odds == ba)].book.iloc[0]
        ih, ia = 1/bh, 1/ba
        blend = 1/(1+np.exp(-0.5*(lg(float(r.p_base)) + lg(ih/(ih+ia)))))
        for side, prob, odds, book in ((r.home_team, blend, bh, bkh),
                                       (r.away_team, 1-blend, ba, bka)):
            ev = prob * odds - 1
            if ev > MIN_EV:
                kelly = (prob*odds - 1) / (odds - 1)
                cands.append(dict(placed=pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
                    date=r.date, home=r.home_team, away=r.away_team, side=side,
                    prob=round(prob, 4), odds=odds, book=book, ev=round(ev, 4),
                    stake=0.0, result=np.nan, profit=np.nan, _kelly=kelly))
    if not cands:
        return print("Paper bets: no edges clear the 6% bar - nothing placed this round")
    C = pd.DataFrame(cands)
    C["stake"] = (C._kelly * KELLY_FRACTION * BANKROLL).round(0)
    if C.stake.sum() > BANKROLL:
        C["stake"] = (C.stake * BANKROLL / C.stake.sum()).round(0)
    C = C[C.stake >= 1].drop(columns="_kelly")
    C["date_key"] = pd.to_datetime(C.date).dt.strftime("%Y-%m-%d")
    if os.path.exists(LEDGER):
        old = _load_ledger()
        C = C[~C.set_index(KEY).index.isin(old.set_index(KEY).index)]
        n_new = len(C)
        C = pd.concat([old, C], ignore_index=True)
    else:
        n_new = len(C)
    C = C.drop_duplicates(subset=KEY, keep="first")
    C.drop(columns="date_key").to_csv(LEDGER, index=False)
    print(f"Paper bets: {n_new} new bets recorded -> bets_ledger.csv")


if __name__ == "__main__":
    grade()
    if len(sys.argv) > 1 and sys.argv[1] == "place":
        place()
