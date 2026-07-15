"""
SGM Simulator v1
================
Player-level NRL match simulator for pricing correlated same-game-multi legs.

Engine:
  1. Team try counts ~ Poisson(rate) where log(rate) is a regression on
     Elo edge, home advantage, lineup stats-strength edge, and wet weather.
  2. Conversions ~ Binomial(tries, team conversion rate);
     penalty goals & field goals ~ empirical Poisson rates.
  3. Each simulated try is attributed to a named player via shrunken
     career try-share weights (position-informed prior).

Usage:
  python sgm_simulator.py validate            # walk-forward validation
  python sgm_simulator.py price <round>       # simulate this week's games,
                                              # print win/line/total/scorer prices

Known v1 limitations (documented, to fine-tune during paper-testing):
  - no minutes weighting (bench players slightly over-credited)
  - team-level conversion rate (no kicker identity)
  - wet effect on scorer mix not modelled (only on totals/leveller)
"""
import json, os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

N_SIMS = 20000
RNG = np.random.default_rng(7)

# ---------------------------------------------------------------- data build
def build_team_match_table():
    """One row per team per match (2021+): tries, goals stats, features."""
    from nrl_model import build_features
    from weather_features import add_wet_column
    recs = [json.loads(l) for l in open(os.path.join(HERE, "player_stats.jsonl"))]
    recs = [r for r in recs if r.get("players") and r.get("home_score") is not None]
    from stats_ratings import NICK
    rows = []
    for r in recs:
        date = pd.to_datetime(r["kickoff"]).tz_localize(None).normalize()
        agg = {}
        for p in r["players"]:
            s = agg.setdefault(p["side"], dict(tries=0, conv=0, convAtt=0, pen=0, fg=0))
            s["tries"] += p.get("tries", 0) or 0
            s["conv"] += p.get("conversions", 0) or 0
            s["convAtt"] += p.get("conversionAttempts", 0) or 0
            s["pen"] += p.get("penaltyGoals", 0) or 0
            s["fg"] += p.get("fieldGoals", 0) or 0
        for side in ("home", "away"):
            if side not in agg:
                continue
            rows.append(dict(date=date, side=side,
                             team=NICK[r["home"] if side == "home" else r["away"]],
                             opp=NICK[r["away"] if side == "home" else r["home"]],
                             **agg[side]))
    T = pd.DataFrame(rows)

    results = pd.read_csv(os.path.join(HERE, "nrl_results.csv"), parse_dates=["date"])
    dfx, X, y, mask, elo = build_features(results)
    from stats_ratings import build_stats_feature
    stats_feat, stats_rt = build_stats_feature(results)
    wet = add_wet_column(results)
    # map (date, home, away) -> features
    fmap = {}
    for i, r in enumerate(results.itertuples()):
        fmap[(r.date, r.home_team, r.away_team)] = dict(
            elo_diff=X.iloc[i]["elo_diff"], stats_diff=stats_feat[i], wet=wet[i])

    feats = []
    for r in T.itertuples():
        key = (r.date, r.team, r.opp) if r.side == "home" else (r.date, r.opp, r.team)
        f = None
        for dd in (0, 1, -1):
            k = (key[0] + pd.Timedelta(days=dd), key[1], key[2])
            if k in fmap:
                f = fmap[k]; break
        if f is None:
            feats.append((np.nan,)*4)
            continue
        sign = 1 if r.side == "home" else -1
        feats.append((sign * f["elo_diff"] / 100.0,
                      sign * (f["stats_diff"] if f["stats_diff"] == f["stats_diff"] else 0.0),
                      1.0 if r.side == "home" else 0.0,
                      float(f["wet"])))
    T[["elo_edge", "stats_edge", "is_home", "wet"]] = pd.DataFrame(feats, index=T.index)
    T = T.dropna(subset=["elo_edge"])
    return T, stats_rt, elo


def fit_try_model(T, upto=None):
    from sklearn.linear_model import PoissonRegressor
    est = T if upto is None else T[T.date < upto]
    Xc = est[["elo_edge", "stats_edge", "is_home", "wet"]].values
    age_days = (est.date.max() - est.date).dt.days.values
    wts = 0.5 ** (age_days / 365.0)          # one-season half-life
    m = PoissonRegressor(alpha=0.5, max_iter=1000).fit(Xc, est.tries, sample_weight=wts)
    conv_rate = np.average(est.conv / est.convAtt.clip(lower=1), weights=wts)
    pen_rate = np.average(est.pen, weights=wts)
    fg_rate = np.average(est.fg, weights=wts)
    return m, conv_rate, pen_rate, fg_rate


def build_try_shares():
    """player nrl_id -> shrunken tries-per-appearance, plus name/position."""
    recs = [json.loads(l) for l in open(os.path.join(HERE, "player_stats.jsonl"))]
    POS_PRIOR = {"Winger": .14, "Fullback": .11, "Centre": .11, "Five-Eighth": .06,
                 "Halfback": .05, "Hooker": .04, "Lock": .05, "2nd Row": .06,
                 "Second Row": .06, "Prop": .03, "Interchange": .04}
    agg = {}
    for r in recs:
        for p in r.get("players", []):
            d = agg.setdefault(p["playerId"], dict(tries=0, games=0,
                               name=p.get("name"), pos=p.get("position")))
            d["tries"] += p.get("tries", 0) or 0
            d["games"] += 1
    K = 10
    out = {}
    for pid, d in agg.items():
        prior = POS_PRIOR.get(d["pos"], .06)
        out[pid] = dict(rate=(d["tries"] + K * prior) / (d["games"] + K),
                        name=d["name"], pos=d["pos"], games=d["games"])
    return out


# ------------------------------------------------------------------ simulate
def simulate_match(elo_diff, stats_diff, wet, home_ids, away_ids, shares,
                   model_pack, n=N_SIMS, rng=RNG):
    m, conv_rate, pen_rate, fg_rate = model_pack
    xh = [[elo_diff/100.0, stats_diff, 1.0, wet]]
    xa = [[-elo_diff/100.0, -stats_diff, 0.0, wet]]
    lam_h, lam_a = m.predict(xh)[0], m.predict(xa)[0]
    th = rng.poisson(lam_h, n); ta = rng.poisson(lam_a, n)
    ch = rng.binomial(th, conv_rate); ca = rng.binomial(ta, conv_rate)
    ph = rng.poisson(pen_rate, n); pa = rng.poisson(pen_rate, n)
    fh = rng.poisson(fg_rate, n); fa = rng.poisson(fg_rate, n)
    hs = 4*th + 2*ch + 2*ph + fh
    as_ = 4*ta + 2*ca + 2*pa + fa
    # golden point: break draws 50/50 with a field goal
    draw = hs == as_
    gp = rng.random(n) < 0.5
    hs = hs + (draw & gp)
    as_ = as_ + (draw & ~gp)

    def scorers(t_counts, ids):
        if not ids:
            return np.zeros((n, 0), dtype=bool)
        w = np.array([shares.get(i, {"rate": .05})["rate"] for i in ids], dtype=float)
        w = w / w.sum()
        maxt = int(t_counts.max()) if len(t_counts) else 0
        scored = np.zeros((n, len(ids)), dtype=bool)
        for k in range(1, maxt + 1):
            active = t_counts >= k
            pick = rng.choice(len(ids), size=int(active.sum()), p=w)
            scored[np.where(active)[0], pick] = True
        return scored
    return dict(hs=hs, as_=as_, th=th, ta=ta,
                h_scored=scorers(th, home_ids), a_scored=scorers(ta, away_ids),
                lam=(lam_h, lam_a))


# ------------------------------------------------------------------ validate
def validate():
    from sklearn.metrics import log_loss, brier_score_loss
    T, stats_rt, elo = build_team_match_table()
    print(f"Team-match table: {len(T)} rows, {T.date.dt.year.min()}-{T.date.dt.year.max()}")
    cut = pd.Timestamp("2025-01-01")
    pack = fit_try_model(T, upto=cut)
    m, conv, pen, fg = pack
    print(f"Try model coefs [elo/100, stats, home, wet]: {np.round(m.coef_, 4)}")
    print(f"Conversion rate {conv:.3f} | penalty goals/team {pen:.2f} | FG/team {fg:.2f}")

    test = T[(T.date >= cut) & (T.side == "home")]
    shares = build_try_shares()
    ys, p_win, p_o415, y_o415, tot_pred, tot_act = [], [], [], [], [], []
    for r in test.itertuples():
        opp = T[(T.date == r.date) & (T.side == "away") & (T.team == r.opp)]
        if not len(opp):
            continue
        o = opp.iloc[0]
        sim = simulate_match(r.elo_edge*100, r.stats_edge, r.wet, [], [], shares, pack, n=4000)
        total_actual = 4*(r.tries + o.tries) + 2*(r.conv + o.conv) + 2*(r.pen + o.pen) + r.fg + o.fg
        ys.append(1 if total_actual >= 0 and (4*r.tries+2*r.conv+2*r.pen+r.fg) >
                  (4*o.tries+2*o.conv+2*o.pen+o.fg) else 0)
        p_win.append((sim["hs"] > sim["as_"]).mean())
        p_o415.append((sim["hs"] + sim["as_"] > 41.5).mean())
        y_o415.append(1 if total_actual > 41.5 else 0)
        tot_pred.append((sim["hs"] + sim["as_"]).mean())
        tot_act.append(total_actual)
    ys, p_win = np.array(ys), np.array(p_win)
    print(f"\nWalk-forward test on {len(ys)} matches (2025-26):")
    print(f"  Win prob:   logloss={log_loss(ys, np.clip(p_win,.01,.99)):.4f}  "
          f"acc={( (p_win>.5)==ys ).mean():.3f}")
    print(f"  Totals:     brier(o41.5)={brier_score_loss(y_o415, p_o415):.4f}  "
          f"base-rate brier={brier_score_loss(y_o415, [np.mean(y_o415)]*len(y_o415)):.4f}")
    print(f"  Mean predicted total {np.mean(tot_pred):.1f} vs actual {np.mean(tot_act):.1f}")
    # try-scorer calibration
    print("\nAnytime-try-scorer calibration (2025-26, players with 10+ games):")
    recs = [json.loads(l) for l in open(os.path.join(HERE, 'player_stats.jsonl'))]
    # predicted anytime rate bucket vs actual scoring frequency
    rows = []
    for r in recs:
        d = pd.to_datetime(r['kickoff']).tz_localize(None)
        if d < cut or not r.get('players'):
            continue
        for side in ('home', 'away'):
            team_p = [p for p in r['players'] if p['side'] == side]
            rates = np.array([shares.get(p['playerId'], {'rate': .05})['rate'] for p in team_p])
            if rates.sum() <= 0:
                continue
            shr = rates / rates.sum()
            for p, s in zip(team_p, shr):
                sh = shares.get(p['playerId'])
                if sh and sh['games'] >= 10:
                    pr = 1 - (1 - s) ** 3.7          # share of each of ~3.7 team tries
                    rows.append((pr, 1 if (p.get('tries', 0) or 0) > 0 else 0))
    C = pd.DataFrame(rows, columns=['pred', 'actual'])
    C['bucket'] = pd.cut(C.pred, [0, .15, .25, .35, .5, 1.0])
    print(C.groupby('bucket', observed=True).agg(n=('actual','size'),
          predicted=('pred','mean'), actual=('actual','mean')).round(3).to_string())


def price_round(round_no):
    import predict_teamlists as pt
    from weather_features import forecast_rain, nrl_city_to_key, WET_MM
    T, stats_rt, elo = build_team_match_table()
    pack = fit_try_model(T)
    shares = build_try_shares()
    results = pd.read_csv(os.path.join(HERE, "nrl_results.csv"), parse_dates=["date"])
    season = int(results.season.max())
    fixtures = pt.fetch_round_teamlists(season, round_no)
    for fx in fixtures:
        if fx["state"] not in ("Upcoming", "Pre"):
            continue
        h, a = fx["home"], fx["away"]
        hids = [t[3] for t in fx["teams"]["home"] if t[3]]
        aids = [t[3] for t in fx["teams"]["away"] if t[3]]
        s_h = sum(max(stats_rt.get(i, 0), 0) for i in hids)
        s_a = sum(max(stats_rt.get(i, 0), 0) for i in aids)
        kd = pd.to_datetime(fx["kickoff"]).tz_localize(None) if fx.get("kickoff") else results.date.max()
        mm = forecast_rain(nrl_city_to_key(fx.get("venue_city"), h), kd)
        wet = 1.0 if (mm is not None and mm >= WET_MM) else 0.0
        sim = simulate_match(elo.get(h, 1500) - elo.get(a, 1500), s_h - s_a, wet,
                             hids, aids, shares, pack)
        tot = sim["hs"] + sim["as_"]
        marg = sim["hs"] - sim["as_"]
        print(f"\n{h} v {a}" + ("  [WET forecast]" if wet else ""))
        print(f"  win: home {(marg>0).mean():.1%} | line -6.5 home {(marg>6.5).mean():.1%} | "
              f"+6.5 away {(marg>-6.5).mean():.1%}")
        print(f"  totals: mean {tot.mean():.1f} | over 41.5 {(tot>41.5).mean():.1%} | "
              f"over 47.5 {(tot>47.5).mean():.1%}")
        top = sorted(range(len(hids)), key=lambda i: -sim["h_scored"][:, i].mean())[:3]
        names = {t[3]: f"{t[0]} {t[1]}" for t in fx["teams"]["home"] + fx["teams"]["away"]}
        print("  anytime try-scorer (home top 3): " + ", ".join(
            f"{names.get(hids[i],'?')} {sim['h_scored'][:,i].mean():.0%}" for i in top))
        # correlation demo: home win + over vs naive independent product
        joint = ((marg > 0) & (tot > 41.5)).mean()
        indep = (marg > 0).mean() * (tot > 41.5).mean()
        print(f"  SGM correlation check [home win x over 41.5]: joint {joint:.1%} vs "
              f"independent {indep:.1%} ({(joint-indep)*100:+.1f} pts)")

LEDGER = os.path.join(HERE, "sgm_ledger.csv")

def write_ledger(round_no):
    """Record simulator prices (and book h2h) pre-kickoff for paper-testing."""
    import predict_teamlists as pt
    from weather_features import forecast_rain, nrl_city_to_key, WET_MM
    T, stats_rt, elo = build_team_match_table()
    pack = fit_try_model(T)
    shares = build_try_shares()
    results = pd.read_csv(os.path.join(HERE, "nrl_results.csv"), parse_dates=["date"])
    season = int(results.season.max())
    fixtures = pt.fetch_round_teamlists(season, round_no)
    rows = []
    stamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
    for fx in fixtures:
        if fx["state"] not in ("Upcoming", "Pre"):
            continue
        h, a = fx["home"], fx["away"]
        hids = [t[3] for t in fx["teams"]["home"] if t[3]]
        aids = [t[3] for t in fx["teams"]["away"] if t[3]]
        s_h = sum(max(stats_rt.get(i, 0), 0) for i in hids)
        s_a = sum(max(stats_rt.get(i, 0), 0) for i in aids)
        kd = pd.to_datetime(fx["kickoff"]).tz_localize(None) if fx.get("kickoff") else results.date.max()
        mm = forecast_rain(nrl_city_to_key(fx.get("venue_city"), h), kd)
        wet = 1.0 if (mm is not None and mm >= WET_MM) else 0.0
        sim = simulate_match(elo.get(h, 1500) - elo.get(a, 1500), s_h - s_a, wet,
                             hids, aids, shares, pack)
        marg = sim["hs"] - sim["as_"]; tot = sim["hs"] + sim["as_"]
        names = {t[3]: f"{t[0]} {t[1]}" for t in fx["teams"]["home"] + fx["teams"]["away"]}
        try:
            ho, ao = float(fx.get("home_odds") or 0), float(fx.get("away_odds") or 0)
        except (TypeError, ValueError):
            ho = ao = 0
        def add(market, sel, p, book=""):
            rows.append(dict(run_time=stamp, kickoff=str(kd.date()), home=h, away=a,
                             market=market, selection=sel, sim_prob=round(p, 4),
                             sim_fair_odds=round(1/max(p, 1e-4), 2),
                             book_odds=book, actual=""))
        add("h2h", h, (marg > 0).mean(), ho if ho > 1 else "")
        add("h2h", a, (marg < 0).mean(), ao if ao > 1 else "")
        add("line", f"{h} -6.5", (marg > 6.5).mean())
        add("line", f"{a} +6.5", (marg > -6.5).mean() and (1 - (marg > 6.5).mean()) or 0)
        rows[-1]["sim_prob"] = round(float((marg < 6.5).mean()), 4)
        rows[-1]["sim_fair_odds"] = round(1/max((marg < 6.5).mean(), 1e-4), 2)
        add("total", "over 41.5", (tot > 41.5).mean())
        add("total", "over 47.5", (tot > 47.5).mean())
        for side_key, ids, sc in (("home", hids, sim["h_scored"]), ("away", aids, sim["a_scored"])):
            top = sorted(range(len(ids)), key=lambda i: -sc[:, i].mean())[:3]
            for i in top:
                add("anytime_try", names.get(ids[i], "?"), float(sc[:, i].mean()))
        # one correlated combo per game
        joint = float(((marg > 0) & (tot > 41.5)).mean())
        add("sgm_combo", f"{h} win + over 41.5", joint)
    new = pd.DataFrame(rows)
    if os.path.exists(LEDGER):
        old = pd.read_csv(LEDGER)
        key = ["kickoff", "home", "away", "market", "selection"]
        new = new[~new.set_index(key).index.isin(old.set_index(key).index)]
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(LEDGER, index=False)
    print(f"Ledger: {len(new)} total rows -> sgm_ledger.csv "
          f"(fill book_odds column from your betting app before kickoff)")

def grade_ledger():
    if not os.path.exists(LEDGER):
        return print("no ledger yet")
    L = pd.read_csv(LEDGER)
    results = pd.read_csv(os.path.join(HERE, "nrl_results.csv"), parse_dates=["date"])
    recs = [json.loads(l) for l in open(os.path.join(HERE, "player_stats.jsonl"))]
    scorers = {}
    for r in recs:
        d = str(pd.to_datetime(r["kickoff"]).tz_localize(None).date())
        for p in r.get("players", []):
            if (p.get("tries", 0) or 0) > 0:
                scorers.setdefault(d, set()).add(p.get("name"))
    graded = 0
    for i in L[L.actual.isna() | (L.actual == "")].index:
        row = L.loc[i]
        m = results[(results.home_team == row.home) & (results.away_team == row.away) &
                    ((results.date - pd.Timestamp(row.kickoff)).abs() <= pd.Timedelta(days=2))]
        if not len(m):
            continue
        g = m.iloc[-1]; marg = g.home_score - g.away_score; tot = g.home_score + g.away_score
        mk, sel = row.market, row.selection
        val = None
        if mk == "h2h":
            val = 1 if (marg > 0) == (sel == row.home) else 0
        elif mk == "line":
            val = 1 if (marg > 6.5 if "-6.5" in sel else marg < 6.5) else 0
        elif mk == "total":
            val = 1 if tot > float(sel.split()[-1]) else 0
        elif mk == "sgm_combo":
            val = 1 if (marg > 0 and tot > 41.5) else 0
        elif mk == "anytime_try":
            day = scorers.get(str(g.date.date()))
            if day is not None:
                val = 1 if sel in day else 0
        if val is not None:
            L.loc[i, "actual"] = val; graded += 1
    L.to_csv(LEDGER, index=False)
    done = L[L.actual.isin([0, 1, "0", "1", 0.0, 1.0])]
    if len(done):
        p = done.sim_prob.astype(float); yy = done.actual.astype(float)
        print(f"Graded {graded} new. Ledger record: {len(done)} legs, "
              f"hit rate {yy.mean():.1%}, avg sim prob {p.mean():.1%}, "
              f"brier {((p-yy)**2).mean():.4f}")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "ledger":
        grade_ledger()
        if len(sys.argv) > 2:
            write_ledger(int(sys.argv[2]))
        else:
            import urllib.request
            import predict_teamlists as pt
            results = pd.read_csv(os.path.join(HERE, "nrl_results.csv"), parse_dates=["date"])
            season = int(results.season.max())
            d = json.loads(urllib.request.urlopen(urllib.request.Request(
                f"https://www.nrl.com/draw/data?competition=111&season={season}",
                headers=pt.UA)).read())
            write_ledger(d["selectedRoundId"])
    elif len(sys.argv) > 1 and sys.argv[1] == "grade":
        grade_ledger()
    elif len(sys.argv) > 1 and sys.argv[1] == "validate":
        validate()
    elif len(sys.argv) > 2 and sys.argv[1] == "price":
        price_round(int(sys.argv[2]))
    else:
        print("Usage: python sgm_simulator.py validate | price <round>")
