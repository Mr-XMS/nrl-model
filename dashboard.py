"""
NRL Model Dashboard
===================
A local web dashboard over the model's ledger files.

Setup (one-time):   pip3 install streamlit plotly
Run:                streamlit run dashboard.py
Then it opens in your browser at http://localhost:8501
"""
import os
import numpy as np
import pandas as pd
import streamlit as st

HERE = os.path.dirname(os.path.abspath(__file__))
def path(f): return os.path.join(HERE, f)

st.set_page_config(page_title="NRL Model", page_icon="🏉", layout="wide")

TEAM_STYLE = {
 "brisbane-broncos": ("Broncos", "#6C1D45"), "canberra-raiders": ("Raiders", "#95C11F"),
 "canterbury-bankstown-bulldogs": ("Bulldogs", "#00468B"),
 "cronulla-sutherland-sharks": ("Sharks", "#00A9E0"), "dolphins": ("Dolphins", "#BE1E2D"),
 "gold-coast-titans": ("Titans", "#0FAAA2"), "manly-warringah-sea-eagles": ("Sea Eagles", "#78002E"),
 "melbourne-storm": ("Storm", "#632390"), "newcastle-knights": ("Knights", "#EE3524"),
 "north-queensland-cowboys": ("Cowboys", "#002B5C"), "parramatta-eels": ("Eels", "#006EB5"),
 "penrith-panthers": ("Panthers", "#E6007E"), "south-sydney-rabbitohs": ("Rabbitohs", "#00453A"),
 "st-george-illawarra-dragons": ("Dragons", "#E2231B"), "sydney-roosters": ("Roosters", "#00305E"),
 "warriors": ("Warriors", "#151F6D"), "wests-tigers": ("Tigers", "#F68B1F"),
}
BADGE_KEY = {
 "brisbane-broncos": "broncos", "canberra-raiders": "raiders",
 "canterbury-bankstown-bulldogs": "bulldogs", "cronulla-sutherland-sharks": "sharks",
 "dolphins": "dolphins", "gold-coast-titans": "titans",
 "manly-warringah-sea-eagles": "sea-eagles", "melbourne-storm": "storm",
 "newcastle-knights": "knights", "north-queensland-cowboys": "cowboys",
 "parramatta-eels": "eels", "penrith-panthers": "panthers",
 "south-sydney-rabbitohs": "rabbitohs", "st-george-illawarra-dragons": "dragons",
 "sydney-roosters": "roosters", "warriors": "warriors", "wests-tigers": "wests-tigers",
}
def chip(slug, bold=False):
    name, col = TEAM_STYLE.get(slug, (slug, "#666"))
    w = "700" if bold else "500"
    key = BADGE_KEY.get(slug)
    if key:
        # official badge hotlinked from nrl.com; colour dot appears if it fails to load
        icon = (f'<img src="https://www.nrl.com/.theme/{key}/badge.svg" '
                f'style="width:22px;height:22px;vertical-align:middle" '
                f'onerror="this.style.display=\'none\';this.nextElementSibling.style.display=\'inline-block\'">'
                f'<span style="width:12px;height:12px;border-radius:3px;background:{col};'
                f'display:none"></span>')
    else:
        icon = (f'<span style="width:12px;height:12px;border-radius:3px;background:{col};'
                f'display:inline-block"></span>')
    return (f'<span style="display:inline-flex;align-items:center;gap:6px;font-weight:{w}">'
            f'{icon}{name}</span>')
def conf_bar(p):
    pct = int(round(p * 100))
    return (f'<div style="display:flex;align-items:center;gap:8px">'
            f'<div style="background:#eee;border-radius:6px;width:90px;height:10px">'
            f'<div style="background:#2b8a3e;width:{pct}%;height:10px;border-radius:6px"></div></div>'
            f'<b>{pct}%</b></div>')

st.title("🏉 NRL Prediction Model")

def load(f, dates=None):
    p = path(f)
    if not os.path.exists(p):
        return None
    try:
        return pd.read_csv(p, parse_dates=dates or [])
    except Exception as e:
        st.warning(f"Couldn't read {f}: {e}")
        return None

comp = load("model_comparison.csv", ["date"])
preds = load("predictions.csv", ["date"])
ledger = load("sgm_ledger.csv")
hist = load("odds_history.csv")
fhist = load("forecast_history.csv")
scen = load("scenarios.csv")
players = load("player_ratings.csv")

(tab_week, tab_record, tab_sgm, tab_market, tab_scen, tab_players,
 tab_verdict, tab_bet) = st.tabs(
    ["📋 This Week", "📈 Track Record", "🎯 SGM Ledger", "💹 Market",
     "🩹 Scenarios", "👤 Players", "⚖️ Verdict", "💰 Bet Simulator"])

# ---------------------------------------------------------------- This Week
with tab_week:
    if comp is None or not len(comp):
        st.info("No predictions logged yet.")
    else:
        upcoming = comp[comp.actual_home_win.isna()].copy()
        if not len(upcoming):
            st.info("No upcoming games logged — next Tuesday run will add the new round.")
        else:
            st.subheader("Current round picks")
            # kickoff/venue from the latest forecast snapshot, if recorded
            ko = {}
            if fhist is not None and "kickoff" in fhist.columns:
                fh = fhist.dropna(subset=["kickoff"]).copy()
                for r in fh.itertuples():
                    try:
                        t = pd.to_datetime(r.kickoff).tz_convert("Australia/Sydney")
                    except Exception:
                        try:
                            t = pd.to_datetime(r.kickoff).tz_localize("UTC").tz_convert("Australia/Sydney")
                        except Exception:
                            continue
                    ko[(r.home_team, r.away_team)] = (t, getattr(r, "venue_city", ""))
            html = ['<table style="width:100%;border-collapse:collapse;font-size:15px">',
                    '<tr style="text-align:left;color:#888;border-bottom:2px solid #ddd">'
                    '<th style="padding:8px 6px">Kickoff</th><th>Match</th><th>Pick</th>'
                    '<th>Confidence</th><th>Market</th><th>Edge</th></tr>']
            for r in upcoming.sort_values("date").itertuples():
                p = r.p_base
                pick_team = r.home_team if p > 0.5 else r.away_team
                conf = p if p > 0.5 else 1 - p
                mkt = getattr(r, "p_market", np.nan)
                edge = (p - mkt) if mkt == mkt else np.nan
                k = ko.get((r.home_team, r.away_team))
                when = (k[0].strftime("%a %-I:%M%p").replace("AM","am").replace("PM","pm")
                        + f'<br><span style="color:#999;font-size:12px">{(k[1] or "")}</span>')                     if k else r.date.strftime("%a %d %b")
                edge_html = "—"
                if edge == edge:
                    col = "#c92a2a" if abs(edge) > 0.06 else "#888"
                    flag = " 🚩" if abs(edge) > 0.06 else ""
                    edge_html = f'<span style="color:{col};font-weight:600">{edge*100:+.0f} pts{flag}</span>'
                html.append(
                    f'<tr style="border-bottom:1px solid #eee">'
                    f'<td style="padding:10px 6px;white-space:nowrap">{when}</td>'
                    f'<td>{chip(r.home_team)} <span style="color:#bbb">v</span> {chip(r.away_team)}</td>'
                    f'<td>{chip(pick_team, bold=True)}</td>'
                    f'<td>{conf_bar(conf)}</td>'
                    f'<td>{f"{mkt:.0%} home" if mkt == mkt else "—"}</td>'
                    f'<td>{edge_html}</td></tr>')
            html.append("</table>")
            st.markdown("".join(html), unsafe_allow_html=True)
            st.caption("🚩 = model and market disagree by more than 6 points. "
                       "B (fitness) and C (weather) variants are graded in Track Record.")
        if fhist is not None and len(fhist):
            st.subheader("What changed since the previous run")
            f = fhist.copy()
            f["run_time"] = pd.to_datetime(f.run_time)
            runs = sorted(f.run_time.unique())
            if len(runs) >= 2:
                cur = f[f.run_time == runs[-1]].set_index(["home_team", "away_team"])
                prev = f[f.run_time == runs[-2]].set_index(["home_team", "away_team"])
                moves = []
                for k in cur.index:
                    if k in prev.index:
                        d = (cur.loc[k, "p_base"] - prev.loc[k, "p_base"]) * 100
                        dm = (cur.loc[k, "p_market"] - prev.loc[k, "p_market"]) * 100                             if pd.notna(cur.loc[k, "p_market"]) and pd.notna(prev.loc[k, "p_market"]) else None
                        if abs(d) >= 1 or (dm is not None and abs(dm) >= 1):
                            moves.append(dict(Game=f"{k[0]} v {k[1]}",
                                Model=f"{d:+.0f} pts", Market=f"{dm:+.0f} pts" if dm is not None else "—",
                                Wet="☔" if cur.loc[k, "wet"] == 1 and prev.loc[k, "wet"] == 0 else ""))
                if moves:
                    st.dataframe(pd.DataFrame(moves), use_container_width=True, hide_index=True)
                else:
                    st.caption("No meaningful movement since the previous run.")
            else:
                st.caption("Movement appears here once two runs are recorded.")

# ------------------------------------------------------------- Track Record
with tab_record:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Simple model (weekly picks)")
        if preds is not None and len(preds.dropna(subset=["actual_home_win"])):
            done = preds.dropna(subset=["actual_home_win"]).copy()
            done["correct"] = ((done.p_home_win > .5) ==
                               (done.actual_home_win == 1)).astype(int)
            acc = done.correct.mean()
            brier = ((done.p_home_win - done.actual_home_win) ** 2).mean()
            m1, m2, m3 = st.columns(3)
            m1.metric("Graded picks", len(done))
            m2.metric("Accuracy", f"{acc:.0%}")
            m3.metric("Brier score", f"{brier:.3f}")
            done = done.sort_values("date")
            done["cumulative accuracy"] = done.correct.expanding().mean()
            st.line_chart(done.set_index("date")["cumulative accuracy"])
            st.caption("Long-run backtest expectation: ~65%")
        else:
            st.info("No graded picks yet.")
    with c2:
        st.subheader("Model trial (A / B / C / market / blend)")
        if comp is not None:
            done = comp.dropna(subset=["actual_home_win"]).copy()
            if len(done):
                y = done.actual_home_win.astype(float)
                rows = []
                for name, col in [("A: baseline", "p_base"), ("B: fitness ramp", "p_retadj"),
                                  ("C: weather", "p_wet"), ("Market", "p_market"),
                                  ("Blend", "p_blend")]:
                    if col not in done.columns:
                        continue
                    p = pd.to_numeric(done[col], errors="coerce").dropna()
                    if not len(p):
                        continue
                    yy = y.loc[p.index]
                    pc = p.clip(1e-6, 1 - 1e-6)
                    rows.append(dict(Model=name, Graded=len(p),
                        Accuracy=f"{((pc > .5).astype(int) == yy).mean():.0%}",
                        LogLoss=round(float(-(yy*np.log(pc) + (1-yy)*np.log(1-pc)).mean()), 4),
                        Brier=round(float(((pc - yy) ** 2).mean()), 4)))
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                st.caption("Lower log loss = better. The season-long question: "
                           "does B beat A, does C beat A, does Blend beat Market?")
            else:
                st.info("No graded trial games yet.")

# ------------------------------------------------------------------- SGM
with tab_sgm:
    if ledger is None or not len(ledger):
        st.info("No SGM ledger yet.")
    else:
        L = ledger.copy()
        L["book_odds"] = pd.to_numeric(L.book_odds, errors="coerce")
        L["actual"] = pd.to_numeric(L.actual, errors="coerce")
        graded = L.dropna(subset=["actual"])
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Legs recorded", len(L))
        m2.metric("With book odds", int(L.book_odds.notna().sum()))
        m3.metric("Graded", len(graded))
        if len(graded):
            m4.metric("Hit rate vs predicted",
                      f"{graded.actual.mean():.0%} vs {graded.sim_prob.mean():.0%}")
        val = L[(L.book_odds.notna()) & (L.book_odds > L.sim_fair_odds)]
        st.subheader(f"Paper-value legs (book pays above our fair odds): {len(val)}")
        if len(val):
            vg = val.dropna(subset=["actual"])
            if len(vg):
                roi = (vg.actual * vg.book_odds - 1).mean()
                st.metric("Value-leg paper ROI", f"{roi:+.1%}",
                          help="Average return per $1 staked on flagged legs. "
                               "The season's key number.")
            show = val[["kickoff", "home", "away", "market", "selection",
                        "sim_fair_odds", "book_odds", "actual"]].copy()
            st.dataframe(show, use_container_width=True, hide_index=True)
        # calibration
        if len(graded) >= 30:
            st.subheader("Simulator calibration")
            g = graded.copy()
            g["bucket"] = pd.cut(g.sim_prob, [0, .2, .35, .5, .7, 1.0])
            cal = g.groupby("bucket", observed=True).agg(
                n=("actual", "size"), predicted=("sim_prob", "mean"),
                actual=("actual", "mean")).round(3)
            st.dataframe(cal, use_container_width=True)

# ------------------------------------------------------------------ Market
with tab_market:
    if hist is None or not len(hist):
        st.info("No odds history yet — the scanner logs prices each run.")
    else:
        H = hist.copy()
        H["scan_time"] = pd.to_datetime(H.scan_time)
        games = H[["home", "away", "kickoff"]].drop_duplicates().sort_values("kickoff", ascending=False)
        options = [f"{r.home} v {r.away} ({r.kickoff})" for r in games.itertuples()]
        sel = st.selectbox("Game", options)
        if sel:
            idx = options.index(sel)
            g = games.iloc[idx]
            gh = H[(H.home == g.home) & (H.away == g.away) & (H.kickoff == g.kickoff)]
            gh = gh[gh.team == g.home].copy()
            gh["implied home %"] = 100 / gh.odds
            pivot = gh.pivot_table(index="scan_time", columns="book",
                                   values="implied home %", aggfunc="mean")
            st.line_chart(pivot)
            st.caption(f"Implied probability for {g.home} (home) across books over time. "
                       "Watch whether the market drifts toward the model's number.")


# ---------------------------------------------------------------- Scenarios
with tab_scen:
    st.subheader("Injury scenarios (flagged players, latest Tuesday run)")
    if scen is None or not len(scen):
        st.info("No scenarios yet - generated automatically each Tuesday.")
    else:
        games = scen[["home", "away"]].drop_duplicates()
        pick = st.selectbox("Game", [f"{r.home} v {r.away}" for r in games.itertuples()])
        if pick:
            h, a = pick.split(" v ")
            sub = scen[(scen.home == h) & (scen.away == a)]
            for r in sub.itertuples():
                st.markdown(f"**{r.player}** ({r.team}, {r.status})")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Plays fully fit", f"{r.p_fit:.0%}")
                c2.metric("As named", f"{r.p_named:.0%}")
                c3.metric("Withdrawn", f"{r.p_withdrawn:.0%}")
                c4.metric("Swing", f"{r.swing_pts} pts")
        st.caption("Home-team win probability under each scenario. Swings for "
                   "playmakers are understated (known stats-layer bias).")

# ------------------------------------------------------------------ Players
with tab_players:
    st.subheader("Player rating browser")
    if players is None or not len(players):
        st.info("No ratings file yet - generated each Tuesday.")
    else:
        q = st.text_input("Search player or team")
        P = players.copy()
        if q:
            m = P.player.str.contains(q, case=False, na=False) |                 P.team.str.contains(q, case=False, na=False)
            P = P[m]
        st.dataframe(P, use_container_width=True, hide_index=True, height=480)
        st.caption("presence_rating: win contribution from team-sheet history "
                   "(0 = league average). stats_rating: from game statistics - "
                   "currently undervalues playmakers/hookers (known bias, "
                   "offseason fix). tries_per_game feeds the SGM simulator.")

# ------------------------------------------------------------------ Verdict
with tab_verdict:
    st.subheader("Season verdict (assembling as games grade)")
    rng = np.random.default_rng(1)
    if comp is not None:
        done = comp.dropna(subset=["actual_home_win"]).copy()
        if len(done) >= 5:
            y = done.actual_home_win.astype(float).values
            def ll(p):
                p = np.clip(pd.to_numeric(p, errors="coerce").values, 1e-6, 1-1e-6)
                return -(y*np.log(p) + (1-y)*np.log(1-p))
            comparisons = [("B (fitness) vs A", "p_retadj", "p_base"),
                           ("C (weather) vs A", "p_wet", "p_base"),
                           ("Blend vs Market", "p_blend", "p_market")]
            rows = []
            for name, c1, c2 in comparisons:
                if c1 not in done.columns or c2 not in done.columns:
                    continue
                ok = done[c1].notna() & done[c2].notna()
                if ok.sum() < 5:
                    continue
                d = ll(done.loc[ok, c2]) - ll(done.loc[ok, c1])   # + means c1 better
                boots = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
                lo, hi = np.percentile(boots, [5, 95])
                verdict = "✅ leading" if lo > 0 else ("❌ trailing" if hi < 0 else "🔶 undecided")
                rows.append(dict(Comparison=name, Games=int(ok.sum()),
                                 Gap=f"{d.mean():+.4f}", CI90=f"[{lo:+.4f}, {hi:+.4f}]",
                                 Verdict=verdict))
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                st.caption("Gap = mean log-loss advantage per game (+ = first model better). "
                           "A verdict needs the 90% interval to exclude zero.")
        else:
            st.info("Trial verdicts appear after ~5 graded games; "
                    "they become meaningful after ~50.")
    if ledger is not None and len(ledger):
        L = ledger.copy()
        L["book_odds"] = pd.to_numeric(L.book_odds, errors="coerce")
        L["actual"] = pd.to_numeric(L.actual, errors="coerce")
        val = L[(L.book_odds > L.sim_fair_odds)].dropna(subset=["actual"])
        st.subheader("Value-leg paper ROI")
        if len(val) >= 10:
            r = (val.actual * val.book_odds - 1).values
            boots = [r[rng.integers(0, len(r), len(r))].mean() for _ in range(2000)]
            lo, hi = np.percentile(boots, [5, 95])
            c1, c2 = st.columns(2)
            c1.metric("ROI per $1", f"{r.mean():+.1%}")
            c2.metric("90% interval", f"[{lo:+.0%}, {hi:+.0%}]")
            st.caption("The strategy question: is the lower bound above 0%?")
        else:
            st.info(f"{len(val)} graded value legs so far - ROI shown from 10+.")
        graded = L.dropna(subset=["actual"])
        if len(graded) >= 30:
            st.subheader("Simulator calibration curve")
            g = graded.copy()
            g["bucket"] = pd.cut(g.sim_prob, np.arange(0, 1.05, .1))
            cal = g.groupby("bucket", observed=True).agg(
                predicted=("sim_prob", "mean"), actual=("actual", "mean")).dropna()
            chart = pd.DataFrame({"actual": cal.actual.values,
                                  "perfect": cal.predicted.values},
                                 index=cal.predicted.round(2))
            st.line_chart(chart)
            st.caption("Actual should hug the 'perfect' diagonal.")


# ------------------------------------------------------------ Bet Simulator
with tab_bet:
    st.subheader("Round staking simulator (paper)")
    st.caption("Allocates a hypothetical bankroll across this round's genuine edges "
               "using quarter-Kelly staking, then simulates 10,000 rounds so you see "
               "the full distribution — not just the average. Paper tool: the model's "
               "edges are unproven until the season's ledgers say otherwise.")
    if comp is None or hist is None or not len(hist):
        st.info("Needs logged predictions and at least one odds scan.")
    else:
        bankroll = st.number_input("Hypothetical round bankroll ($)", 10, 10000, 100, step=10)
        min_edge = st.slider("Minimum EV to bet (%)", 2, 15, 4) / 100.0
        up = comp[comp.actual_home_win.isna()].copy()
        H = hist.copy()
        H["scan_time"] = pd.to_datetime(H.scan_time)
        latest = H[H.scan_time == H.scan_time.max()]
        lg = lambda q: np.log(np.clip(q, 1e-6, 1-1e-6) / (1 - np.clip(q, 1e-6, 1-1e-6)))
        cands = []
        for r in up.itertuples():
            gh = latest[(latest.home == r.home_team) & (latest.away == r.away_team)]
            if not len(gh):
                continue
            bh = gh[gh.team == r.home_team].odds.max()
            ba = gh[gh.team == r.away_team].odds.max()
            bkh = gh[(gh.team == r.home_team) & (gh.odds == bh)].book.iloc[0] if bh == bh else ""
            bka = gh[(gh.team == r.away_team) & (gh.odds == ba)].book.iloc[0] if ba == ba else ""
            if not (bh > 1 and ba > 1):
                continue
            ih, ia = 1/bh, 1/ba
            mkt = ih / (ih + ia)
            pb = 1 / (1 + np.exp(-0.5 * (lg(float(r.p_base)) + lg(mkt))))   # live blend
            for side, prob, odds, book in ((r.home_team, pb, bh, bkh),
                                           (r.away_team, 1-pb, ba, bka)):
                ev = prob * odds - 1
                if ev > min_edge:
                    kelly = (prob * odds - 1) / (odds - 1)
                    cands.append(dict(game=f"{r.home_team.split('-')[-1]} v {r.away_team.split('-')[-1]}",
                                      side=side, prob=prob, odds=odds, book=book,
                                      ev=ev, kelly=kelly))
        if not cands:
            st.success(f"No bets clear the {min_edge:.0%} EV bar at current prices — "
                       "the disciplined recommendation this round is: don't bet. "
                       "That is a real and common output of a real system.")
        else:
            C = pd.DataFrame(cands)
            C["stake"] = C.kelly * 0.25 * bankroll          # quarter-Kelly
            if C.stake.sum() > bankroll:                    # cap at bankroll
                C["stake"] *= bankroll / C.stake.sum()
            C["stake"] = C.stake.round(0)
            C = C[C.stake >= 1]
            show = C.copy()
            show["model prob"] = (show.prob*100).round(0).astype(int).astype(str) + "%"
            show["EV"] = (show.ev*100).round(1).astype(str) + "%"
            show["stake $"] = show.stake.astype(int)
            st.dataframe(show[["game","side","model prob","odds","book","EV","stake $"]],
                         use_container_width=True, hide_index=True)
            total = C.stake.sum()
            st.write(f"Total staked: **${total:.0f}** of ${bankroll} "
                     f"(unstaked bankroll sits out — no edge, no bet)")
            # Monte Carlo the slate
            rng = np.random.default_rng(11)
            wins = rng.random((10000, len(C))) < C.prob.values
            pl = (wins * (C.stake.values * (C.odds.values - 1))
                  - (~wins) * C.stake.values).sum(axis=1)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Expected profit", f"${pl.mean():+.0f}")
            c2.metric("Chance round loses money", f"{(pl < 0).mean():.0%}")
            c3.metric("Median outcome", f"${np.median(pl):+.0f}")
            c4.metric("Worst 5% of rounds", f"${np.percentile(pl, 5):+.0f}")
            counts, edges_ = np.histogram(pl, bins=30)
            st.bar_chart(pd.DataFrame({"rounds": counts},
                         index=np.round(edges_[:-1], 0)))
            st.caption("Distribution of round profit/loss across 10,000 simulations, "
                       "using the model's own probabilities. If the model is "
                       "overconfident, reality is worse than this chart.")
