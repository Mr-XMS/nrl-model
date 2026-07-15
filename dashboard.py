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

tab_week, tab_record, tab_sgm, tab_market, tab_scen, tab_players, tab_verdict = st.tabs(
    ["📋 This Week", "📈 Track Record", "🎯 SGM Ledger", "💹 Market",
     "🩹 Scenarios", "👤 Players", "⚖️ Verdict"])

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
            rows = []
            for r in upcoming.itertuples():
                p = r.p_base
                pick, conf = (r.home_team, p) if p > 0.5 else (r.away_team, 1 - p)
                mkt = getattr(r, "p_market", np.nan)
                edge = (p - mkt) if mkt == mkt else np.nan
                rows.append(dict(
                    Date=r.date.date(), Home=r.home_team, Away=r.away_team,
                    Pick=pick, Confidence=f"{conf:.0%}",
                    Market=f"{mkt:.0%} home" if mkt == mkt else "—",
                    Edge=f"{edge*100:+.0f} pts" if edge == edge else "—",
                    Flag="🚩" if (edge == edge and abs(edge) > 0.06) else ""))
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
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
