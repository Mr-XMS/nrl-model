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

tab_week, tab_record, tab_sgm, tab_market = st.tabs(
    ["📋 This Week", "📈 Track Record", "🎯 SGM Ledger", "💹 Market"])

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
