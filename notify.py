"""Push a weekly summary to your phone via ntfy.sh (optional).
Setup: install the ntfy app, subscribe to a topic with a hard-to-guess name
(e.g. nrl-model-x8k2p), and save that topic name in ntfy_topic.txt here.
Silently does nothing if no topic file exists."""
import os, sys, urllib.request
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
tf = os.path.join(HERE, "ntfy_topic.txt")
if not os.path.exists(tf):
    sys.exit(0)
topic = open(tf).read().strip()
mode = sys.argv[1] if len(sys.argv) > 1 else "grades"
msg = ""
try:
    if mode == "grades":
        p = pd.read_csv(os.path.join(HERE, "predictions.csv"))
        done = p.dropna(subset=["actual_home_win"])
        if len(done):
            recent = done.tail(10)
            ok = ((recent.p_home_win > .5) == (recent.actual_home_win == 1)).sum()
            tot = ((done.p_home_win > .5) == (done.actual_home_win == 1)).mean()
            msg = f"Graded: {ok}/{len(recent)} recent. Season {tot:.0%} ({len(done)} picks)."
        L = pd.read_csv(os.path.join(HERE, "sgm_ledger.csv"))
        L["actual"] = pd.to_numeric(L.actual, errors="coerce")
        L["book_odds"] = pd.to_numeric(L.book_odds, errors="coerce")
        val = L[(L.book_odds > L.sim_fair_odds)].dropna(subset=["actual"])
        if len(val):
            roi = (val.actual * val.book_odds - 1).mean()
            msg += f" Value-leg ROI: {roi:+.0%} ({len(val)} legs)."
    else:  # picks
        c = pd.read_csv(os.path.join(HERE, "model_comparison.csv"))
        up = c[c.actual_home_win.isna()]
        lines = []
        for r in up.itertuples():
            pick, conf = (r.home_team, r.p_base) if r.p_base > .5 else (r.away_team, 1-r.p_base)
            mkt = getattr(r, "p_market", float("nan"))
            e = f" EDGE{(r.p_base-mkt)*100:+.0f}" if mkt == mkt and abs(r.p_base-mkt) > .06 else ""
            lines.append(f"{pick.split('-')[-1]} {conf:.0%}{e}")
        msg = "This week: " + "; ".join(lines)
except Exception as e:
    msg = f"NRL model notify error: {e}"
if msg:
    req = urllib.request.Request(f"https://ntfy.sh/{topic}", data=msg.encode(),
                                 headers={"Title": "NRL Model"})
    urllib.request.urlopen(req, timeout=15)
    print("notification sent")
