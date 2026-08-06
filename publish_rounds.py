"""
publish_rounds.py - build the free public archive of graded forecasts.
======================================================================
Embargo rule (the only rule):

    A round's forecast becomes public at 00:00 Australia/Sydney on the
    Monday following its last fixture - i.e. midnight Sunday night.

The unlock is a pure clock event. It does not depend on results having
been scraped, on games being marked complete, or on anything else that
can fail upstream. A round page is generated the first time the script
runs after its unlock moment, and never before, so an embargoed round
simply does not exist on the server.

Results fill in later, whenever grading catches up. A page published
with blank scores is correct behaviour, not a bug.

Outputs (all under docs/, servable by GitHub Pages with no extra infra):
    docs/index.html          season archive + running record
    docs/round-NN.html       one page per published round
    docs/publish_log.csv     audit trail: when each round was opened

Safe to run on every automation invocation; it is idempotent and
self-gating. Run with --dry-run to see what would publish.
"""
import os
import re
import sys
import subprocess
from datetime import datetime, timedelta, time
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DOCS = os.path.join(HERE, "docs")
TZ = ZoneInfo("Australia/Sydney")
SEASON = 2026
SITE_NAME = "NRL Forecast"
TAGLINE = "An open research project"

# Set SITE_URL in the Cloudflare Pages build environment once the custom
# domain is live, e.g. https://nrlforecast.com (no trailing slash).
SITE_URL = os.environ.get("SITE_URL", "").rstrip("/")

BRIER_POP = ('<details class="tip"><summary>i</summary><div class="pop">'
             '<strong>Brier score</strong> grades the probabilities, not just '
             'the pick. Call a game at 90% and get it right, you score well; '
             'call it at 90% and get it wrong, you are punished hard. '
             '<strong>Lower is better.</strong> 0 is perfect, 0.25 is what '
             'you would score by saying &ldquo;50/50&rdquo; to every game, '
             'and anything above that is worse than guessing.'
             '</div></details>')

PRICE_WEEK = "$2 single round"
PRICE_SEASON = "$27 / season"


# --------------------------------------------------------------- helpers
def now_syd():
    return datetime.now(TZ)


def unlock_moment(last_fixture_date):
    """First Monday 00:00 Sydney strictly after the round's last fixture.

    Always lands on a Sunday midnight. If a round somehow carries a
    Monday fixture, the walk pushes the unlock to the *following*
    Monday, so a live forecast can never be published early.
    """
    d = last_fixture_date + timedelta(days=1)
    while d.weekday() != 0:            # 0 = Monday
        d += timedelta(days=1)
    return datetime.combine(d, time(0, 0), tzinfo=TZ)


def tidy_round(label):
    """'Round 20 - Women in League Round' -> en-dash, matching house style."""
    return re.sub(r"\s+-\s+", " – ", str(label))


def team_list_moment(first_fixture_date):
    """Team lists are released about 4pm on the Tuesday before a round.

    This is the moment subscribers get the board, and the gap between it
    and the Sunday-midnight unlock is the only thing a subscription
    actually buys - so the site states both moments explicitly.
    """
    d = first_fixture_date
    while d.weekday() != 1:            # 1 = Tuesday
        d -= timedelta(days=1)
    return datetime.combine(d, time(16, 0), tzinfo=TZ)


WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six",
         7: "seven", 8: "eight"}


def round_number(label):
    m = re.search(r"(\d+)", str(label))
    return int(m.group(1)) if m else None


# Truncated / variant slugs seen in the logs. predictions.csv carries a
# Round 19 row with away_team="cronulla", which silently failed to join
# against results and left that game ungraded. Normalise on the way in so
# the archive grades what actually happened; fix the scraper separately.
SLUG_ALIAS = {
    "cronulla": "cronulla-sutherland-sharks",
    "cronulla-sharks": "cronulla-sutherland-sharks",
    "manly": "manly-warringah-sea-eagles",
    "manly-sea-eagles": "manly-warringah-sea-eagles",
    "canterbury": "canterbury-bankstown-bulldogs",
    "canterbury-bulldogs": "canterbury-bankstown-bulldogs",
    "st-george": "st-george-illawarra-dragons",
    "north-queensland": "north-queensland-cowboys",
    "south-sydney": "south-sydney-rabbitohs",
    "new-zealand-warriors": "warriors",
}


def norm(df, cols=("home_team", "away_team")):
    for c in cols:
        if c in df.columns:
            df[c] = df[c].map(lambda s: SLUG_ALIAS.get(str(s), s))
    return df


def esc(s):
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def pretty(slug):
    """canterbury-bankstown-bulldogs -> Canterbury-Bankstown Bulldogs"""
    return "-".join(w.capitalize() for w in str(slug).split("-"))


def head_sha():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=HERE,
                              capture_output=True, text=True, timeout=10
                              ).stdout.strip() or "unversioned"
    except Exception:
        return "unversioned"


# ------------------------------------------------------------------ data
def load():
    preds = pd.read_csv(os.path.join(HERE, "predictions.csv"))
    preds = preds[preds.season == SEASON].copy()
    preds["date"] = pd.to_datetime(preds.date).dt.date
    preds = norm(preds)

    comp = pd.read_csv(os.path.join(HERE, "model_comparison.csv"))
    comp["date"] = pd.to_datetime(comp.date).dt.date
    comp = norm(comp)
    comp = comp.drop_duplicates(["date", "home_team", "away_team"], keep="first")

    res = pd.read_csv(os.path.join(HERE, "nrl_results.csv"))
    res = res[res.season == SEASON].copy()
    res["date"] = pd.to_datetime(res.date).dt.date
    res = norm(res)

    scores = None
    p = os.path.join(HERE, "predicted_scores.csv")
    if os.path.exists(p):
        scores = pd.read_csv(p)
        scores["date"] = pd.to_datetime(scores.kickoff).dt.date

    key = ["date", "home_team", "away_team"]
    df = preds.merge(
        comp[key + ["run_date", "p_base", "p_market", "p_blend", "forecast_rain_mm"]],
        on=key, how="left", suffixes=("", "_c"))
    df = df.merge(
        res[key + ["home_score", "away_score"]], on=key, how="left")
    if scores is not None:
        s = norm(scores.rename(columns={"home": "home_team", "away": "away_team"}))
        df = df.merge(s[key + ["med_home", "med_away"]], on=key, how="left")
    else:
        df["med_home"] = np.nan
        df["med_away"] = np.nan

    # headline probability: full lineup-aware model where logged, else the
    # simple team model that produced the Monday provisional board
    df["p_model"] = df.p_base.fillna(df.p_home_win)
    df["frozen"] = df.run_date.fillna(df.run_date_c if "run_date_c" in df else np.nan)
    if "run_date_c" in df.columns:
        df["frozen"] = df.run_date_c.fillna(df.run_date)
    df["rnd"] = df["round"].apply(round_number)
    return df.dropna(subset=["rnd"]).sort_values(["rnd", "date"])


def grade(row):
    """Return (winner_slug, model_correct) or (None, None) if ungraded."""
    if pd.isna(row.home_score) or pd.isna(row.away_score):
        return None, None
    if row.home_score == row.away_score:
        return "draw", None
    home_won = row.home_score > row.away_score
    winner = row.home_team if home_won else row.away_team
    picked_home = row.p_model > 0.5
    return winner, bool(picked_home == home_won)


# ------------------------------------------------------------------- CSS
CSS = """
:root{
  --paper:#EEF1F4; --card:#FFFFFF; --ink:#10161D; --muted:#5C6874;
  --rule:#C9D2DA; --seal:#2D4B7A; --hit:#1B6B49; --miss:#A8402C;
  --bar-model:#2D4B7A; --bar-market:#9AA9B8;
}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--paper);color:var(--ink);
  font-family:"IBM Plex Sans",system-ui,-apple-system,sans-serif;
  font-size:16px;line-height:1.55}
.wrap{max-width:900px;margin:0 auto;padding:0 20px 72px}
a{color:var(--seal)}
h1,h2,h3,.disp{font-family:"Barlow Condensed","IBM Plex Sans",sans-serif;
  font-weight:700;letter-spacing:.01em;text-transform:uppercase;margin:0}
.num{font-family:"IBM Plex Mono",ui-monospace,monospace;
  font-variant-numeric:tabular-nums}

header{border-bottom:2px solid var(--ink);margin-bottom:34px}
.masthead{display:flex;justify-content:space-between;align-items:flex-end;
  gap:16px;flex-wrap:wrap;padding:30px 0 14px}
.masthead h1{font-size:clamp(30px,6vw,46px);line-height:.95}
.masthead .tag{color:var(--muted);font-size:13px;letter-spacing:.14em;
  text-transform:uppercase}
nav a{margin-left:18px;font-size:14px;text-decoration:none;
  border-bottom:1px solid var(--rule)}
nav a:hover{border-color:var(--seal)}

.lede{max-width:62ch;color:var(--muted);margin:0 0 34px}
.lede strong{color:var(--ink)}

/* the seal - signature device */
.seal{border:1px solid var(--rule);background:var(--card);border-radius:2px;
  padding:20px 22px;margin:0 0 34px;display:flex;gap:22px;flex-wrap:wrap;
  align-items:center;justify-content:space-between}
.seal.locked{border-color:var(--seal);border-left:5px solid var(--seal)}
.seal .stamp{font-size:12px;letter-spacing:.16em;text-transform:uppercase;
  color:var(--seal);font-weight:600}
.seal .when{font-size:13px;color:var(--muted)}
.buy{display:flex;gap:10px;flex-wrap:wrap}
.buy a{display:inline-block;text-decoration:none;padding:9px 15px;
  border:1px solid var(--seal);border-radius:2px;font-size:14px;
  font-weight:600;white-space:nowrap}
.buy a.primary{background:var(--seal);color:#fff}
.buy .soon{font-size:15px;font-weight:600;line-height:1.55;text-align:right}
.buy .soon-b{font-weight:400;color:var(--muted);font-size:14px}
.buy .soon-c{display:inline-block;margin-top:6px;font-weight:400;
  font-size:11px;letter-spacing:.13em;text-transform:uppercase;
  color:var(--seal);border:1px solid var(--seal);border-radius:2px;
  padding:3px 8px}
@media (max-width:560px){.buy .soon{text-align:left}}
.seal.locked{display:block}
.lock-head{display:flex;justify-content:space-between;gap:20px;
  flex-wrap:wrap;align-items:flex-start}
.timing{display:grid;grid-template-columns:1fr auto 1fr;gap:0;
  align-items:stretch;margin:20px 0 0;border:1px solid var(--rule);
  background:var(--paper)}
.timing>div{padding:14px 16px}
.t-now{background:var(--seal);color:#fff}
.t-now .t-who{color:#B9CCE6}
.t-now .t-note{color:#D7E2EF}
.t-who{font-size:11px;letter-spacing:.15em;text-transform:uppercase;
  font-weight:600;color:var(--muted);margin-bottom:3px}
.t-when{font-family:"Barlow Condensed",sans-serif;font-weight:700;
  font-size:21px;text-transform:uppercase;line-height:1.15;margin-bottom:6px}
.t-note{font-size:13px;line-height:1.45;color:var(--muted);max-width:34ch}
.t-gap{display:flex;align-items:center;justify-content:center;
  background:var(--card);border-left:1px solid var(--rule);
  border-right:1px solid var(--rule);padding:0 14px!important}
.t-gap span{font-family:"IBM Plex Mono",monospace;font-size:11px;
  letter-spacing:.13em;text-transform:uppercase;color:var(--seal);
  white-space:nowrap;writing-mode:vertical-rl;transform:rotate(180deg)}
.lock-foot{margin:16px 0 0;font-size:14px;color:var(--muted);max-width:70ch}
.lock-foot strong{color:var(--ink)}
@media (max-width:640px){
  .timing{grid-template-columns:1fr}
  .t-gap{border-left:none;border-right:none;border-top:1px solid var(--rule);
    border-bottom:1px solid var(--rule);padding:7px 14px!important}
  .t-gap span{writing-mode:horizontal-tb;transform:none}
  .t-note{max-width:none}
}

/* record strip */
.record{display:grid;grid-template-columns:repeat(auto-fit,minmax(118px,1fr));
  gap:1px;background:var(--rule);border:1px solid var(--rule);margin:0 0 34px}
.record div{background:var(--card);padding:14px 16px}
.record .k{font-size:11px;letter-spacing:.13em;text-transform:uppercase;
  color:var(--muted)}
.record .v{font-size:26px;font-weight:600;line-height:1.2}

/* fixtures */
.game{border-top:1px solid var(--rule);padding:18px 0}
.game:last-child{border-bottom:1px solid var(--rule)}
.gmeta{font-size:12px;color:var(--muted);letter-spacing:.06em;
  text-transform:uppercase;margin-bottom:7px;
  display:flex;gap:12px;flex-wrap:wrap}
.teams{display:flex;justify-content:space-between;align-items:baseline;
  gap:14px;flex-wrap:wrap}
.teams .t{font-size:19px;font-weight:600}
.teams .t .away{color:var(--muted);font-weight:400}
.score{font-size:19px;font-weight:600}
.verdict{font-size:12px;letter-spacing:.1em;text-transform:uppercase;
  font-weight:700}
.hit{color:var(--hit)} .miss{color:var(--miss)} .nores{color:var(--muted)}

/* probability bar */
.bars{margin-top:11px;display:grid;gap:5px;max-width:560px}
.bar{display:grid;grid-template-columns:56px 1fr 46px;gap:9px;
  align-items:center;font-size:12px;color:var(--muted)}
.track{height:7px;background:#DDE3E9;border-radius:1px;overflow:hidden}
.fill{height:100%}
.fill.model{background:var(--bar-model)}
.fill.market{background:var(--bar-market)}
.bar .val{text-align:right;color:var(--ink)}

/* round index */
.rounds{display:grid;gap:1px;background:var(--rule);
  border:1px solid var(--rule)}
.rounds a,.rounds .row{display:flex;justify-content:space-between;
  align-items:center;gap:14px;background:var(--card);padding:15px 17px;
  text-decoration:none;color:var(--ink)}
.rounds a:hover{background:#F6F8FA}
.rounds .row.sealed{background:#E7ECF1;color:var(--muted)}
.rounds .lbl{font-weight:600}
.rounds .right{font-size:13px;color:var(--muted);text-align:right}


/* weekly clock - the embargo, drawn */
.clock{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));
  gap:1px;background:var(--rule);border:1px solid var(--rule);margin:0 0 8px}
.clock div{background:var(--card);padding:13px 15px}
.clock .d{font-size:11px;letter-spacing:.13em;text-transform:uppercase;
  color:var(--seal);font-weight:600;margin-bottom:4px}
.clock .w{font-size:14px;line-height:1.4}
.clock div.open{background:var(--seal);color:#fff}
.clock div.open .d{color:#B9CCE6}

/* sequential layers - each one adjusts the number above it */
.layer{display:grid;grid-template-columns:34px 1fr;gap:16px;
  border-left:2px solid var(--rule);padding:0 0 26px 20px;margin-left:8px;
  position:relative}
.layer:last-of-type{border-left-color:transparent;padding-bottom:6px}
.layer .n{font-family:"IBM Plex Mono",monospace;font-size:13px;font-weight:600;
  color:#fff;background:var(--seal);width:26px;height:26px;border-radius:50%;
  display:flex;align-items:center;justify-content:center;
  position:absolute;left:-14px;top:0}
.layer .body{grid-column:1/-1}
.layer h3{font-size:19px;margin:0 0 6px}
.layer p{margin:0 0 8px;color:var(--muted);max-width:60ch}
.layer .params{font-family:"IBM Plex Mono",monospace;font-size:12.5px;
  color:var(--ink);background:#E4E9EE;border-radius:2px;padding:8px 11px;
  display:inline-block;line-height:1.7}

/* concurrent scenarios - fan out sideways, deliberately unlike the stack */
.fan{display:grid;grid-template-columns:repeat(auto-fit,minmax(168px,1fr));
  gap:12px;margin:16px 0 8px}
.fan .arm{border:1px solid var(--rule);background:var(--card);
  border-top:3px solid var(--seal);padding:13px 14px;border-radius:2px}
.fan .arm .nm{font-weight:600;font-size:15px;margin-bottom:4px}
.fan .arm .ds{font-size:13px;color:var(--muted);line-height:1.45}
.fan .arm.live{border-top-color:var(--hit)}
.tbl{width:100%;border-collapse:collapse;font-size:14px;margin:8px 0 6px}
.tbl th{text-align:left;font-size:11px;letter-spacing:.11em;
  text-transform:uppercase;color:var(--muted);font-weight:600;
  border-bottom:1px solid var(--ink);padding:7px 9px 7px 0}
.tbl td{padding:8px 9px 8px 0;border-bottom:1px solid var(--rule)}
.tbl td.num,.tbl th.num{text-align:right;padding-right:0;
  font-family:"IBM Plex Mono",monospace;font-variant-numeric:tabular-nums}
.tbl tr.best td{font-weight:600}
.caption{font-size:13px;color:var(--muted);margin:4px 0 30px;max-width:62ch}
.limits li{color:var(--muted);margin-bottom:9px;max-width:62ch}
.limits strong{color:var(--ink)}
h2.sec{font-size:24px;margin:46px 0 6px}
.sub{color:var(--muted);max-width:62ch;margin:0 0 22px}


/* Brier explainer - <details> so it works with no JS at all */
.record .k{display:flex;align-items:center;gap:5px}
details.tip{position:relative;display:inline}
details.tip summary{list-style:none;cursor:pointer;width:14px;height:14px;
  border:1px solid var(--muted);color:var(--muted);border-radius:50%;
  font-size:10px;line-height:12px;text-align:center;display:inline-block;
  font-family:"IBM Plex Sans",sans-serif}
details.tip summary::-webkit-details-marker{display:none}
details.tip summary:hover{border-color:var(--seal);color:var(--seal)}
details.tip[open] summary{background:var(--seal);color:#fff;
  border-color:var(--seal)}
details.tip .pop{position:absolute;z-index:20;top:22px;left:-10px;width:270px;
  background:var(--card);border:1px solid var(--seal);border-radius:2px;
  padding:12px 14px;font-size:13px;line-height:1.5;color:var(--muted);
  text-transform:none;letter-spacing:0;font-weight:400;
  box-shadow:0 6px 20px rgba(16,22,29,.13)}
details.tip .pop strong{color:var(--ink)}
@media (max-width:560px){details.tip .pop{left:auto;right:-10px;width:240px}}

/* how the model is tracking against the market */
.versus{border:1px solid var(--rule);border-left:5px solid var(--muted);
  background:var(--card);padding:18px 20px;margin:0 0 34px}
.versus h3{font-size:17px;margin:0 0 8px}
.versus p{margin:0 0 9px;color:var(--muted);font-size:14.5px;max-width:64ch}
.versus p:last-child{margin-bottom:0}
.versus .num{color:var(--ink);font-weight:600}

footer{margin-top:52px;padding-top:20px;border-top:1px solid var(--rule);
  font-size:13px;color:var(--muted)}
footer code{font-size:12px}
@media (max-width:520px){
  .bar{grid-template-columns:50px 1fr 40px}
  .record .v{font-size:22px}
}
@media (prefers-reduced-motion:reduce){*{transition:none!important}}
"""

FONTS = ('<link rel="preconnect" href="https://fonts.googleapis.com">'
         '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
         '<link href="https://fonts.googleapis.com/css2?'
         'family=Barlow+Condensed:wght@600;700&'
         'family=IBM+Plex+Mono:wght@400;600&'
         'family=IBM+Plex+Sans:wght@400;600&display=swap" rel="stylesheet">')


def shell(title, body, desc, home="index.html", canonical=""):
    return f"""<!DOCTYPE html>
<html lang="en-AU"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{esc(title)}</title>
<meta name="description" content="{esc(desc)}">
{f'<link rel="canonical" href="{SITE_URL}/{canonical}">' if SITE_URL and canonical else ''}
<meta property="og:type" content="website">
<meta property="og:title" content="{esc(title)}">
<meta property="og:description" content="{esc(desc)}">
{FONTS}
<style>{CSS}</style>
</head><body><div class="wrap">
<header><div class="masthead">
  <div><h1><a href="{home}" style="color:inherit;text-decoration:none">{SITE_NAME}</a></h1>
       <div class="tag">{TAGLINE}</div></div>
  <nav><a href="{home}">Archive</a><a href="method.html">Method</a></nav>
</div></header>
{body}
<footer>
  <p>Every forecast published here was recorded to a version history before
  the relevant round began. Records are not amended after the result.
  Incorrect selections remain on the page.</p>
  <p>Forecasts are the output of a statistical model and are published for
  research purposes. They are not betting advice. Gamble responsibly.
  Gambling Help 1800 858 858.</p>
  <p class="num">Built {esc(now_syd().strftime('%Y-%m-%d %H:%M'))} AEST &middot;
     <code>{esc(head_sha())}</code></p>
</footer>
</div></body></html>
"""


def bar_row(label, p, cls):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return ""
    return (f'<div class="bar"><span>{label}</span>'
            f'<div class="track"><div class="fill {cls}" '
            f'style="width:{max(0.0, min(1.0, float(p)))*100:.0f}%"></div></div>'
            f'<span class="val num">{float(p)*100:.0f}%</span></div>')


def render_round(rnd, g, unlock):
    label = tidy_round(g["round"].iloc[0])
    frozen = pd.to_datetime(g["frozen"].dropna()).min()
    frozen_txt = frozen.strftime("%d %b %Y") if pd.notna(frozen) else "—"

    graded = [grade(r) for r in g.itertuples()]
    ok = [c for _, c in graded if c is not None]
    hits, n = sum(ok), len(ok)

    rows = []
    for r, (winner, correct) in zip(g.itertuples(), graded):
        pick = r.home_team if r.p_model > 0.5 else r.away_team
        conf = r.p_model if r.p_model > 0.5 else 1 - r.p_model

        if correct is True:
            v = '<span class="verdict hit">Correct</span>'
        elif correct is False:
            v = '<span class="verdict miss">Missed</span>'
        elif winner == "draw":
            v = '<span class="verdict nores">Drawn</span>'
        else:
            v = '<span class="verdict nores">Awaiting result</span>'

        score = ""
        if pd.notna(r.home_score):
            score = (f'<span class="score num">{int(r.home_score)}'
                     f'&ndash;{int(r.away_score)}</span>')

        meta = [r.date.strftime("%a %d %b")]
        if isinstance(getattr(r, "venue", None), str):
            meta.append(esc(r.venue))
        rain = getattr(r, "forecast_rain_mm", np.nan)
        if pd.notna(rain) and float(rain) >= 5:
            meta.append(f"{float(rain):.0f}mm forecast")
        if pd.notna(getattr(r, "med_home", np.nan)):
            meta.append(f"sim {int(r.med_home)}&ndash;{int(r.med_away)}")

        bars = (bar_row("Model", r.p_model, "model")
                + bar_row("Market", getattr(r, "p_market", np.nan), "market"))

        rows.append(f"""<div class="game">
  <div class="gmeta">{' &middot; '.join(meta)}</div>
  <div class="teams">
    <span class="t">{esc(pretty(r.home_team))}
      <span class="away">v {esc(pretty(r.away_team))}</span></span>
    <span>{score} {v}</span>
  </div>
  <div class="gmeta" style="margin:9px 0 0">
    Called <strong style="color:var(--ink)">{esc(pretty(pick))}</strong>
    at <span class="num">{conf*100:.0f}%</span>
  </div>
  {f'<div class="bars">{bars}</div>' if bars else ''}
</div>""")

    hdr = (f'<div class="record"><div><div class="k">Round</div>'
           f'<div class="v num">{rnd}</div></div>'
           f'<div><div class="k">Fixtures</div><div class="v num">{len(g)}</div></div>'
           f'<div><div class="k">Correct</div>'
           f'<div class="v num">{hits}/{n}</div></div>'
           f'<div><div class="k">Accuracy</div><div class="v num">'
           f'{(hits/n*100):.0f}%</div></div></div>' if n else
           '<div class="record"><div><div class="k">Round</div>'
           f'<div class="v num">{rnd}</div></div>'
           f'<div><div class="k">Fixtures</div><div class="v num">{len(g)}</div>'
           '</div><div><div class="k">Status</div>'
           '<div class="v">Awaiting results</div></div></div>')

    body = f"""
<h2 style="font-size:30px">{esc(label)}</h2>
<div class="seal">
  <div>
    <div class="stamp">Sealed {esc(frozen_txt)} &middot; Opened
      {esc(unlock.strftime('%d %b, midnight'))}</div>
    <div class="when">Recorded before the first kickoff and unchanged
      since.</div>
  </div>
</div>
{hdr}
{''.join(rows)}
<p style="margin-top:26px"><a href="index.html">All rounds</a></p>
"""
    desc = (f"{label} {SEASON} NRL forecast. Model probabilities recorded "
            f"before kickoff and graded against results.")
    return shell(f"{label} | {SITE_NAME} {SEASON}", body, desc,
                 canonical=f"round-{rnd}.html")


def market_block():
    """Model vs market on the matched sample, computed fresh each build.

    Accuracy and log loss disagree here and the copy says so: the model
    picks more winners while the market is better calibrated. Publishing
    only the flattering one of those two is the thing this whole site is
    supposed to not do.
    """
    comp = pd.read_csv(os.path.join(HERE, "model_comparison.csv"))
    comp = norm(comp).drop_duplicates(["date", "home_team", "away_team"],
                                      keep="first")
    d = comp.dropna(subset=["actual_home_win", "p_market", "p_base"])
    if len(d) < 4:
        return ""
    y = d.actual_home_win.astype(float)

    def ll(p):
        p = np.clip(pd.to_numeric(p, errors="coerce"), 1e-6, 1 - 1e-6)
        return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())

    mh = int(((d.p_base > .5).astype(int) == y).sum())
    kh = int(((d.p_market > .5).astype(int) == y).sum())
    lm, lk = ll(d.p_base), ll(d.p_market)
    n = len(d)

    if mh > kh:
        lead = (f"the model selected <span class=\"num\">{mh}</span> "
                f"winners against the market's <span class=\"num\">{kh}"
                f"</span>")
    elif mh < kh:
        lead = (f"the market selected <span class=\"num\">{kh}</span> "
                f"winners against the model's <span class=\"num\">{mh}"
                f"</span>")
    else:
        lead = (f"the model and the market each selected "
                f"<span class=\"num\">{mh}</span> winners")

    if lk < lm:
        cal = ("The market&rsquo;s probabilities remain better calibrated, "
               "indicating that model errors have occurred at high "
               "confidence.")
    else:
        cal = ("Model probabilities are also better calibrated than the "
               "market&rsquo;s over this sample, which is the more demanding "
               "of the two tests.")

    return f"""<div class="versus">
  <h3>Performance against the market</h3>
  <p>Across the <span class="num">{n}</span> matches with a recorded closing
     price, {lead}.</p>
  <p>{cal} Log loss is <span class="num">{lk:.3f}</span> for the market
     against <span class="num">{lm:.3f}</span> for the model. Reducing this
     gap is the current priority.</p>
  <p>A sample of this size is not conclusive. The full season provides the
     test.</p>
</div>"""


def render_index(df, published, sealed):
    done = df[df.rnd.isin(published)].copy()
    graded = [grade(r) for r in done.itertuples()]
    ok = [c for _, c in graded if c is not None]
    hits, n = sum(ok), len(ok)

    y = []
    p = []
    for r, (w, c) in zip(done.itertuples(), graded):
        if c is None or w == "draw":
            continue
        y.append(1.0 if r.home_score > r.away_score else 0.0)
        p.append(float(r.p_model))
    brier = float(np.mean((np.array(p) - np.array(y)) ** 2)) if p else None

    rec = (f'<div class="record">'
           f'<div><div class="k">Rounds published</div>'
           f'<div class="v num">{len(published)}</div></div>'
           f'<div><div class="k">Graded picks</div><div class="v num">{n}</div></div>'
           f'<div><div class="k">Correct</div><div class="v num">{hits}</div></div>'
           f'<div><div class="k">Accuracy</div>'
           f'<div class="v num">{(hits/n*100):.0f}%</div></div>'
           + (f'<div><div class="k">Brier {BRIER_POP}</div>'
              f'<div class="v num">{brier:.3f}</div></div>'
              if brier is not None else "")
           + '</div>') if n else ""

    rows = []
    for rnd in sealed:
        g = df[df.rnd == rnd]
        unlock = unlock_moment(max(g.date))
        rows.append(f"""<div class="row sealed">
  <span class="lbl">Round {rnd}: sealed</span>
  <span class="right">Opens {esc(unlock.strftime('%a %d %b'))}, midnight<br>
    {len(g)} fixtures</span></div>""")
    for rnd in sorted(published, reverse=True):
        g = df[df.rnd == rnd]
        gr = [grade(r) for r in g.itertuples()]
        ok2 = [c for _, c in gr if c is not None]
        right = (f"{sum(ok2)}/{len(ok2)} correct" if ok2 else "results pending")
        part = ("" if len(g) >= 6 else
                ' <span style="font-weight:400;color:var(--muted)">'
                '(partial round: model commenced mid-round)</span>')
        rows.append(f"""<a href="round-{rnd}.html">
  <span class="lbl">{esc(tidy_round(g['round'].iloc[0]))}{part}</span>
  <span class="right">{right}<br>{esc(min(g.date).strftime('%d %b'))}</span></a>""")

    cta = ""
    if sealed:
        nxt = min(sealed)
        g = df[df.rnd == nxt]
        u = unlock_moment(max(g.date))
        tl = team_list_moment(min(g.date))
        gap = max(1, round((u - tl).total_seconds() / 86400))
        cta = f"""<div class="seal locked" id="subscribe">
  <div class="lock-head">
    <div>
      <div class="stamp">Round {nxt} is sealed</div>
      <div class="when">The board is computed when team lists are released
        and remains sealed until the round concludes.</div>
    </div>
    <div class="buy">
      <span class="soon">{PRICE_SEASON} &middot; about $1 a round<br>
        <span class="soon-b">{PRICE_WEEK}</span><br>
        <span class="soon-c">Subscriptions open Round 1, 2027</span></span>
    </div>
  </div>
  <div class="timing">
    <div class="t-now">
      <div class="t-who">Subscribers</div>
      <div class="t-when">{esc(tl.strftime('%A %-d %B, 4pm'))}</div>
      <div class="t-note">Team lists released. The forecast is computed,
        recorded and available to subscribers the same afternoon.</div>
    </div>
    <div class="t-gap"><span>{WORDS.get(gap, gap)} days later</span></div>
    <div class="t-later">
      <div class="t-who">Everyone</div>
      <div class="t-when">{esc(u.strftime('%A %-d %B, midnight'))}</div>
      <div class="t-note">The round has concluded. The same board is
        published here in full, at no cost, permanently.</div>
    </div>
  </div>
  <p class="lock-foot"><strong>A subscription does not purchase exclusive
    information.</strong> Every forecast is published to everyone. It purchases
    {WORDS.get(gap, gap)} days of earlier access, and funds the research.</p>
</div>"""

    body = f"""
<p class="lede">A statistical model forecasts every NRL fixture. Each
forecast is <strong>recorded to a public version history before the round
begins</strong> and cannot be altered afterwards. When the round finishes, the
full board is published here: every selection, its probability, and the result.
<strong>The archive is free and remains free.</strong> A subscription changes
only the timing of access.</p>
{rec}
{market_block()}
{cta}
<h2 style="font-size:22px;margin:34px 0 14px">{SEASON} season</h2>
<div class="rounds">{''.join(rows)}</div>

<h2 id="method" style="font-size:22px;margin:44px 0 12px">Method</h2>
<p class="lede">Six layers, applied in sequence: a team Elo rating, the named
17, player ratings derived from career match statistics, injury and
availability, the venue rainfall forecast, and the market price. Five competing
versions of the model forecast every fixture simultaneously and all five are
graded.</p>
<p class="lede"><a href="method.html">Read the full method</a></p>
<p class="lede">This is not a tipping service, and no system for profiting
from bookmakers is offered. It is a research project that publishes its
predictions in advance so that they can be tested against results.</p>
<p class="lede">The project depends on data collection, a weather API and
market pricing, and requires several hours of work each week. Subscriptions
fund its continuation.</p>
"""
    desc = (f"Free archive of {SEASON} NRL match forecasts. Probabilities "
            "recorded before kickoff and graded against results.")
    return shell(f"{SITE_NAME} | {SEASON} archive", body, desc,
                 canonical="index.html")



# ------------------------------------------------------------- method page
# Parameters are stated here exactly as they appear in the model source.
# If you change nrl_model.py, injury_adjust.py, stats_ratings.py or
# weather_features.py, change these too - a method page that drifts from
# the code is worse than no method page.
PARAMS = dict(elo_k=32, hfa=55, regress=30, shrink_games=8,
              returning=0.80, in_doubt=0.60, wet_mm=5.0, blend_w=50,
              spine="Fullback, Five-eighth, Halfback, Hooker")

ARMS = [
    ("Version A: baseline", "p_base",
     "The full model with no experimental adjustment. This is the number "
     "published as the forecast."),
    ("Version B: fitness ramp", "p_retadj",
     "Identical to A, except that returning players are discounted on a "
     "graded ramp rather than a flat multiplier."),
    ("Version C: weather", "p_wet",
     "Identical to A, with an adjustment that shrinks short-priced "
     "favourites when heavy rain is forecast at the venue."),
    ("Market", "p_market",
     "The bookmaker price with the margin removed. Not a model, but the "
     "benchmark against which each model is measured."),
    ("Blend", "p_blend",
     "A 50/50 logit average of Version A and the market. If the model holds "
     "information the market lacks, this should outperform both."),
]


def trial_table(df):
    """Live standings of the concurrent model arms, graded games only."""
    comp = pd.read_csv(os.path.join(HERE, "model_comparison.csv"))
    comp = norm(comp).drop_duplicates(["date", "home_team", "away_team"],
                                      keep="first")
    d = comp.dropna(subset=["actual_home_win"])
    rows = []
    for label, col, _ in ARMS:
        if col not in d.columns:
            continue
        p = pd.to_numeric(d[col], errors="coerce").dropna()
        if not len(p):
            continue
        y = d.loc[p.index, "actual_home_win"].astype(float)
        pc = p.clip(1e-6, 1 - 1e-6)
        ll = float(-(y * np.log(pc) + (1 - y) * np.log(1 - pc)).mean())
        rows.append((label, len(p), float(((pc > .5).astype(int) == y).mean()), ll))
    if not rows:
        return "", 0
    best = min(r[3] for r in rows)
    body = "".join(
        f'<tr class="{"best" if abs(ll - best) < 1e-9 else ""}"><td>{lab}</td>'
        f'<td class="num">{n}</td><td class="num">{acc*100:.0f}%</td>'
        f'<td class="num">{ll:.4f}</td></tr>'
        for lab, n, acc, ll in rows)
    return (f'<table class="tbl"><thead><tr><th>Version</th>'
            f'<th class="num">Graded</th><th class="num">Correct</th>'
            f'<th class="num">Log loss</th></tr></thead>'
            f'<tbody>{body}</tbody></table>'), max(r[1] for r in rows)


def scenario_example():
    """Pull the largest live lineup scenario as a worked example."""
    p = os.path.join(HERE, "scenarios.csv")
    if not os.path.exists(p):
        return ""
    sc = pd.read_csv(p)
    if not len(sc):
        return ""
    r = sc.reindex(sc.swing_pts.abs().sort_values(ascending=False).index).iloc[0]
    return f"""<div class="fan">
  <div class="arm"><div class="nm">Declared fit</div>
    <div class="ds">{esc(pretty(r.home))} win
      <strong class="num">{r.p_fit*100:.1f}%</strong></div></div>
  <div class="arm live"><div class="nm">As named</div>
    <div class="ds">{esc(pretty(r.home))} win
      <strong class="num">{r.p_named*100:.1f}%</strong></div></div>
  <div class="arm"><div class="nm">Withdrawn</div>
    <div class="ds">{esc(pretty(r.home))} win
      <strong class="num">{r.p_withdrawn*100:.1f}%</strong></div></div>
</div>
<p class="caption">Example from the current round:
<strong>{esc(r.player)}</strong> ({esc(pretty(r.team))}, returning from injury).
The three cases are computed together. The spread between them, here
{r.swing_pts:.1f} points, quantifies the effect of the selection. Most players
move the probability by less than one point. Publishing the spread in advance
prevents a late withdrawal being used retrospectively to explain a
result.</p>"""


def render_method(df):
    q = PARAMS
    matches = len(pd.read_csv(os.path.join(HERE, "nrl_results.csv")))
    table, trial_n = trial_table(df)

    layers = [
        ("Team rating",
         "Each club holds an Elo rating updated after every result. Defeating "
         "a higher rated opponent produces a larger adjustment than defeating "
         "a lower rated one. Ratings regress toward the mean between seasons "
         "to limit the persistence of historical form.",
         f"start 1500 &middot; K = {q['elo_k']} &middot; home advantage = "
         f"{q['hfa']} pts<br>between-season regression = {q['regress']}% "
         f"&middot; trained on {matches:,} matches since 2009"),
        ("Who is actually playing",
         "Team lists are published on Tuesday. The model reads the named 17, "
         "calculates retention against the previous week&rsquo;s side, and "
         "measures continuity across the spine positions, which carry a "
         "side&rsquo;s structure.",
         f"spine = {q['spine']}<br>retention and spine continuity computed "
         "per side, per round"),
        ("Player quality from statistics",
         "Each player holds a rating derived from per-game career statistics "
         "rather than reputation. Players with limited appearances are shrunk "
         "toward the league average, so that small samples do not produce "
         "extreme ratings.",
         f"shrinkage prior = {q['shrink_games']} games<br>ratings refresh "
         "weekly as new match statistics land"),
        ("Availability",
         "The NRL casualty ward is collected each week. Players returning from "
         "injury are discounted for match fitness. Players listed as in doubt "
         "are discounted further, as the named side is not always the side "
         "that takes the field.",
         f"returning from injury &times; {q['returning']:.2f}<br>"
         f"listed in doubt &times; {q['in_doubt']:.2f}"),
        ("Conditions",
         "The rainfall forecast for each venue is retrieved at prediction "
         "time. Wet conditions historically compress the margin between "
         "sides: favourites win less frequently than their rating implies "
         "and scoring falls. The magnitude of the adjustment is estimated "
         "from the historical record.",
         f"wet threshold = {q['wet_mm']:.0f}mm forecast rainfall<br>"
         "effect fitted across all historical matches with match-day rainfall"),
        ("The market",
         "The bookmaker price is adjusted to remove the margin and averaged "
         "with the model on the log-odds scale. The market aggregates "
         "information not available to the model, and the model incorporates "
         "lineup detail the market may price slowly. Neither input is used in "
         "isolation.",
         f"blend = {q['blend_w']}/{100-q['blend_w']} logit average of "
         "model and de-vigged price"),
    ]
    stack = "".join(
        f'<div class="layer"><div class="n">{i}</div><div class="body">'
        f'<h3>{h}</h3><p>{p}</p><div class="params">{par}</div></div></div>'
        for i, (h, p, par) in enumerate(layers, 1))

    arms = "".join(
        f'<div class="arm{" live" if col == "p_base" else ""}">'
        f'<div class="nm">{lab}</div><div class="ds">{d}</div></div>'
        for lab, col, d in ARMS)

    body = f"""
<h2 style="font-size:32px;margin-bottom:8px">Method</h2>
<p class="lede">This page sets out the model&rsquo;s construction, its
inputs and its limitations. Every parameter listed is the value used in the
code that generated the forecasts in the archive.</p>

<h2 class="sec">The weekly clock</h2>
<p class="sub">The forecast is produced on Tuesday, when team lists are
released. This is the largest single information event in the weekly cycle.
The forecast is recorded at that point and is not revised.</p>
<div class="clock">
  <div><div class="d">Monday</div><div class="w">Results scraped, last
    round graded, player statistics refreshed.</div></div>
  <div><div class="d">Tuesday</div><div class="w">Team lists released. The
    forecast is computed and recorded. Available to subscribers.</div></div>
  <div><div class="d">Thu &ndash; Sun</div><div class="w">Prices and late
    lineup changes are logged, but the recorded forecast does not
    change.</div></div>
  <div class="open"><div class="d">Sunday midnight</div><div class="w">The
    round is published here in full, at no cost.</div></div>
</div>
<p class="caption">Publication is triggered by time, not by data
availability. A collection failure may delay the results column. It cannot
delay the forecast.</p>

<h2 class="sec">Six layers</h2>
<p class="sub">Each layer adjusts the probability produced by the preceding
one. The sequence is deliberate: team strength, squad selection, player
quality, availability, conditions, then market price.</p>
{stack}

<h2 class="sec">Concurrent scenarios</h2>
<p class="sub">Two parallel processes run on every fixture.</p>

<h3 style="font-size:19px;margin:26px 0 6px">Competing versions of the model</h3>
<p class="sub">Five versions forecast every match simultaneously. Version A
is published as the forecast. All five are recorded and graded. This allows the
value of any adjustment to be settled by the record rather than by judgement.
Versions that do not demonstrate an improvement are removed.</p>
<div class="fan">{arms}</div>
{table}
<p class="caption">{SEASON} standings across {trial_n} graded matches. Lower
log loss is better: it rewards confident correct forecasts and penalises
confident errors, which accuracy does not.
<strong>This sample is not sufficient to support a conclusion.</strong>
Separating these versions reliably requires several hundred matches. The
differences between A, B and C currently sit below the level of noise. The
table is published to record the process, not to claim a result.</p>

<h3 style="font-size:19px;margin:32px 0 6px">Lineup scenarios</h3>
<p class="sub">When a player is listed as returning or in doubt, the fixture
is forecast three times: assuming full fitness, as named on the team sheet, and
assuming withdrawal.</p>
{scenario_example()}

<h2 class="sec">Recorded performance</h2>
<p class="sub">Across twelve walk-forward seasons, each predicted using only
prior seasons, the team model selects approximately 65% of matches correctly.
Season-to-season results range between 58% and 74%. Tested against exchange
closing prices on 609 matches, the full model is approximately level with the
market on log loss, and a blend of the two outperformed either input alone.</p>
<p class="sub">That final result is the basis of the project and also its
weakest claim. It was measured once, on historical data, using a blend weight
selected against that same data. The {SEASON} season is the first out-of-sample
test. The market is currently ahead. Whether the gap closes is the research
question, and the archive is the record against which it will be
answered.</p>

<h2 class="sec">Known weaknesses</h2>
<p class="sub">Documented in full, at the same level of detail as the
results.</p>
<ul class="limits">
  <li><strong>Several parameters are set by hand rather than fitted.</strong>
  The Elo K factor, the home advantage, the injury multipliers and the
  shrinkage prior are reasonable values rather than optimised ones. Home
  advantage is treated as identical at every venue, which is demonstrably
  incorrect.</li>
  <li><strong>The wet-weather flag is a hard threshold.</strong>
  {q['wet_mm']:.0f}mm of forecast rainfall is classified as wet and 4.9mm is
  not. This is a coarse treatment and has already produced at least one fixture
  on the wrong side of the boundary.</li>
  <li><strong>Squad-level recovery is not modelled.</strong> Individual
  returning players are discounted, but no feature captures a spine
  reassembling simultaneously. The market prices this effect and the model has
  been exposed by it.</li>
  <li><strong>The largest disagreements with the market have produced the
  weakest results.</strong> Historically, large divergences carried more
  signal. This season they have not. If this persists, the model should shrink
  further toward the market as divergence increases.</li>
  <li><strong>Margins are not modelled.</strong> The model predicts the
  winner but not the margin, discarding most of the information contained in a
  result.</li>
  <li><strong>Features were selected against a single test window.</strong>
  Measured gains are therefore optimistic and some will diminish under proper
  validation.</li>
</ul>

<h2 class="sec">Scope</h2>
<p class="sub">This is not betting advice, and no system for profiting from
bookmakers is offered. Approximately one third of the selections in the archive
are incorrect. No commercial arrangement exists with any wagering operator.
Forecasts are published in advance so that they can be tested against
results.</p>
<p style="margin-top:26px"><a href="index.html">Back to the archive</a></p>
"""
    desc = ("How the NRL forecast model works: Elo team ratings, player "
            "ratings from match statistics, injury and weather adjustment, "
            "and a market blend. Known weaknesses documented.")
    return shell(f"Method | {SITE_NAME}", body, desc,
                 canonical="method.html")


# ------------------------------------------------------------------- main
def main():
    dry = "--dry-run" in sys.argv
    df = load()
    now = now_syd()

    published, sealed = [], []
    for rnd, g in df.groupby("rnd"):
        u = unlock_moment(max(g.date))
        (published if now >= u else sealed).append(int(rnd))
    sealed.sort()

    if not published:
        print("publish_rounds: nothing past its unlock yet")
        return

    if dry:
        for r in sorted(published):
            print(f"  would publish round {r}")
        for r in sealed:
            g = df[df.rnd == r]
            print(f"  sealed: round {r} -> opens "
                  f"{unlock_moment(max(g.date)):%Y-%m-%d %H:%M %Z}")
        return

    os.makedirs(DOCS, exist_ok=True)
    open(os.path.join(DOCS, ".nojekyll"), "w").close()

    logp = os.path.join(DOCS, "publish_log.csv")
    log = (pd.read_csv(logp) if os.path.exists(logp)
           else pd.DataFrame(columns=["round", "unlock", "first_published", "sha"]))
    seen = set(log["round"].astype(int)) if len(log) else set()
    new = []

    for rnd in sorted(published):
        g = df[df.rnd == rnd]
        u = unlock_moment(max(g.date))
        with open(os.path.join(DOCS, f"round-{rnd}.html"), "w") as f:
            f.write(render_round(rnd, g, u))
        if rnd not in seen:
            new.append(dict(round=rnd, unlock=u.strftime("%Y-%m-%d %H:%M"),
                            first_published=now.strftime("%Y-%m-%d %H:%M"),
                            sha=head_sha()))

    with open(os.path.join(DOCS, "index.html"), "w") as f:
        f.write(render_index(df, published, sealed))
    with open(os.path.join(DOCS, "method.html"), "w") as f:
        f.write(render_method(df))

    # A round that has not reached its unlock has no page, so a guessed URL
    # like /round-24.html must 404 rather than reveal anything.
    nf = ('<h2 style="font-size:30px">Page not found</h2>'
          '<p class="lede">This page does not exist. Rounds that have not '
          'concluded are not published. Each round is released at midnight on '
          'the Sunday it ends.</p>'
          '<p class="lede"><a href="index.html">Go to the archive</a></p>')
    with open(os.path.join(DOCS, "404.html"), "w") as f:
        f.write(shell(f"Not found | {SITE_NAME}", nf, "Page not found."))

    if new:
        log = pd.concat([log, pd.DataFrame(new)], ignore_index=True)
        log.sort_values("round").to_csv(logp, index=False)
        print(f"publish_rounds: opened round(s) "
              f"{', '.join(str(r['round']) for r in new)}")

    # Discoverability: the free archive only works as marketing if it is
    # indexed. Sealed rounds are absent from the sitemap because their
    # pages do not exist yet.
    if SITE_URL:
        urls = [f"{SITE_URL}/index.html", f"{SITE_URL}/method.html"] + [
            f"{SITE_URL}/round-{r}.html" for r in sorted(published)]
        today = now.strftime("%Y-%m-%d")
        sm = ('<?xml version="1.0" encoding="UTF-8"?>\n'
              '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
              + "".join(f"  <url><loc>{u}</loc><lastmod>{today}</lastmod></url>\n"
                        for u in urls) + "</urlset>\n")
        with open(os.path.join(DOCS, "sitemap.xml"), "w") as f:
            f.write(sm)
        with open(os.path.join(DOCS, "robots.txt"), "w") as f:
            f.write(f"User-agent: *\nAllow: /\nSitemap: {SITE_URL}/sitemap.xml\n")
    else:
        print("publish_rounds: SITE_URL unset - skipping sitemap/robots")

    nxt = (f"; next unlock round {sealed[0]} at "
           f"{unlock_moment(max(df[df.rnd == sealed[0]].date)):%Y-%m-%d %H:%M}"
           if sealed else "")
    print(f"publish_rounds: {len(published)} round page(s) written"
          f", {len(sealed)} sealed{nxt}")


if __name__ == "__main__":
    main()
