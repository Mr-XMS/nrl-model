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

PRICE_WEEK = "$2 / round"
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


def shell(title, body, desc, home="index.html"):
    return f"""<!DOCTYPE html>
<html lang="en-AU"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{esc(title)}</title>
<meta name="description" content="{esc(desc)}">
<meta property="og:title" content="{esc(title)}">
<meta property="og:description" content="{esc(desc)}">
{FONTS}
<style>{CSS}</style>
</head><body><div class="wrap">
<header><div class="masthead">
  <div><h1><a href="{home}" style="color:inherit;text-decoration:none">{SITE_NAME}</a></h1>
       <div class="tag">{TAGLINE}</div></div>
  <nav><a href="{home}">Archive</a><a href="{home}#method">Method</a></nav>
</div></header>
{body}
<footer>
  <p>Every forecast on this site was recorded and committed to a public
  git history before the round began. Nothing is edited after the fact;
  losses stay on the page.</p>
  <p>Forecasts are the output of a statistical model and are published for
  research and interest. They are not betting advice. Gamble responsibly &mdash;
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
    label = str(g["round"].iloc[0])
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
    <div class="stamp">Sealed {esc(frozen_txt)} &mdash; opened
      {esc(unlock.strftime('%d %b, midnight'))}</div>
    <div class="when">Recorded before the first kickoff and unchanged since.</div>
  </div>
</div>
{hdr}
{''.join(rows)}
<p style="margin-top:26px"><a href="index.html">&larr; All rounds</a></p>
"""
    desc = (f"{label} {SEASON} NRL forecast — model probabilities recorded "
            f"before kickoff, graded against results.")
    return shell(f"{label} — {SITE_NAME} {SEASON}", body, desc)


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
           + (f'<div><div class="k">Brier</div>'
              f'<div class="v num">{brier:.3f}</div></div>' if brier is not None else "")
           + '</div>') if n else ""

    rows = []
    for rnd in sealed:
        g = df[df.rnd == rnd]
        unlock = unlock_moment(max(g.date))
        rows.append(f"""<div class="row sealed">
  <span class="lbl">Round {rnd} &mdash; sealed</span>
  <span class="right">Opens {esc(unlock.strftime('%a %d %b'))}, midnight<br>
    {len(g)} fixtures</span></div>""")
    for rnd in sorted(published, reverse=True):
        g = df[df.rnd == rnd]
        gr = [grade(r) for r in g.itertuples()]
        ok2 = [c for _, c in gr if c is not None]
        right = (f"{sum(ok2)}/{len(ok2)} correct" if ok2 else "results pending")
        rows.append(f"""<a href="round-{rnd}.html">
  <span class="lbl">{esc(str(g['round'].iloc[0]))}</span>
  <span class="right">{right}<br>{esc(min(g.date).strftime('%d %b'))}</span></a>""")

    cta = ""
    if sealed:
        nxt = min(sealed)
        g = df[df.rnd == nxt]
        cta = f"""<div class="seal locked">
  <div>
    <div class="stamp">Round {nxt} is sealed</div>
    <div class="when">It opens free to everyone at midnight
      {esc(unlock_moment(max(g.date)).strftime('%A %d %B'))}.
      Subscribers see it the Tuesday before.</div>
  </div>
  <div class="buy">
    <a class="primary" href="#subscribe">{PRICE_SEASON}</a>
    <a href="#subscribe">{PRICE_WEEK}</a>
  </div>
</div>"""

    body = f"""
<p class="lede">A statistical model forecasts every NRL fixture, and the
forecast is <strong>frozen and committed to a public record before the round
starts</strong>. Once the round is over it is published here in full &mdash;
every call, every probability, every miss.
<strong>The archive is free, permanently.</strong> Subscribing only changes
<em>when</em> you see it.</p>
{cta}
{rec}
<h2 style="font-size:22px;margin:34px 0 14px">{SEASON} season</h2>
<div class="rounds">{''.join(rows)}</div>

<h2 id="method" style="font-size:22px;margin:44px 0 12px">How it works</h2>
<p class="lede">An Elo team rating is combined with player-level ratings built
from career match statistics, adjusted for the named 17, injury returns and
the venue rain forecast, then blended with the market price. Over twelve
walk-forward seasons the team model calls about 65% of matches. Week to week
that number swings hard, and this archive is the honest record of it.</p>
<p class="lede">Nothing here is a tip and nothing is sold as a system for
beating a bookmaker. It is a research project that happens to publish its
predictions in advance, which is the only way a forecast can be tested.</p>
"""
    desc = (f"Free archive of {SEASON} NRL match forecasts — probabilities "
            "recorded before kickoff and graded against results.")
    return shell(f"{SITE_NAME} — {SEASON} archive", body, desc)


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

    if new:
        log = pd.concat([log, pd.DataFrame(new)], ignore_index=True)
        log.sort_values("round").to_csv(logp, index=False)
        print(f"publish_rounds: opened round(s) "
              f"{', '.join(str(r['round']) for r in new)}")

    nxt = (f"; next unlock round {sealed[0]} at "
           f"{unlock_moment(max(df[df.rnd == sealed[0]].date)):%Y-%m-%d %H:%M}"
           if sealed else "")
    print(f"publish_rounds: {len(published)} round page(s) written"
          f", {len(sealed)} sealed{nxt}")


if __name__ == "__main__":
    main()
