"""
NRL weekly updater
==================
Run after each round (or any time):

  python update_nrl.py

What it does:
  1. Re-scrapes the current season from rugbyleagueproject.org
  2. Merges newly completed matches into nrl_results.csv
  3. Grades any previous predictions against the new results (accuracy log)
  4. Retrains the model on the full updated dataset
  5. Predicts the next round of unplayed fixtures -> predictions.csv

Files it maintains alongside itself:
  nrl_results.csv      master results dataset
  predictions.csv      every prediction ever made (graded once results land)
"""
import os, re, sys, time
import urllib.request
from datetime import datetime
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "nrl_results.csv")
PRED_LOG = os.path.join(HERE, "predictions.csv")

HEADERS = {"User-Agent": "Mozilla/5.0 (research; contact via site form)"}
MONTHS = {m: i + 1 for i, m in enumerate(
    ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"])}
ALIASES = {"bulldogs": "canterbury-bankstown-bulldogs", "melbourne": "melbourne-storm",
           "cronulla": "cronulla-sutherland-sharks"}

ROW_RE = re.compile(
    r'<tr>\s*'
    r'<td><a href="/competitions/\d+">.*?</td>\s*'
    r'<td align="right">(?P<date>[^<]*)</td>\s*'
    r'<td>(?P<daytime>[^<]*)</td>\s*'
    r'<td class="team"><a href="/seasons/[^/]+/(?P<home>[^/]+)/summary\.html">[^<]*</a></td>\s*'
    r'<td class="n">(?P<hp>\d+|&nbsp;)(?:<!--[^>]*-->)?\s*</td>\s*'
    r'<td class="team"><a href="/seasons/[^/]+/(?P<away>[^/]+)/summary\.html">[^<]*</a></td>\s*'
    r'<td class="n">(?P<ap>\d+|&nbsp;)(?:<!--[^>]*-->)?\s*</td>\s*'
    r'(?P<rest>.*?)</tr>', re.S)
VENUE_RE = re.compile(r'<a href="/venues/\d+">([^<]+)</a>')


def fetch_season(year):
    url = f"https://www.rugbyleagueproject.org/seasons/nrl-{year}/results.html"
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=30) as r:
        html = r.read().decode("utf-8", errors="ignore")
    played, upcoming = [], []
    cur_round, cur_month = None, None
    for chunk in re.split(r'(?=<tr>)', html):
        rh = re.search(r'<th colspan="11">\s*<a href="/seasons/nrl-\d+/[^"]*">([^<]+)</a>', chunk)
        if rh:
            cur_round = rh.group(1).strip()
            continue
        m = ROW_RE.search(chunk)
        if not m:
            continue
        d = m.group("date").strip()
        dm = re.match(r'([A-Z][a-z]{2})\s+(\d+)', d)
        if dm:
            cur_month = MONTHS[dm.group(1)]
            day = int(dm.group(2))
        elif d.isdigit():
            day = int(d)
        else:
            continue
        vm = VENUE_RE.search(m.group("rest"))
        row = {
            "season": year,
            "round": cur_round,
            "date": f"{year}-{cur_month:02d}-{day:02d}",
            "home_team": ALIASES.get(m.group("home"), m.group("home")),
            "away_team": ALIASES.get(m.group("away"), m.group("away")),
            "venue": vm.group(1) if vm else None,
        }
        if m.group("hp").isdigit():
            row["home_score"] = int(m.group("hp"))
            row["away_score"] = int(m.group("ap"))
            played.append(row)
        else:
            upcoming.append(row)
    return played, upcoming


def merge_results(played):
    df = pd.read_csv(RESULTS, parse_dates=["date"])
    df[["home_team", "away_team"]] = df[["home_team", "away_team"]].replace(ALIASES)
    df = df.drop_duplicates(subset=["season", "date", "home_team", "away_team"], keep="first")
    new = pd.DataFrame(played)
    if not len(new):
        print("WARNING: RLP parser found no completed matches - site layout may have changed")
        return df, 0
    new["date"] = pd.to_datetime(new["date"])
    key = ["season", "date", "home_team", "away_team"]
    merged = pd.concat([df, new], ignore_index=True)
    merged = merged.drop_duplicates(subset=key, keep="first")
    merged = merged.sort_values("date").reset_index(drop=True)
    added = len(merged) - len(df)
    merged.to_csv(RESULTS, index=False)
    return merged, added


def grade_predictions(results):
    """Fill in actual outcomes for previously logged predictions."""
    if not os.path.exists(PRED_LOG):
        return
    preds = pd.read_csv(PRED_LOG, parse_dates=["date"])
    ungraded = preds["actual_home_win"].isna()
    if not ungraded.any():
        return
    res = results.set_index(["date", "home_team", "away_team"])
    graded = 0
    for i in preds[ungraded].index:
        k = (preds.at[i, "date"], preds.at[i, "home_team"], preds.at[i, "away_team"])
        if k in res.index:
            r = res.loc[k]
            if r["home_score"] != r["away_score"]:
                preds.at[i, "actual_home_win"] = int(r["home_score"] > r["away_score"])
                graded += 1
    preds.to_csv(PRED_LOG, index=False)
    done = preds.dropna(subset=["actual_home_win"])
    if len(done):
        correct = ((done["p_home_win"] > 0.5) == (done["actual_home_win"] == 1)).mean()
        brier = ((done["p_home_win"] - done["actual_home_win"]) ** 2).mean()
        print(f"Graded {graded} new predictions. "
              f"Running record: {len(done)} graded, accuracy {correct:.1%}, Brier {brier:.4f}")


def predict_upcoming(results, upcoming):
    sys.path.insert(0, HERE)
    from nrl_model import build_features
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler

    if not upcoming:
        print("No upcoming fixtures found.")
        return

    # only the next round (earliest unplayed round)
    up = pd.DataFrame(upcoming)
    up["date"] = pd.to_datetime(up["date"])

    # GUARD: a fixture the source still lists as unplayed, but whose date has
    # passed, means the upstream results feed has stalled - not that the game
    # is upcoming. Forecasting it would write a "prediction" for a match that
    # has already been decided, which is exactly the claim this project makes
    # it never does. Drop those fixtures and refuse to publish the round.
    today = pd.Timestamp(datetime.now().date())
    stale = up[up["date"] < today]
    if len(stale):
        rounds = ", ".join(sorted(stale["round"].unique()))
        print(f"\n!! STALE RESULTS: {len(stale)} fixture(s) in {rounds} were "
              f"played on or before {stale['date'].max().date()} but the source "
              f"has published no score.")
        print("!! The results feed is behind. Skipping prediction to avoid "
              "logging a forecast for a completed match.")
        print("!! Check https://www.rugbyleagueproject.org/seasons/"
              f"nrl-{datetime.now().year}/results.html")
        up = up[up["date"] >= today]
        if up.empty:
            return

    next_round = up.sort_values("date").iloc[0]["round"]
    up = up[up["round"] == next_round].copy()

    # GUARD: even within the next round, drop any individual fixture already
    # under way. A Thursday-night game must not be "predicted" on Friday.
    now = pd.Timestamp(datetime.now())
    started = up[up["date"] < pd.Timestamp(now.date())]
    if len(started):
        print(f"  skipping {len(started)} fixture(s) already played")
        up = up[up["date"] >= pd.Timestamp(now.date())]
    if up.empty:
        print("No un-started fixtures remain in the next round.")
        return

    # train on all completed matches
    df, X, y, mask, _ = build_features(results)
    tr = mask & (df.season >= 2011)
    scaler = StandardScaler().fit(X[tr])
    lr = LogisticRegression(C=0.1, max_iter=2000).fit(scaler.transform(X[tr]), y[tr])
    gb = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=300,
                                        l2_regularization=1.0, random_state=42).fit(X[tr], y[tr])

    # append fixtures as phantom rows to compute their pre-match features
    phantom = up.copy()
    phantom["home_score"] = 0
    phantom["away_score"] = 0
    combined = pd.concat([results, phantom], ignore_index=True)
    _, Xall, *_ = build_features(combined)
    Xf = Xall.iloc[-len(up):]

    p = 0.5 * lr.predict_proba(scaler.transform(Xf))[:, 1] + 0.5 * gb.predict_proba(Xf)[:, 1]
    up["p_home_win"] = np.round(p, 3)
    up["predicted_winner"] = np.where(p > 0.5, up["home_team"], up["away_team"])
    up["run_date"] = datetime.now().strftime("%Y-%m-%d")
    up["actual_home_win"] = np.nan

    print(f"\nPredictions — {next_round}:")
    for _, r in up.sort_values("date").iterrows():
        fav, pf = (r.home_team, r.p_home_win) if r.p_home_win > 0.5 else (r.away_team, 1 - r.p_home_win)
        print(f"  {r.date.date()}  {r.home_team:32s} v {r.away_team:32s} -> {fav} ({pf:.0%})")

    cols = ["run_date", "season", "round", "date", "home_team", "away_team",
            "venue", "p_home_win", "predicted_winner", "actual_home_win"]
    if os.path.exists(PRED_LOG):
        log = pd.read_csv(PRED_LOG, parse_dates=["date"])
        key = ["date", "home_team", "away_team"]
        # A forecast may be revised until its own kickoff, so a late team
        # change is reflected. After kickoff the row is sealed permanently.
        today = pd.Timestamp(datetime.now().date())
        li, ui = log.set_index(key), up.set_index(key)
        dup = ui.index.isin(li.index)
        keep = log[~li.index.isin(ui.index[dup]) | (log["date"] < today)]
        revise = up[dup & (up["date"] >= today).values]
        fresh = up[~dup]
        n_rev = len(log) - len(keep)
        if n_rev:
            print(f"  revised {n_rev} forecast(s) still before kickoff")
        log = pd.concat([keep, revise[cols], fresh[cols]], ignore_index=True)
        log = log.sort_values(["date", "home_team"]).reset_index(drop=True)
    else:
        log = up[cols]
    log.to_csv(PRED_LOG, index=False)
    print(f"\nLogged to {os.path.basename(PRED_LOG)}")


def main():
    year = datetime.now().year
    print(f"Fetching NRL {year} from Rugby League Project...")
    played, upcoming = fetch_season(year)
    results, added = merge_results(played)
    print(f"Season {year}: {len(played)} completed matches on site, {added} new added to dataset "
          f"({len(results)} total). {len(upcoming)} future fixtures listed.")
    grade_predictions(results)
    # refresh player lineups for any newly completed matches
    try:
        from scrape_players import get_match_index, scrape_all
        idx = get_match_index([year])
        scrape_all(idx, os.path.join(HERE, "lineups.jsonl"),
                   os.path.join(HERE, "scrape_progress.txt"), workers=3, delay=0.35)
        print("Lineups refreshed.")
    except Exception as e:
        print(f"Lineup refresh skipped ({e})")
    predict_upcoming(results, upcoming)


if __name__ == "__main__":
    main()
