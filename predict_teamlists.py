"""
Player-based round forecast from announced team lists
======================================================
Fetches this week's announced 17s from nrl.com (published Tuesdays),
maps players to the rating database, and predicts each remaining match
with the hybrid (Elo + player) model.

  python predict_teamlists.py            # forecast current round
  python predict_teamlists.py 19         # forecast a specific round

Fallback: if a team list can't be fetched or matched, that side uses
last week's fielded 17 (noted in the output).
"""
import json, os, re, sys, unicodedata
import urllib.request
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from nrl_model import build_features
from nrl_player_model import (load_lineups, attach_lineups, build_player_features,
                              fit_player_ratings, SPINE)
from injury_adjust import get_casualty_list, injury_status
from weather_features import add_wet_column, forecast_rain, nrl_city_to_key, WET_MM
from stats_ratings import build_stats_feature, refresh_stats

UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
NICK_TO_SLUG = {
    "Broncos": "brisbane-broncos", "Raiders": "canberra-raiders",
    "Bulldogs": "canterbury-bankstown-bulldogs", "Sharks": "cronulla-sutherland-sharks",
    "Dolphins": "dolphins", "Titans": "gold-coast-titans",
    "Sea Eagles": "manly-warringah-sea-eagles", "Storm": "melbourne-storm",
    "Knights": "newcastle-knights", "Cowboys": "north-queensland-cowboys",
    "Eels": "parramatta-eels", "Panthers": "penrith-panthers",
    "Rabbitohs": "south-sydney-rabbitohs", "Dragons": "st-george-illawarra-dragons",
    "Roosters": "sydney-roosters", "Warriors": "warriors", "Wests Tigers": "wests-tigers",
}

def get_json(url):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))

def norm(s):
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z]", "", s.lower())

def build_name_index(lineups, results):
    """player normalised name -> {rlp_id: set(teams)} using scraped history."""
    # map match_id -> teams
    by_name = {}
    for rec in lineups.values():
        for p in rec["players"]:
            team = rec["home_team"] if p["side"] == "home" else rec["away_team"]
            key = norm(p["player"])
            by_name.setdefault(key, {}).setdefault(p["player_id"], set()).add(team)
    return by_name

def match_player(first, last, team_slug, name_index):
    key = norm(first + last)
    cands = name_index.get(key)
    if not cands:
        # try surname + first initial (e.g. nicknames: Mitch vs Mitchell)
        tail = norm(last)
        init = norm(first)[:1]
        cands = {}
        for k, v in name_index.items():
            if k.endswith(tail) and k.startswith(init):
                for pid, teams in v.items():
                    cands.setdefault(pid, set()).update(teams)
        if not cands:
            return None
    # prefer a candidate who has played for this club
    for pid, teams in cands.items():
        if team_slug in teams:
            return pid
    return next(iter(cands))   # transfer/new club: take any match

def fetch_round_teamlists(season, round_no):
    draw = get_json(f"https://www.nrl.com/draw/data?competition=111&season={season}&round={round_no}")
    out = []
    for f in draw["fixtures"]:
        if f.get("type") != "Match":
            continue
        mc = get_json("https://www.nrl.com" + f["matchCentreUrl"].rstrip("/") + "/data")
        rec = dict(state=f["matchState"],
                   kickoff=mc.get("startTime"),
                   venue_city=mc.get("venueCity"),
                   home_odds=mc.get("homeTeam", {}).get("odds"),
                   away_odds=mc.get("awayTeam", {}).get("odds"),
                   home=NICK_TO_SLUG[f["homeTeam"]["nickName"]],
                   away=NICK_TO_SLUG[f["awayTeam"]["nickName"]], teams={})
        for side in ("homeTeam", "awayTeam"):
            players = mc.get(side, {}).get("players") or []
            named = [p for p in players if p.get("number") and p["number"] <= 17]
            rec["teams"][side[:4]] = [(p["firstName"], p["lastName"], p["position"], p.get("playerId")) for p in named]
        # capture result if played
        if f["matchState"] in ("FullTime", "PostGame"):
            try:
                rec["score"] = (mc["homeTeam"].get("score"), mc["awayTeam"].get("score"))
            except Exception:
                pass
        out.append(rec)
    return out

RETURN_FACTOR = 0.85     # variant B: first 3 games after a 28-200 day absence
COMPARE_LOG = os.path.join(HERE, "model_comparison.csv")

def build_appearances(lineups):
    """pid -> sorted list of appearance dates."""
    apps = {}
    for rec in lineups.values():
        d = pd.to_datetime(rec["date"])
        for p in rec["players"]:
            apps.setdefault(p["player_id"], []).append(d)
    return {k: sorted(v) for k, v in apps.items()}

def in_return_window(pid, fixture_date, apps):
    """True if this fixture falls within a player's first 3 games back
    after a 28-200 day mid-season absence."""
    dates = apps.get(pid)
    if not dates:
        return False
    prior = [d for d in dates if d < fixture_date]
    if not prior:
        return False
    # walk back up to 2 games: if a long gap occurred within the last
    # 3 appearances (incl. the upcoming one), we're in the window
    seq = prior[-3:] + [fixture_date]
    for i in range(1, len(seq)):
        gap = (seq[i] - seq[i-1]).days
        if 28 <= gap <= 200:
            games_since = len(seq) - 1 - i
            return games_since <= 2
    return False

def grade_comparison(results):
    if not os.path.exists(COMPARE_LOG):
        return
    log = pd.read_csv(COMPARE_LOG, parse_dates=["date"])
    ung = log["actual_home_win"].isna()
    for i in log[ung].index:
        m = results[(results.home_team == log.at[i, "home_team"]) &
                    (results.away_team == log.at[i, "away_team"]) &
                    ((results.date - log.at[i, "date"]).abs() <= pd.Timedelta(days=3))]
        if len(m):
            r = m.iloc[-1]
            if r.home_score != r.away_score:
                log.at[i, "actual_home_win"] = int(r.home_score > r.away_score)
    log.to_csv(COMPARE_LOG, index=False)
    done = log.dropna(subset=["actual_home_win"])
    if len(done) < 1:
        return
    yb = done["actual_home_win"].astype(int)
    variants = [("A: baseline", "p_base"), ("B: return-adjusted", "p_retadj")]
    if "p_wet" in done.columns:
        variants.append(("C: weather-adjusted", "p_wet"))
    if "p_market" in done.columns:
        variants.append(("Market (bookmaker)", "p_market"))
        variants.append(("Blend (model+market)", "p_blend"))
    for name, col in variants:
        if col not in done.columns or done[col].isna().all():
            continue
        p = done[col].dropna().clip(1e-6, 1 - 1e-6)
        yb2 = done.loc[p.index, "actual_home_win"].astype(int)
        yb = yb2
        acc = ((p > .5).astype(int) == yb).mean()
        ll = -(yb * np.log(p) + (1 - yb) * np.log(1 - p)).mean()
        br = ((p - yb) ** 2).mean()
        print(f"  {name:22s} n={len(done)}  acc={acc:.3f}  logloss={ll:.4f}  brier={br:.4f}")
    diff = done[(done.p_base - done.p_retadj).abs() > 1e-9]
    print(f"  Matches where the two models actually differ: {len(diff)} of {len(done)}")

def main(round_arg=None):
    results = pd.read_csv(os.path.join(HERE, "nrl_results.csv"), parse_dates=["date"])
    lineups = load_lineups()
    lbr = attach_lineups(results, lineups)
    season = int(results.season.max())

    # figure out round number
    if round_arg:
        round_no = int(round_arg)
    else:
        draw = get_json(f"https://www.nrl.com/draw/data?competition=111&season={season}")
        round_no = draw["selectedRoundId"]
    print(f"Fetching announced team lists: {season} Round {round_no}...")
    fixtures = fetch_round_teamlists(season, round_no)
    try:
        casualties = get_casualty_list()
        print(f"Casualty ward: {len(casualties)} players listed")
    except Exception as e:
        casualties = {}
        print(f"Casualty ward unavailable ({e}) - no injury adjustments")

    try:
        refresh_stats(season)
    except Exception as e:
        print(f"Stats refresh skipped ({e})")
    # ---- train hybrid on all completed matches with lineups
    df, X_elo, y, mask, _ = build_features(results)
    X_pl = build_player_features(results, lbr)
    X = pd.concat([X_elo.reset_index(drop=True), X_pl], axis=1)
    stats_feat, stats_rt = build_stats_feature(results)
    X["stats_strength_diff"] = stats_feat
    ok = X.notna().all(axis=1).values
    tr = mask.values & ok & (df.season >= 2021)
    cols = list(X.columns)
    sc = StandardScaler().fit(X.loc[tr, cols])
    clf = LogisticRegression(C=0.1, max_iter=2000).fit(sc.transform(X.loc[tr, cols]), y[tr])
    # Model C: same features + wet flag + wet leveller interaction
    wet_hist = add_wet_column(results)
    Xc = X.copy()
    Xc["wet"] = wet_hist
    Xc["wet_x_elo"] = wet_hist * (Xc["elo_prob"] - 0.5)
    cols_c = cols + ["wet", "wet_x_elo"]
    sc_c = StandardScaler().fit(Xc.loc[tr, cols_c])
    clf_c = LogisticRegression(C=0.1, max_iter=2000).fit(sc_c.transform(Xc.loc[tr, cols_c]), y[tr])

    # ---- current player ratings fit on everything to date
    pid_index, rows = {}, []
    for lu, row in zip(lbr, results.itertuples()):
        if lu is None or row.home_score == row.away_score:
            continue
        hp = [p["player_id"] for p in lu["players"] if p["side"] == "home"]
        ap = [p["player_id"] for p in lu["players"] if p["side"] == "away"]
        for p in hp + ap:
            pid_index.setdefault(p, len(pid_index))
        rows.append((hp, ap, 1 if row.home_score > row.away_score else 0))
    ratings, _ = fit_player_ratings(rows, len(pid_index), pid_index)
    name_index = build_name_index(lineups, results)

    # last fielded 17 & spine per team
    last17, last_spine, spine_run = {}, {}, {}
    for lu, row in zip(lbr, results.itertuples()):
        if lu is None:
            continue
        for side, team in (("home", row.home_team), ("away", row.away_team)):
            ids = {p["player_id"] for p in lu["players"] if p["side"] == side}
            sp = frozenset(p["player_id"] for p in lu["players"]
                           if p["side"] == side and p["position"] in SPINE)
            spine_run[team] = spine_run.get(team, 0) + 1 if last_spine.get(team) == sp else 1
            last17[team], last_spine[team] = ids, sp

    apps = build_appearances(lineups)
    print("Running model comparison record:")
    grade_comparison(results)
    comp_rows = []
    print()
    for fx in fixtures:
        h, a = fx["home"], fx["away"]
        sides = {}
        notes = []
        for key, team in (("home", h), ("away", a)):
            named = fx["teams"].get(key, [])
            ids, unmatched, spine_ids = [], [], set()
            factors = {}
            nrl_ids = [t[3] for t in named if t[3]]
            for first, lastn, pos, _nrlid in named:
                pos = {"Five-Eighth": "Five-eighth"}.get(pos, pos)
                pid = match_player(first, lastn, team, name_index)
                status, factor, rec = injury_status(team, first, lastn, round_no, casualties)
                if status:
                    notes.append(f"{team}: {first} {lastn} ({rec['injury']}, "
                                 f"listed return {rec['expected_return']}) - "
                                 f"{'IN DOUBT' if status == 'in_doubt' else 'first game back'}, "
                                 f"rating x{factor}")
                if pid is None:
                    unmatched.append(f"{first} {lastn}")
                else:
                    ids.append(pid)
                    factors[pid] = factor
                    if pos in SPINE:
                        spine_ids.add(pid)
            if len(ids) < 13:   # fallback: last week's 17
                ids = list(last17.get(team, []))
                spine_ids = set(last_spine.get(team, set()))
                notes.append(f"{team}: no usable team list, using last week's 17")
            elif unmatched:
                notes.append(f"{team}: {len(unmatched)} debutant/unmatched "
                             f"(rated league-average): {', '.join(unmatched)}")
            sides[key] = dict(ids=ids, spine=frozenset(spine_ids), nrl_ids=nrl_ids,
                              factors=factors if len(ids) >= 13 else {})

        fixture_date = pd.to_datetime(fx["kickoff"]).tz_localize(None) if fx.get("kickoff") \
            else results.date.max() + pd.Timedelta(days=3)
        def eff(pid, fac, ret_adj):
            r = ratings[pid_index[pid]]
            if r <= 0:
                return r
            f = fac.get(pid, 1.0)
            if ret_adj and in_return_window(pid, fixture_date, apps):
                f = min(f, RETURN_FACTOR)
            return r * f
        strength = {k: sum(eff(p, v["factors"], False) for p in v["ids"] if p in pid_index)
                    for k, v in sides.items()}
        s_stats = {k: sum(stats_rt.get(pid, 0.0) for pid in v["nrl_ids"]) for k, v in sides.items()}
        strength_b = {k: sum(eff(p, v["factors"], True) for p in v["ids"] if p in pid_index)
                      for k, v in sides.items()}
        ret = {k: len(set(v["ids"]) & last17.get(t, set())) / 17.0
               for (k, v), t in zip(sides.items(), (h, a))}
        cont = {k: (spine_run.get(t, 0) + 1 if v["spine"] == last_spine.get(t) else 1)
                for (k, v), t in zip(sides.items(), (h, a))}

        # Elo features via phantom fixture
        phantom = results.iloc[-1:].copy()
        phantom["date"] = results.date.max() + pd.Timedelta(days=3)
        phantom["home_team"], phantom["away_team"] = h, a
        phantom["home_score"], phantom["away_score"] = 0, 0
        _, Xall, *_ = build_features(pd.concat([results, phantom], ignore_index=True))
        feat = Xall.iloc[[-1]].copy()
        feat["player_strength_diff"] = strength["home"] - strength["away"]
        feat["stats_strength_diff"] = s_stats["home"] - s_stats["away"]
        feat["spine_cont_diff"] = min(cont["home"], 10) - min(cont["away"], 10)
        feat["retention_diff"] = ret["home"] - ret["away"]

        p = clf.predict_proba(sc.transform(feat[cols]))[0, 1]
        city = nrl_city_to_key(fx.get("venue_city"), h)
        mm = forecast_rain(city, fixture_date)
        wet_flag = 1 if (mm is not None and mm >= WET_MM) else 0
        feat_c = feat.copy()
        feat_c["wet"] = wet_flag
        feat_c["wet_x_elo"] = wet_flag * (feat_c["elo_prob"] - 0.5)
        p_c = clf_c.predict_proba(sc_c.transform(feat_c[cols_c]))[0, 1]
        feat_b = feat.copy()
        feat_b["player_strength_diff"] = strength_b["home"] - strength_b["away"]
        p_b = clf.predict_proba(sc.transform(feat_b[cols]))[0, 1]
        p_mkt = p_blend = None
        try:
            ho, ao = float(fx.get("home_odds") or 0), float(fx.get("away_odds") or 0)
            if ho > 1 and ao > 1:
                ih, ia = 1/ho, 1/ao
                p_mkt = ih / (ih + ia)
                lg = lambda q: np.log(q/(1-q))
                p_blend = 1/(1+np.exp(-0.5*(lg(p) + lg(p_mkt))))
        except (TypeError, ValueError):
            pass
        fav, pf = (h, p) if p > 0.5 else (a, 1 - p)
        status = fx["state"]
        line = f"{h} v {a}: {fav} ({pf:.0%})"
        if abs(p - p_b) > 0.001:
            line += f"  [return-adjusted variant: home {p_b:.0%}]"
        if wet_flag:
            line += f"  [WET forecast {mm:.0f}mm - weather variant: home {p_c:.0%}]"
        if fx["state"] in ("Upcoming", "Pre"):
            fh = os.path.join(HERE, "forecast_history.csv")
            pd.DataFrame([dict(run_time=pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
                date=fixture_date.normalize(), home_team=h, away_team=a,
                p_base=round(p, 4), p_market=round(p_mkt, 4) if p_mkt is not None else np.nan,
                wet=wet_flag, notes=len(notes))]).to_csv(
                fh, mode="a", header=not os.path.exists(fh), index=False)
        if p_mkt is not None and fx["state"] in ("Upcoming", "Pre"):
            edge = p - p_mkt
            line += f"\n    market {p_mkt:.0%} home | blend {p_blend:.0%} home"
            if abs(edge) > 0.06:
                side = h if edge > 0 else a
                line += f"  >>> EDGE: model likes {side} ({abs(edge)*100:.0f} pts vs market)"
        if fx["state"] in ("Upcoming", "Pre"):
            comp_rows.append(dict(run_date=pd.Timestamp.now().strftime("%Y-%m-%d"),
                                  date=fixture_date.normalize(), home_team=h, away_team=a,
                                  p_base=round(p, 4), p_retadj=round(p_b, 4),
                                  p_wet=round(p_c, 4), forecast_rain_mm=mm,
                                  p_market=round(p_mkt, 4) if p_mkt is not None else np.nan,
                                  p_blend=round(p_blend, 4) if p_blend is not None else np.nan,
                                  actual_home_win=np.nan))
        if status in ("FullTime", "PostGame") and fx.get("score"):
            line += f"   [PLAYED: {fx['score'][0]}-{fx['score'][1]}]"
        elif status not in ("Upcoming", "Pre"):
            line += f"   [{status} - in progress]"
        print(line)
        print(f"    player strength diff {strength['home']-strength['away']:+.2f} | "
              f"retention {ret['home']:.0%} v {ret['away']:.0%} | "
              f"spine run {cont['home']} v {cont['away']}")
        for n in notes:
            print(f"    note: {n}")
        print()

    if comp_rows:
        new = pd.DataFrame(comp_rows)
        n_before = 0
        if os.path.exists(COMPARE_LOG):
            old = pd.read_csv(COMPARE_LOG, parse_dates=["date"])
            n_before = len(old)
            key = ["date", "home_team", "away_team"]
            new = new[~new.set_index(key).index.isin(old.set_index(key).index)]
            new = pd.concat([old, new], ignore_index=True)
        appended = len(new) - n_before
        new.to_csv(COMPARE_LOG, index=False)
        if appended > 0:
            print(f"Logged {appended} new fixtures to model comparison trial.")
        else:
            print("All fixtures already logged this week (first predictions kept).")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
