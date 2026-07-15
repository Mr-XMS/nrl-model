"""
Player scenario analysis
========================
For every injury-flagged player named in this round's team lists, show the
match win probability under three scenarios:

  FIT        plays at full rating
  NAMED      plays with the injury discount (production assumption)
  WITHDRAWN  late out, replaced by a league-average player
             (retention drops, spine continuity resets if a spine player)

Usage:  python scenario_analysis.py [round]
"""
import os, sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import predict_teamlists as pt
from nrl_model import build_features
from nrl_player_model import load_lineups, attach_lineups, build_player_features, \
    fit_player_ratings, SPINE
from injury_adjust import get_casualty_list, injury_status
from stats_ratings import build_stats_feature, positional_means


def main(round_arg=None):
    results = pd.read_csv(os.path.join(HERE, "nrl_results.csv"), parse_dates=["date"])
    lineups = load_lineups()
    lbr = attach_lineups(results, lineups)
    season = int(results.season.max())

    if round_arg:
        round_no = int(round_arg)
    else:
        import urllib.request, json
        req = urllib.request.Request(
            f"https://www.nrl.com/draw/data?competition=111&season={season}", headers=pt.UA)
        round_no = json.loads(urllib.request.urlopen(req).read())["selectedRoundId"]
    print(f"Scenario analysis: {season} Round {round_no}\n")
    fixtures = pt.fetch_round_teamlists(season, round_no)
    casualties = get_casualty_list()

    # ---- train classifier (same base as production)
    df, X_elo, y, mask, _ = build_features(results)
    X_pl = build_player_features(results, lbr)
    X = pd.concat([X_elo.reset_index(drop=True), X_pl], axis=1)
    stats_feat, stats_rt = build_stats_feature(results)
    pos_means, modal_pos = positional_means()
    def centred(nrlid):
        r = stats_rt.get(nrlid, 0.0)
        return r - pos_means.get(modal_pos.get(nrlid, ""), 0.0)
    X["stats_strength_diff"] = stats_feat
    ok = X.notna().all(axis=1).values
    tr = mask.values & ok & (df.season >= 2021)
    cols = list(X.columns)
    sc = StandardScaler().fit(X.loc[tr, cols])
    clf = LogisticRegression(C=0.1, max_iter=2000).fit(sc.transform(X.loc[tr, cols]), y[tr])

    # ---- presence ratings on all data
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
    name_index = pt.build_name_index(lineups, results)

    # last fielded 17 / spine per team
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

    FACTORS = {"in_doubt": 0.60, "returning": 0.80}

    for fx in fixtures:
        if fx["state"] not in ("Upcoming", "Pre"):
            continue
        h, a = fx["home"], fx["away"]
        # build both sides at production baseline
        side_info = {}
        flagged = []
        for key, team in (("home", h), ("away", a)):
            players = []
            for first, lastn, pos, nrlid in fx["teams"].get(key, []):
                pos = {"Five-Eighth": "Five-eighth"}.get(pos, pos)
                pid = pt.match_player(first, lastn, team, name_index)
                status, factor, rec = injury_status(team, first, lastn, round_no, casualties)
                players.append(dict(pid=pid, nrlid=nrlid, pos=pos, factor=factor,
                                    name=f"{first} {lastn}", status=status, team=team, key=key))
                if status:
                    flagged.append(players[-1])
            side_info[key] = players
        if not flagged:
            continue

        def predict(overrides={}):
            """overrides: {player name: 'fit'|'out'}"""
            s_pres, s_stat, ret, cont = {}, {}, {}, {}
            for key, team in (("home", h), ("away", a)):
                pres = stat = 0.0
                ids, spine_ids = set(), set()
                for p in side_info[key]:
                    mode = overrides.get(p["name"])
                    if mode == "out":
                        continue                      # replaced by league-average (0)
                    f = 1.0 if mode == "fit" else p["factor"]
                    if p["pid"] is not None:
                        r = ratings[pid_index[p["pid"]]] if p["pid"] in pid_index else 0.0
                        pres += r * f if r > 0 else r
                        ids.add(p["pid"])
                        if p["pos"] in SPINE:
                            spine_ids.add(p["pid"])
                    if p["nrlid"]:
                        rs = centred(p["nrlid"])
                        stat += rs * f if rs > 0 else rs
                s_pres[key], s_stat[key] = pres, stat
                ret[key] = len(ids & last17.get(team, set())) / 17.0
                cont[key] = (spine_run.get(team, 0) + 1
                             if frozenset(spine_ids) == last_spine.get(team) else 1)
            phantom = results.iloc[-1:].copy()
            phantom["date"] = results.date.max() + pd.Timedelta(days=3)
            phantom["home_team"], phantom["away_team"] = h, a
            phantom["home_score"], phantom["away_score"] = 0, 0
            _, Xall, *_ = build_features(pd.concat([results, phantom], ignore_index=True))
            feat = Xall.iloc[[-1]].copy()
            feat["player_strength_diff"] = s_pres["home"] - s_pres["away"]
            feat["retention_diff"] = ret["home"] - ret["away"]
            feat["spine_cont_diff"] = min(cont["home"], 10) - min(cont["away"], 10)
            feat["stats_strength_diff"] = s_stat["home"] - s_stat["away"]
            return clf.predict_proba(sc.transform(feat[cols]))[0, 1]

        p0 = predict()
        print(f"{h} v {a}  (home win, production baseline: {p0:.1%})")
        csv_rows = globals().setdefault("_CSV_ROWS", [])
        for p in flagged:
            p_fit = predict({p["name"]: "fit"})
            p_out = predict({p["name"]: "out"})
            own = "home" if p["key"] == "home" else "away"
            tag = "IN DOUBT" if p["status"] == "in_doubt" else "returning"
            print(f"  {p['name']:26s} ({p['team']}, {tag})")
            print(f"      plays fully fit {p_fit:.1%} | as named {p0:.1%} | "
                  f"WITHDRAWN {p_out:.1%}   swing fit-to-out: "
                  f"{abs(p_fit - p_out)*100:.1f} pts")
            csv_rows.append(dict(home=h, away=a, player=p["name"], team=p["team"],
                status=tag, p_fit=round(p_fit,4), p_named=round(p0,4),
                p_withdrawn=round(p_out,4), swing_pts=round(abs(p_fit-p_out)*100,1)))
        print()

    rows = globals().get("_CSV_ROWS", [])
    pd.DataFrame(rows).to_csv(os.path.join(HERE, "scenarios.csv"), index=False)
    print(f"{len(rows)} scenarios -> scenarios.csv")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
