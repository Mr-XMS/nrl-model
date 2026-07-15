"""
Stats-informed player ratings
=============================
Stage 1: learn each stat's win value from team-level stat differentials
         (ridge on match margin).
Stage 2: walk-forward shrunken per-player average stat value using strictly
         prior games. Team stats-strength = sum over the fielded 17.

Provides:
  refresh_stats(season)            top up player_stats.jsonl for new matches
  build_stats_feature(results)     stats_strength_diff aligned to results rows
  current_ratings()                {nrl_player_id: rating} using all data
"""
import json, os
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
STATS = os.path.join(HERE, "player_stats.jsonl")
SHRINK_GAMES = 8

NICK = {"Broncos":"brisbane-broncos","Raiders":"canberra-raiders",
 "Bulldogs":"canterbury-bankstown-bulldogs","Sharks":"cronulla-sutherland-sharks",
 "Dolphins":"dolphins","Titans":"gold-coast-titans","Sea Eagles":"manly-warringah-sea-eagles",
 "Storm":"melbourne-storm","Knights":"newcastle-knights","Cowboys":"north-queensland-cowboys",
 "Eels":"parramatta-eels","Panthers":"penrith-panthers","Rabbitohs":"south-sydney-rabbitohs",
 "Dragons":"st-george-illawarra-dragons","Roosters":"sydney-roosters",
 "Warriors":"warriors","Wests Tigers":"wests-tigers"}


def refresh_stats(season):
    """Scrape stats for any newly completed matches this season."""
    import scrape_stats as ss
    entries = ss.collect_match_urls(season)
    ss.scrape_stats(entries, out_path=STATS)


def _load_player_games():
    recs = [json.loads(l) for l in open(STATS)]
    recs = [r for r in recs if r.get("players") and r.get("home_score") is not None]
    rows = []
    for r in recs:
        date = pd.to_datetime(r["kickoff"]).tz_localize(None).normalize()
        for p in r["players"]:
            d = {k: v for k, v in p.items() if isinstance(v, (int, float)) and v is not None}
            d.update(match_url=r["url"], date=date, season=r["season"], side=p["side"],
                     player_id=p["playerId"],
                     team=NICK[r["home"] if p["side"] == "home" else r["away"]],
                     margin=(r["home_score"] - r["away_score"]) * (1 if p["side"] == "home" else -1))
            rows.append(d)
    PG = pd.DataFrame(rows).fillna(0)
    stat_cols = [c for c in PG.columns if c not in
                 ("match_url","date","season","side","player_id","team","margin","number","playerId")
                 and PG[c].dtype != object and "Rate" not in c and "Percent" not in c]
    return PG, stat_cols


def _stage1_weights(PG, stat_cols, upto_season=None):
    T = PG.groupby(["match_url","side"])[stat_cols + ["margin"]].sum().reset_index()
    Tp = T.pivot(index="match_url", columns="side")
    diff = pd.DataFrame({c: Tp[(c,"home")] - Tp[(c,"away")] for c in stat_cols})
    diff["margin"] = Tp[("margin","home")]
    diff["season"] = PG.groupby("match_url").season.first()
    est = diff if upto_season is None else diff[diff.season <= upto_season]
    sc = StandardScaler().fit(est[stat_cols])
    ridge = Ridge(alpha=50).fit(sc.transform(est[stat_cols]), est.margin)
    return sc, ridge.coef_


def _walk_forward(PG, stat_cols, sc, w):
    PG = PG.sort_values("date").copy()
    PG["value"] = sc.transform(PG[stat_cols]) @ w / 17.0
    psum, pn, prior = {}, {}, []
    for r in PG.itertuples():
        s, n = psum.get(r.player_id, 0.0), pn.get(r.player_id, 0)
        prior.append(s / (n + SHRINK_GAMES))
        psum[r.player_id] = s + r.value
        pn[r.player_id] = n + 1
    PG["rating_stats"] = prior
    return PG, psum, pn


def build_stats_feature(results):
    """Return (feature array aligned to results, {player_id: current rating})."""
    PG, stat_cols = _load_player_games()
    sc, w = _stage1_weights(PG, stat_cols)
    PG, psum, pn = _walk_forward(PG, stat_cols, sc, w)

    S = PG.groupby(["match_url","side"]).rating_stats.sum().unstack()
    S["diff"] = S["home"] - S["away"]
    S = S.join(PG.groupby("match_url")[["date"]].first())
    S["home_team"] = PG[PG.side=="home"].groupby("match_url").team.first()
    S["away_team"] = PG[PG.side=="away"].groupby("match_url").team.first()
    smap = {}
    for r in S.itertuples():
        smap[(r.date.strftime("%Y%m%d"), r.home_team, r.away_team)] = r.diff

    feat = np.full(len(results), np.nan)
    for i, r in enumerate(results.itertuples()):
        for dd in (0, 1, -1):
            k = ((r.date + pd.Timedelta(days=dd)).strftime("%Y%m%d"), r.home_team, r.away_team)
            if k in smap:
                feat[i] = smap[k]
                break
    ratings = {pid: psum[pid] / (pn[pid] + SHRINK_GAMES) for pid in psum}
    return feat, ratings


def positional_means():
    """Mean stats rating by position - for centring individual-level uses.
    (Team-level features are unaffected: identical positional structure
    means centring cancels in home-away differences.)"""
    import json
    PG, stat_cols = _load_player_games()
    sc, w = _stage1_weights(PG, stat_cols)
    PG, psum, pn = _walk_forward(PG, stat_cols, sc, w)
    ratings = {pid: psum[pid] / (pn[pid] + SHRINK_GAMES) for pid in psum}
    pos = {}
    recs = [json.loads(l) for l in open(STATS)]
    for r in recs:
        for p in r.get("players", []):
            if p.get("position"):
                pos.setdefault(p["playerId"], []).append(p["position"])
    from collections import Counter
    modal = {pid: Counter(v).most_common(1)[0][0] for pid, v in pos.items()}
    bypos = {}
    for pid, rt in ratings.items():
        if pn.get(pid, 0) >= 5 and pid in modal:
            bypos.setdefault(modal[pid], []).append(rt)
    import numpy as np
    means = {k: float(np.mean(v)) for k, v in bypos.items() if len(v) >= 10}
    return means, modal
