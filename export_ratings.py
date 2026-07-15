"""Export player_ratings.csv for the dashboard player browser."""
import os, sys, json
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from nrl_player_model import load_lineups, attach_lineups, fit_player_ratings
from stats_ratings import build_stats_feature, positional_means

results = pd.read_csv(os.path.join(HERE, "nrl_results.csv"), parse_dates=["date"])
lineups = load_lineups(); lbr = attach_lineups(results, lineups)
pid_index, rows, names, games, team, lastd = {}, [], {}, {}, {}, {}
for lu, row in zip(lbr, results.itertuples()):
    if lu is None: continue
    for p in lu["players"]:
        pid_index.setdefault(p["player_id"], len(pid_index))
        names[p["player_id"]] = p["player"].title()
        games[p["player_id"]] = games.get(p["player_id"], 0) + 1
        team[p["player_id"]] = row.home_team if p["side"]=="home" else row.away_team
        lastd[p["player_id"]] = row.date
    if row.home_score != row.away_score:
        rows.append(([q["player_id"] for q in lu["players"] if q["side"]=="home"],
                     [q["player_id"] for q in lu["players"] if q["side"]=="away"],
                     1 if row.home_score > row.away_score else 0))
ratings, _ = fit_player_ratings(rows, len(pid_index), pid_index)
_, stats_rt = build_stats_feature(results)
pos_means, modal_pos = positional_means()

# join stats identity by name
recs = [json.loads(l) for l in open(os.path.join(HERE, "player_stats.jsonl"))]
by_name = {}
tries = {}
for r in recs:
    for p in r.get("players", []):
        by_name[p.get("name","").lower()] = p["playerId"]
        t = tries.setdefault(p["playerId"], [0,0]); t[0]+= p.get("tries",0) or 0; t[1]+=1
out = []
allr = ratings
for pid, i in pid_index.items():
    nm = names[pid]
    nrl = by_name.get(nm.lower())
    tr = tries.get(nrl, [0,0])
    out.append(dict(player=nm, team=team[pid], games=games[pid],
        last_seen=str(lastd[pid].date()),
        presence_rating=round(float(ratings[i]),3),
        presence_pct=int((allr < ratings[i]).mean()*100),
        position=modal_pos.get(nrl, ""),
        stats_rating=round(float(stats_rt.get(nrl,0)),3) if nrl else None,
        stats_vs_position=round(float(stats_rt.get(nrl,0) -
            pos_means.get(modal_pos.get(nrl,""),0.0)),3) if nrl else None,
        tries_per_game=round(tr[0]/tr[1],3) if tr[1] else None))
pd.DataFrame(out).sort_values("presence_rating", ascending=False).to_csv(
    os.path.join(HERE, "player_ratings.csv"), index=False)
print(len(out), "players -> player_ratings.csv")
