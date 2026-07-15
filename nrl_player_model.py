"""
NRL player-based hybrid model
=============================
Adds player-level information to the team Elo model:

  1. Ridge logistic "plus-minus": each match is a row; fielded home players
     get +1, away players -1. Regularised regression learns a per-player
     win contribution. Team strength = sum over the actual 17 named.
  2. Walk-forward: player ratings are refit monthly using only prior
     matches, so every prediction is strictly out-of-sample.
  3. Cohesion features: spine continuity + share of last week's 17 retained.
  4. Hybrid classifier: Elo features + player features -> logistic regression.

Usage:
  python nrl_player_model.py            # evaluate hybrid vs Elo-only
"""
import json
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss

import os
_HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(_HERE, "nrl_results.csv")
LINEUPS = os.path.join(_HERE, "lineups.jsonl")
SPINE = {"Fullback", "Five-eighth", "Halfback", "Hooker"}


def load_lineups():
    recs = []
    with open(LINEUPS) as f:
        for line in f:
            recs.append(json.loads(line))
    return {r["match_id"]: r for r in recs if r.get("n_players", 0) >= 26}


def attach_lineups(results, lineups):
    """Match lineup records to results rows on (date, home, away)."""
    idx = {}
    for r in lineups.values():
        idx[(r["date"], r["home_team"], r["away_team"])] = r
    matched = []
    for _, row in results.iterrows():
        key = (row.date.strftime("%Y-%m-%d"), row.home_team, row.away_team)
        matched.append(idx.get(key))
    return matched


def fit_player_ratings(rows, n_players, pid_index, alpha=0.15):
    """Ridge logistic plus-minus. rows: list of (home_pids, away_pids, home_win)."""
    data, ri, ci, y = [], [], [], []
    for k, (hp, ap, hw) in enumerate(rows):
        for p in hp:
            data.append(1.0); ri.append(k); ci.append(pid_index[p])
        for p in ap:
            data.append(-1.0); ri.append(k); ci.append(pid_index[p])
        y.append(hw)
    X = sparse.csr_matrix((data, (ri, ci)), shape=(len(rows), n_players))
    lr = LogisticRegression(C=alpha, max_iter=3000, solver="lbfgs")
    lr.fit(X, y)
    return lr.coef_[0], lr.intercept_[0]   # per-player ratings, home advantage


def build_player_features(results, lineup_by_row, refit_every=30):
    """Walk-forward player-strength + cohesion features for every match."""
    n = len(results)
    pid_index, pids = {}, []
    for lu in lineup_by_row:
        if lu:
            for p in lu["players"]:
                if p["player_id"] not in pid_index:
                    pid_index[p["player_id"]] = len(pids)
                    pids.append(p["player_id"])

    ratings = np.zeros(len(pids))
    train_rows = []                      # accumulated (home_pids, away_pids, hw)
    last_17 = {}                         # team -> set of player_ids last match
    last_spine = {}                      # team -> frozenset spine pids
    spine_games = {}                     # (team, frozenset) -> consecutive count

    feats = np.full((n, 4), np.nan)      # strength_diff, spine_cont_diff, retention_diff, ha
    since_fit = 0
    fitted = False

    for i, (row, lu) in enumerate(zip(results.itertuples(), lineup_by_row)):
        if lu is None:
            continue
        hp = [p["player_id"] for p in lu["players"] if p["side"] == "home"]
        ap = [p["player_id"] for p in lu["players"] if p["side"] == "away"]
        h_spine = frozenset(p["player_id"] for p in lu["players"]
                            if p["side"] == "home" and p["position"] in SPINE)
        a_spine = frozenset(p["player_id"] for p in lu["players"]
                            if p["side"] == "away" and p["position"] in SPINE)

        # refit ratings periodically on all prior matches
        if len(train_rows) >= 150 and (not fitted or since_fit >= refit_every):
            r, ha = fit_player_ratings(train_rows, len(pids), pid_index)
            ratings = r
            fitted = True
            since_fit = 0

        if fitted:
            s_h = sum(ratings[pid_index[p]] for p in hp)
            s_a = sum(ratings[pid_index[p]] for p in ap)
            # cohesion: consecutive matches this exact spine has started together
            sc_h = spine_games.get((row.home_team, h_spine), 0)
            sc_a = spine_games.get((row.away_team, a_spine), 0)
            ret_h = len(set(hp) & last_17.get(row.home_team, set())) / 17.0
            ret_a = len(set(ap) & last_17.get(row.away_team, set())) / 17.0
            feats[i] = [s_h - s_a, min(sc_h, 10) - min(sc_a, 10), ret_h - ret_a, 1.0]

        # update state
        hw = 1 if row.home_score > row.away_score else 0
        if row.home_score != row.away_score:
            train_rows.append((hp, ap, hw))
            since_fit += 1
        for team, spine in ((row.home_team, h_spine), (row.away_team, a_spine)):
            prev = last_spine.get(team)
            spine_games[(team, spine)] = spine_games.get((team, spine), 0) + 1 \
                if prev == spine else 1
            last_spine[team] = spine
        last_17[row.home_team] = set(hp)
        last_17[row.away_team] = set(ap)

    return pd.DataFrame(feats, columns=["player_strength_diff", "spine_cont_diff",
                                        "retention_diff", "_ha"]).drop(columns="_ha")


def evaluate():
    import sys
    sys.path.insert(0, _HERE)
    from nrl_model import build_features

    results = pd.read_csv(RESULTS, parse_dates=["date"])
    lineups = load_lineups()
    lineup_by_row = attach_lineups(results, lineups)
    n_matched = sum(x is not None for x in lineup_by_row)
    print(f"Lineups matched to {n_matched}/{len(results)} matches")

    df, X_elo, y, mask, _ = build_features(results)
    X_pl = build_player_features(results, lineup_by_row)
    X = pd.concat([X_elo.reset_index(drop=True), X_pl], axis=1)

    have_pl = X_pl.notna().all(axis=1).values
    train = mask.values & have_pl & (df.season >= 2020) & (df.season <= 2023)
    test = mask.values & have_pl & (df.season >= 2024)

    def run(cols, name):
        sc = StandardScaler().fit(X.loc[train, cols])
        lr = LogisticRegression(C=0.1, max_iter=2000).fit(
            sc.transform(X.loc[train, cols]), y[train])
        p = lr.predict_proba(sc.transform(X.loc[test, cols]))[:, 1]
        print(f"{name:34s} acc={accuracy_score(y[test], p > .5):.3f}  "
              f"logloss={log_loss(y[test], p):.4f}  brier={brier_score_loss(y[test], p):.4f}")
        return p, lr, sc

    print(f"\nTrain: {train.sum()} matches (2020-23) | Test: {test.sum()} matches (2024-26)\n")
    elo_cols = list(X_elo.columns)
    pl_cols = list(X_pl.columns)
    run(elo_cols, "Team Elo model (baseline)")
    run(pl_cols, "Player model only")
    p, lr, sc = run(elo_cols + pl_cols, "Hybrid (Elo + players)")

    print("\nHybrid coefficients (standardised):")
    for c, w in sorted(zip(elo_cols + pl_cols, lr.coef_[0]), key=lambda t: -abs(t[1])):
        print(f"  {c:24s} {w:+.3f}")


if __name__ == "__main__":
    evaluate()
