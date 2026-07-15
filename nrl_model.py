"""
NRL match outcome prediction model
==================================
Pipeline:
  1. Elo rating system (margin-of-victory adjusted, season regression)
  2. Feature engineering (form, attack/defence, rest, head-to-head)
  3. Gradient boosting + logistic regression classifiers
  4. Strict time-ordered evaluation (train on past, test on future)

Usage:
  python nrl_model.py                # train + evaluate
  python nrl_model.py predict HOME AWAY   # predict an upcoming match
"""
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss

import os as _os
DATA = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "nrl_results.csv")

# ---------------------------------------------------------------- Elo system
ELO_START = 1500.0
ELO_K = 32.0
ELO_HOME_ADV = 55.0        # Elo points of home advantage (fitted ~ NRL historical)
SEASON_REGRESS = 0.30      # regress 30% toward mean between seasons

def mov_multiplier(margin, elo_diff):
    """Margin-of-victory multiplier (FiveThirtyEight-style) to damp blowouts."""
    return np.log(abs(margin) + 1) * (2.2 / (elo_diff * 0.001 + 2.2))

def expected(home_elo, away_elo):
    return 1.0 / (1.0 + 10 ** (-((home_elo + ELO_HOME_ADV) - away_elo) / 400.0))

# ------------------------------------------------------- feature engineering
def build_features(df):
    df = df.sort_values("date").reset_index(drop=True)
    elo = {}
    last_season = {}
    # rolling stores per team
    hist = {}          # list of dicts per team: {date, pf, pa, win}
    h2h = {}           # (a,b) sorted key -> list of home-team-win results from a's perspective

    feats = []
    for i, row in df.iterrows():
        h, a = row.home_team, row.away_team
        # season regression when a team first appears in a new season
        for t in (h, a):
            if t not in elo:
                elo[t] = ELO_START
                hist[t] = []
                last_season[t] = row.season
            elif last_season[t] != row.season:
                elo[t] = ELO_START + (1 - SEASON_REGRESS) * (elo[t] - ELO_START)
                last_season[t] = row.season

        def form(team, n=5):
            g = hist[team][-n:]
            if not g:
                return dict(wr=0.5, pf=20.0, pa=20.0, n=0)
            return dict(wr=np.mean([x["win"] for x in g]),
                        pf=np.mean([x["pf"] for x in g]),
                        pa=np.mean([x["pa"] for x in g]),
                        n=len(g))

        def rest(team):
            g = hist[team]
            if not g:
                return 7.0
            return min((row.date - g[-1]["date"]).days, 30)

        fh, fa = form(h), form(a)
        key = tuple(sorted((h, a)))
        h2h_list = h2h.get(key, [])
        # h2h from home team's perspective, last 6 meetings
        h2h_recent = [r if hw == h else 1 - r for r, hw in h2h_list[-6:]]
        h2h_rate = np.mean(h2h_recent) if h2h_recent else 0.5

        p_elo = expected(elo[h], elo[a])
        feats.append(dict(
            elo_diff=elo[h] - elo[a],
            elo_prob=p_elo,
            form_diff=fh["wr"] - fa["wr"],
            attack_diff=fh["pf"] - fa["pf"],
            defence_diff=fa["pa"] - fh["pa"],     # positive = home defence better matchup
            rest_diff=rest(h) - rest(a),
            h2h_rate=h2h_rate,
            home_games_played=min(fh["n"], 5),
            away_games_played=min(fa["n"], 5),
            season_progress=(row.date.dayofyear - 60) / 250.0,
        ))

        # ---- update state after the match
        home_win = 1.0 if row.home_score > row.away_score else (0.5 if row.home_score == row.away_score else 0.0)
        margin = row.home_score - row.away_score
        k = ELO_K * (mov_multiplier(margin, elo[h] - elo[a]) if margin != 0 else 1.0)
        delta = k * (home_win - p_elo)
        elo[h] += delta
        elo[a] -= delta
        hist[h].append(dict(date=row.date, pf=row.home_score, pa=row.away_score, win=home_win))
        hist[a].append(dict(date=row.date, pf=row.away_score, pa=row.home_score, win=1 - home_win))
        h2h.setdefault(key, []).append((home_win, h))

    X = pd.DataFrame(feats)
    y = (df.home_score > df.away_score).astype(int)
    # drop draws from training/eval target ambiguity (rare in NRL, golden point era)
    mask = df.home_score != df.away_score
    return df, X, y, mask, elo

# ---------------------------------------------------------------- evaluation
def evaluate():
    raw = pd.read_csv(DATA, parse_dates=["date"])
    df, X, y, mask, final_elo = build_features(raw)

    # burn in: skip first 2 seasons while Elo/form stabilise
    burn = df.season >= 2011
    train_idx = mask & burn & (df.season <= 2023)
    test_idx = mask & (df.season >= 2024)

    Xtr, ytr = X[train_idx], y[train_idx]
    Xte, yte = X[test_idx], y[test_idx]

    scaler = StandardScaler().fit(Xtr)
    lr = LogisticRegression(C=0.1, max_iter=2000).fit(scaler.transform(Xtr), ytr)
    gb = HistGradientBoostingClassifier(
        max_depth=3, learning_rate=0.05, max_iter=300,
        l2_regularization=1.0, random_state=42).fit(Xtr, ytr)

    p_lr = lr.predict_proba(scaler.transform(Xte))[:, 1]
    p_gb = gb.predict_proba(Xte)[:, 1]
    p_ens = 0.5 * p_lr + 0.5 * p_gb
    p_elo = X[test_idx]["elo_prob"].values

    def report(name, p):
        print(f"{name:22s} acc={accuracy_score(yte, p > .5):.3f}  "
              f"logloss={log_loss(yte, p):.4f}  brier={brier_score_loss(yte, p):.4f}")

    n = len(yte)
    print(f"\nTest set: {n} matches, seasons 2024-2026")
    print(f"{'Baseline (home wins)':22s} acc={yte.mean():.3f}")
    report("Elo only", p_elo)
    report("Logistic regression", p_lr)
    report("Gradient boosting", p_gb)
    report("Ensemble (LR+GB)", p_ens)

    print("\nFeature importance (GB, permutation on test):")
    from sklearn.inspection import permutation_importance
    imp = permutation_importance(gb, Xte, yte, n_repeats=10, random_state=0)
    for f, v in sorted(zip(X.columns, imp.importances_mean), key=lambda x: -x[1]):
        print(f"  {f:22s} {v:+.4f}")

    print("\nCurrent Elo ratings (top of table):")
    for t, e in sorted(final_elo.items(), key=lambda x: -x[1])[:17]:
        print(f"  {t:35s} {e:7.1f}")

    return df, X, y, mask, scaler, lr, gb

def predict(home, away):
    raw = pd.read_csv(DATA, parse_dates=["date"])
    df, X, y, mask, final_elo = build_features(raw)
    burn = df.season >= 2011
    tr = mask & burn
    scaler = StandardScaler().fit(X[tr])
    lr = LogisticRegression(C=0.1, max_iter=2000).fit(scaler.transform(X[tr]), y[tr])
    gb = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=300,
                                        l2_regularization=1.0, random_state=42).fit(X[tr], y[tr])
    # rebuild latest state by appending a phantom fixture row and recomputing features
    fixture = raw.iloc[-1:].copy()
    fixture["date"] = raw["date"].max() + pd.Timedelta(days=7)
    fixture["home_team"], fixture["away_team"] = home, away
    fixture["home_score"], fixture["away_score"] = 0, 0
    df2, X2, *_ = build_features(pd.concat([raw, fixture], ignore_index=True))
    x = X2.iloc[[-1]]
    p = 0.5 * lr.predict_proba(scaler.transform(x))[0, 1] + 0.5 * gb.predict_proba(x)[0, 1]
    print(f"\n{home} (home) vs {away} (away)")
    print(f"  P({home} win) = {p:.1%}")
    print(f"  P({away} win) = {1-p:.1%}")
    print(f"  Elo: {final_elo.get(home, 1500):.0f} vs {final_elo.get(away, 1500):.0f}")

if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == "predict":
        predict(sys.argv[2], sys.argv[3])
    else:
        evaluate()
