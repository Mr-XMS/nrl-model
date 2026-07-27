"""Ensure weather.csv covers every graded game (observed rainfall backfill).
Runs Mondays. Idempotent: extends the archive if behind, then reports coverage
so the log always shows whether Model C's observed-rain grading data is complete."""
import os, sys
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from weather_features import add_wet_column

results = pd.read_csv(os.path.join(HERE, "nrl_results.csv"), parse_dates=["date"])
try:
    wet = add_wet_column(results)          # extends the archive for any missing dates
    print(f"Rain backfill: wet flags computed for {int(pd.Series(wet).notna().sum())} "
          f"of {len(results)} matches")
except Exception as e:
    print(f"Rain backfill WARNING: {e}")

wpath = os.path.join(HERE, "weather.csv")
if os.path.exists(wpath):
    W = pd.read_csv(wpath)
    date_col = next((c for c in ("date", "day", "dt") if c in W.columns), W.columns[0])
    dates = pd.to_datetime(W[date_col], errors="coerce")
    season = results[results.season == results.season.max()]
    print(f"Rain archive: {len(W)} rows, latest {dates.max().date()} "
          f"(latest graded game: {season.date.max().date()})")
    if dates.max() < season.date.max():
        print("Rain backfill WARNING: archive lags graded games - "
              "observed-rain grading for Model C will be incomplete")
