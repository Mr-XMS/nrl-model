"""
finals_rounds.py - give finals fixtures their own round identity.
=================================================================
The NRL draw API reports finals fixtures with roundTitle "Round 27", the
same label as the last regular-season round. Nothing downstream
distinguished them, so three weekends of football collapsed into one
bucket:

  * predictions.csv held 10 rows under "Round 27" spanning 3 Sep to 20 Sep
  * publish_rounds.py takes the LAST fixture date in a round to compute the
    unlock, so Round 27's page was waiting on a semi-final two weeks after
    the round itself finished, and never published
  * finals week 1 had no round of its own, so it could not be graded or
    published separately

This module derives the real finals round from the fixture date. The NRL
finals series runs on a fixed shape: four matches in week 1, two semi
finals, two preliminary finals, one grand final, on consecutive weekends
after the last regular round.

Labels are stable strings so they can be compared, sorted and turned into
page slugs. Ordering numbers continue past the regular season (28, 29, 30,
31) so existing "round number" logic keeps working without special cases.
"""
import re

import pandas as pd

# Ordering number -> (label, slug). The numbers extend the regular season
# so any code doing int(round) keeps sorting correctly.
FINALS = {
    28: ("Finals Week 1", "finals-week-1"),
    29: ("Semi Finals", "semi-finals"),
    30: ("Preliminary Finals", "preliminary-finals"),
    31: ("Grand Final", "grand-final"),
}

# 2026 finals series. Each entry is the inclusive date window for that week.
# Windows are generous (Thu-Mon) to absorb scheduling moves.
SERIES = {
    2026: [
        (28, "2026-09-10", "2026-09-14"),
        (29, "2026-09-17", "2026-09-21"),
        (30, "2026-09-24", "2026-09-28"),
        (31, "2026-10-01", "2026-10-05"),
    ],
}


def finals_number(date, season):
    """Ordering number for a finals fixture, or None if regular season."""
    d = pd.Timestamp(date).normalize()
    for num, start, end in SERIES.get(int(season), []):
        if pd.Timestamp(start) <= d <= pd.Timestamp(end):
            return num
    return None


def round_label(raw_label, date, season):
    """Authoritative round label for a fixture.

    The source label is trusted for the regular season and overridden for
    finals, because the source reuses the last regular round's title.
    """
    n = finals_number(date, season)
    if n is None:
        return str(raw_label)
    return FINALS[n][0]


def round_order(raw_label, date, season):
    """Sortable round number, continuing past the regular season."""
    n = finals_number(date, season)
    if n is not None:
        return n
    m = re.search(r"(\d+)", str(raw_label))
    return int(m.group(1)) if m else None


def round_slug(raw_label, date, season):
    """URL slug: 'round-26' for regular rounds, 'finals-week-1' for finals."""
    n = finals_number(date, season)
    if n is not None:
        return FINALS[n][1]
    m = re.search(r"(\d+)", str(raw_label))
    return f"round-{m.group(1)}" if m else "round"


def relabel(df, label_col="round", date_col="date", season_col="season",
            season_default=2026):
    """Rewrite a dataframe's round column in place and return it.

    Safe to run repeatedly: a fixture already carrying its finals label
    resolves to the same label again.
    """
    if label_col not in df.columns:
        return df
    dates = pd.to_datetime(df[date_col])
    seasons = (df[season_col] if season_col in df.columns
               else pd.Series(season_default, index=df.index))
    df[label_col] = [round_label(l, d, s) for l, d, s
                     in zip(df[label_col], dates, seasons)]
    return df


if __name__ == "__main__":
    for d in ["2026-09-06", "2026-09-11", "2026-09-19", "2026-09-26",
              "2026-10-04"]:
        print(f"{d}  {round_label('Round 27', d, 2026):20s} "
              f"order={round_order('Round 27', d, 2026)}  "
              f"slug={round_slug('Round 27', d, 2026)}")
