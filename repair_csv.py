"""
repair_csv.py - remove git conflict markers and duplicate fixtures.
===================================================================
A merge conflict was resolved by committing the conflicted file itself, so
predictions.csv and nrl_results.csv now contain literal '<<<<<<< HEAD',
'=======' and '>>>>>>> <sha>' lines, plus the duplicate fixture rows that
sit either side of them.

This repairs both files:

  1. drop conflict-marker lines
  2. drop duplicate fixtures on (date, home_team, away_team), keeping the
     row with the EARLIEST run_date - the original forecast, not a later
     rewrite, because the archive's claim rests on the first recording
  3. re-apply finals round labels

Run from the repo root. Writes .bak copies first.

    python3 repair_csv.py            # report
    python3 repair_csv.py --write
"""
import os
import shutil
import sys

import pandas as pd

from finals_rounds import relabel

HERE = os.path.dirname(os.path.abspath(__file__))
FILES = ["predictions.csv", "nrl_results.csv", "model_comparison.csv"]
MARKERS = ("<<<<<<<", "=======", ">>>>>>>")
KEY = ["date", "home_team", "away_team"]


def strip_markers(path):
    with open(path) as f:
        lines = f.readlines()
    clean = [l for l in lines if not l.lstrip().startswith(MARKERS)]
    return lines, clean


def main():
    write = "--write" in sys.argv
    for name in FILES:
        path = os.path.join(HERE, name)
        if not os.path.exists(path):
            continue

        lines, clean = strip_markers(path)
        removed = len(lines) - len(clean)

        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            f.writelines(clean)
        df = pd.read_csv(tmp)
        os.remove(tmp)

        n0 = len(df)
        if all(k in df.columns for k in KEY):
            sort_col = "run_date" if "run_date" in df.columns else None
            if sort_col:
                df = df.sort_values(sort_col)
            df = df.drop_duplicates(KEY, keep="first")
        dupes = n0 - len(df)

        if "round" in df.columns:
            df = relabel(df)
        df = df.sort_values("date").reset_index(drop=True)

        print(f"{name}: {removed} marker line(s), {dupes} duplicate "
              f"fixture(s), {len(df)} rows remain")
        if "round" in df.columns:
            tail = df["round"].value_counts().reindex(
                [r for r in df["round"].unique()][-4:])
            for k, v in tail.items():
                print(f"    {k}: {v}")

        if write:
            shutil.copy(path, path + ".bak")
            df.to_csv(path, index=False)

    if not write:
        print("\ndry run. re-run with --write to repair.")
    else:
        print("\nrepaired. .bak copies kept alongside each file.")


if __name__ == "__main__":
    main()
