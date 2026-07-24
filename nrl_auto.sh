#!/bin/bash
# NRL model automation dispatcher - decides what to run based on the day.
# All output is appended to automation.log in the model folder.
cd "$(dirname "$0")"
LOG="automation.log"
DAY=$(date +%u)   # 1=Mon ... 7=Sun
if [ -n "$FORCE_DAY" ]; then DAY="$FORCE_DAY"; echo "forced day: $DAY" >> "$LOG"; fi
echo "" >> "$LOG"
echo "===== $(date '+%Y-%m-%d %H:%M') (day $DAY) =====" >> "$LOG"

case $DAY in
  1)  # Monday: results + grading
      python3 update_nrl.py >> "$LOG" 2>&1
      python3 scrape_stats.py >> "$LOG" 2>&1
      python3 paper_bets.py >> "$LOG" 2>&1
      python3 notify.py grades >> "$LOG" 2>&1
      ;;
  2)  # Tuesday evening: full forecast + SGM ledger (team lists are out)
      python3 update_nrl.py >> "$LOG" 2>&1
      python3 predict_teamlists.py >> "$LOG" 2>&1
      python3 sgm_simulator.py ledger >> "$LOG" 2>&1
      python3 fill_ledger_odds.py >> "$LOG" 2>&1
      python3 arb_scanner.py >> "$LOG" 2>&1
      python3 paper_bets.py place >> "$LOG" 2>&1
      python3 scenario_analysis.py >> "$LOG" 2>&1
      python3 export_ratings.py >> "$LOG" 2>&1
      python3 notify.py picks >> "$LOG" 2>&1
      ;;
  4)  # Thursday: pre-round odds scan
      python3 arb_scanner.py >> "$LOG" 2>&1
      ;;
  5|6|7)  # Fri/Sat/Sun late morning: game-day scan + fresh lineup check
      python3 arb_scanner.py >> "$LOG" 2>&1
      python3 predict_teamlists.py >> "$LOG" 2>&1
      ;;
 99) # research: one-off studies
      python3 origin_study.py >> "$LOG" 2>&1
      ;;
  *)
      echo "no scheduled tasks today" >> "$LOG"
      ;;
esac
echo "===== done =====" >> "$LOG"
