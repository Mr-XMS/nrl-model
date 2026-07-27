# Offseason 2026-27: Pre-Registered Candidates

Written 2026-07-26, BEFORE season-end results are known. Each candidate states
its success criterion now, so October's decisions are judged against rules set
in advance - the same discipline applied to Models B and C.

Season-end adjudication reads: the trial log (model_comparison.csv), the paper
P&L (bets_ledger.csv), the SGM ledger, odds_history, and the Verdict tab CIs.

---

## Standing decisions due in October (already-running trials)

1. **Model B (returning-player ramp)** - PROMOTE if B's log-loss beats A with the
   90% CI excluding zero over the full trial; otherwise KILL. No middle ground.
2. **Model C (wet leveller)** - same rule vs A; additionally re-grade C against
   OBSERVED rainfall (weather.csv) not forecast, and report both. If C-observed
   clearly beats C-forecast, the failure was the forecast, not the thesis.
3. **Blend weight** - if Blend beats Market with CI excluding zero, keep 50/50;
   if the biggest-divergence decile of edges shows negative ROI, adopt candidate #3.
4. **Paper betting** - real money is considered ONLY if: value-leg ROI lower CI > 0,
   paper P&L positive over 15+ rounds, AND simulator calibration holds within
   3 pts in the 0.35-0.7 buckets. All three, or it stays paper in 2027.

## New builds (in priority order)

### 1. Returning-cohort feature
- Hypothesis: teams regaining multiple quality players in one week outperform
  the model's per-player discounts (the Reynolds-Mam-Piakura game).
- Data: returning_home/away columns in forecast_history (logging live from Jul 26).
- Build: count x mean-rating of returning players, both sides, as a feature.
- KILL IF: walk-forward log-loss improvement < 0.002, or coefficient unstable
  across seasons. SHIP IF: >= 0.002 improvement and correct sign every season.

### 2. Graded rain variable
- Replace binary 5mm flag with mm-scaled effect; train on observed archive rain.
- KILL IF: no improvement over binary flag out-of-sample. (The Panthers Rd20
  knife-edge game is the motivating exhibit.)

### 3. Divergence-dependent blend
- Blend weight shifts toward market as |model - market| grows.
- Fit the weight curve on 2022-25 trial-style data, test on 2026.
- KILL IF: fixed 50/50 blend log-loss is not beaten out-of-sample.

### 4. Team pace/style rates for the totals engine
- Team-specific attack/defence try rates (recency-weighted) in the Poisson sim.
- SHIP IF: totals Brier beats base rate with CI excluding zero (currently fails).
  This gates any SGM totals-leg strategy.

### 5. Player value rebuild (opponent-adjusted, creation-aware, minutes-weighted)
- Acceptance tests, pre-registered: Munster and Grant rate clearly positive vs
  position; Tedesco's decline pattern preserved; team-level feature still adds
  >= current 0.0035 log-loss; scenario swing for a withdrawn elite playmaker
  lands in the 3-6 pt range the market implies.

## Explicitly NOT being built (graveyard, with evidence)
Travel distance - rest days - consecutive away weeks - day/night splits -
ladder motivation - isotonic recalibration - Origin fatigue flag (n=427, CI
centred on zero) - Origin win/loss hangover (CI straddles zero) - new-coach
bounce (n=11, negative point estimate). Revisit only with new evidence, not
new enthusiasm.

## Process rules
- Nothing above touches the live model before the season ends (freeze).
- Every build gets a walk-forward harness test before entering any trial.
- 2027 trials run the same A/B/... structure with first-prediction-frozen logging.
