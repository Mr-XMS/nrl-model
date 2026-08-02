# Offseason 2026-27: Pre-Registered Candidates

Written 2026-07-26, BEFORE season-end results are known. Amended 2026-08-02 after Round 22 (Storm/Knights mega-edge split): candidates
1 and 3 refined, candidate 6 added. All criteria still fixed before season end. Each candidate states
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

### 1. Roster-churn feature (bidirectional) + form interaction
- Hypothesis: the model underprices roster churn in BOTH directions - the boost
  of quality returns (Reynolds-Mam-Piakura, Rd20) AND the damage of mass
  absences (Storm 7-out at 59% retention, Rd22, lost by 14 vs a 24-pt edge).
- Corollary: the contrarian form coefficient is right for stable rosters and
  wrong when a streak is personnel-driven (Bulldogs surge was real). Test a
  form x churn interaction: churn-era form = signal, stable-era form = noise.
- Data: returning/indoubt columns in forecast_history (live since Jul 26) +
  retention history from lineups.
- KILL IF: walk-forward log-loss improvement < 0.002 or unstable sign across
  seasons. SHIP IF: >= 0.002 and consistent sign.

### 2. Graded rain variable
- Replace binary 5mm flag with mm-scaled effect; train on observed archive rain.
- KILL IF: no improvement over binary flag out-of-sample. (The Panthers Rd20
  knife-edge game is the motivating exhibit.)

### 3. CONDITIONAL divergence shrinkage (refined Rd22)
- Live mega-edge family (>10 pts) record: 1-3. But the structure is directional:
  edges built on slow signals (ratings collapse, defence - Knights Rd22, won
  30-6) held; edges built on discounting churn/absence (Dolphins, Manly, Storm)
  are 0-3. Blanket shrinkage would have killed the Knights call.
- Build: shrink toward market as divergence grows ONLY when the edge coincides
  with low retention / high churn on the model-backed side; hold full weight
  for ratings-driven edges.
- Fit on 2022-25, test on 2026 incl. the live trial log.
- KILL IF: fixed 50/50 blend is not beaten out-of-sample.

### 4. Team pace/style rates for the totals engine
- Team-specific attack/defence try rates (recency-weighted) in the Poisson sim.
- SHIP IF: totals Brier beats base rate with CI excluding zero (currently fails).
  This gates any SGM totals-leg strategy.

### 5. Player value rebuild (opponent-adjusted, creation-aware, minutes-weighted)
- Acceptance tests, pre-registered: Munster and Grant rate clearly positive vs
  position; Tedesco's decline pattern preserved; team-level feature still adds
  >= current 0.0035 log-loss; scenario swing for a withdrawn elite playmaker
  lands in the 3-6 pt range the market implies.

### 6. Positional debutant uncertainty (added Rd22)
- Exhibit: Watson debut at halfback priced as league-average inside a 24-pt
  edge; market charged variance for an unrated spine player; market was right.
- Build: unmatched/debutant players shrink the prediction toward 50%, weighted
  by position (spine >> outside backs >> forwards), sized from history.
- Pre-registered test: walk-forward every debut-in-spine game 2018-2026; the
  discount must improve log-loss on that subset without hurting the rest.
- KILL IF: no historical subset improvement, or the fitted discount for
  non-spine debutants is indistinguishable from zero AND spine n is too small
  to bound (< 40 games) - in which case park until more data, don't guess.

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
