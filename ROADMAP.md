# Roadmap — toward an institutional-grade platform

Prioritized additions that move this project from a strong research prototype
toward something a systematic-equity desk would recognize. Ordered by
credibility impact per unit effort. Grounded in standard practice (López de
Prado, Grinold–Kahn, Bailey & López de Prado, Almgren–Chriss, Barra/Axioma,
GIPS) and the current codebase. Deeper implementation notes live in
`FUTURE_IMPROVEMENTS.md`.

Legend: `[x]` done · `[~]` in progress · `[ ]` planned · `(data-limited)` needs
data the free sources can't fully provide.

---

## Tier 1 — Data realism (biggest credibility gaps)

- [x] **Fundamental reporting lag (point-in-time fundamentals).** Fundamentals
  now become available `REPORTING_LAG_MONTHS` (default 2) after the fiscal
  quarter-end, not on it — a 10-Q is filed ~40–90 days later. Previously
  `fundamental_features.py` stamped a quarter's data to the quarter-end month,
  a classic look-ahead. *Fixed in the pipeline (rebuild to apply).*
- [~] **Realistic transaction costs + capacity analysis.** Square-root market-
  impact model (`impact ∝ σ·√(participation)`) plus a capacity curve
  (net Sharpe vs AUM), replacing the flat-bps assumption. README named
  Almgren–Chriss as the missing piece. *Core + Costs-tab panel added.*
- [ ] **Point-in-time universe + delisting returns (survivorship).** *(data-limited)*
  **:rotating_light: NOW THE TOP PRIORITY — measured 2026-09-03.** The strategy
  earns ~5x more than its own IC can justify (mean IC 0.028 supports ~7.6%/yr of
  selection; it delivers ~40%/yr), with beta −0.41 and R² 0.02, so none of it is
  market exposure. The held book sits at `ivol_xs` +2.60 and `vol_12m_xs` +2.49 —
  it concentrates in the extreme-volatility tail, which is exactly where the
  survivor filter distorts the universe most. Every high-vol name that went to
  zero is absent. **No backtest number here is usable for real money until this
  is fixed.** Full write-up in
  `docs/superpowers/plans/2026-09-02-feature-repair-and-european-expansion-plan.md`.
  The #1 documented limitation: the universe is *today's* constituents, so the
  backtest holds survivors. A full fix needs point-in-time index membership and
  delisting returns (CRSP-style). Free-data steps: (a) handle delisting returns
  for names that drop out, (b) freeze historical universe snapshots, (c) at
  minimum quantify the bias. Full version requires paid PIT data — be explicit.

## Tier 2 — Institutional portfolio construction

- [ ] **Cross-sectional factor risk model + constrained optimization.** A
  structural risk model (factor exposures × factor covariance + specific risk)
  used for both optimization and risk decomposition; constrained MVO
  (sector-neutral, beta-neutral, factor-exposure targets, position/turnover/
  leverage caps). Today MVO uses a sample covariance of only the top-K names.
  *The single biggest "real book" upgrade.*
- [ ] **Benchmark-relative framework.** Active weights, tracking error,
  Information Ratio vs a benchmark, active share, and (optionally) min-tracking-
  error optimization. Mandates are relative, not absolute.

## Tier 3 — Overfitting control & validation

- [x] **Deflated Sharpe Ratio + Probability of Backtest Overfitting (PBO).**
  Bailey & López de Prado. DSR shrinks the Sharpe for the number of trials,
  trial variance, and non-normality; PBO (via CSCV) estimates the probability
  the selected config is overfit. *Probabilistic + Deflated Sharpe on the
  Backtest Overview; PBO on the Compare tab.*
- [ ] **Purged & embargoed cross-validation (CPCV).** Best-practice financial-ML
  tuning; the auto-tune currently uses plain time-series CV.
- [ ] **Signal orthogonalization / alpha combination.** Residualize signals vs
  known risk factors (isolate residual alpha); combine multiple weak signals via
  cross-sectional regression (Grinold–Kahn) or IC-weighting.

## Tier 4 — Production & reporting realism

- [ ] **Standardized tear sheet.** Consolidated, exportable (HTML/PDF) report:
  rolling Sharpe, monthly/annual tables, drawdown analysis, factor exposures
  over time, per-year attribution (pyfolio/quantstats style).
- [ ] **Paper-trading / live target-portfolio loop.** Generate today's target
  portfolio, persist it, track realized-vs-expected and live IC.
- [ ] **Execution realism.** Trade at next-day open with implementation-shortfall
  modeling instead of frictionless month-end close.

## Breadth (opportunistic)

- [ ] Alternative-data signals (short interest, news/social sentiment,
  options-implied — the IV feature slots exist but are unavailable from free data).
- [ ] Multi-asset / cross-asset signals and hedging.
- [ ] Regional macro (a NOK/Norway macro set for the Oslo book).
