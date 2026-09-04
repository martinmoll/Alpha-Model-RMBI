# Alpha Strategy Dashboard

An end-to-end quantitative equity research platform that implements the full   pipeline: data ingestion, feature engineering, walk-forward model training, portfolio construction, risk decomposition, and live monitoring. Built as a Streamlit dashboard for interactive experimentation and visual diagnostics.

## Motivation

A lot of the academic backtesting examples I saw just stopped at "train a model, compute Sharpe." I wanted to to see if I could take this further:

- **Is the Sharpe statistically significant?** Bootstrap confidence intervals quantify estimation uncertainty instead of reporting a single point estimate.
- **Where do the returns come from?** Fama-French 5-factor regressions separate genuine stock-selection alpha from repackaged factor exposures.
- **Will it survive transaction costs?** The Fundamental Law of Active Management connects forecast quality (IC), breadth (number of bets), and turnover to answer whether the signal justifies the cost of trading.
- **Is the model still working?** KS distribution shift tests, alpha decay curves, and signal staleness detection flag when the model is operating out-of-distribution.

## Methodology

### Data Pipeline

Live market data is fetched from Yahoo Finance (prices, fundamentals), Ken French's Data Library (FF5 factors), and FRED (VIX, yield curve, EPU). The pipeline assembles a monthly cross-sectional panel of S&P 500 constituents with ~118 engineered features spanning momentum, value, quality, volatility, sentiment, and macro categories. All features are cross-sectionally standardized (rank → normal) each month to ensure stationarity.

### Alpha Models

Seven models are available, split by feature capacity:

| Model | Type | Feature Tier | Key Property |
|-------|------|-------------|--------------|
| HGB (HistGradientBoosting) | Tree ensemble | Tier 2 (~118 features) | Nonlinear interactions, built-in monotone constraints |
| Random Forest | Tree ensemble | Tier 2 | Bagging for variance reduction |
| Lasso (L1) | Penalized linear | Tier 1 (~52 features) | Automatic feature selection via sparsity |
| Ridge (L2) | Penalized linear | Tier 1 | Shrinkage without elimination |
| ElasticNet | Penalized linear | Tier 1 | L1+L2 blend |
| Fama-MacBeth | Two-pass regression | Tier 1 | Econometrically grounded, panel-corrected SEs |
| Ensemble | Stacked average | Tier 2 | Combines tree + linear for robustness |

### Walk-Forward Validation

All backtests use strictly out-of-sample walk-forward evaluation: the model is trained on data up to month *t*, predicts cross-sectional returns for month *t+1*, then rolls forward. Retraining frequency (6/12/24 months) and window type (expanding vs. rolling) are configurable. Optional inner time-series CV auto-tunes hyperparameters at each retrain point.

### Portfolio Construction

Five construction methods translate model scores into portfolio weights:

- **Equal weight** — 1/K allocation to top-scored stocks
- **Score weight** — weights proportional to predicted scores
- **Inverse volatility** — tilts toward lower-volatility picks
- **Equal Risk Contribution (ERC)** — equalizes marginal risk contributions
- **Mean-Variance Optimization (MVO)** — Markowitz with transaction-cost regularization

Long-only and long-short strategies are supported, with configurable volatility tilting and regime-aware position sizing.

### Risk & Diagnostics

- **Factor exposure:** FF5 regression betas with rolling windows, Jensen's alpha with HAC standard errors
- **Bootstrap CIs:** Sharpe ratio and alpha confidence intervals via block bootstrap (5,000 resamples)
- **Information Coefficient:** Spearman rank IC, ICIR, hit rate, and IC t-statistics
- **Fundamental Law:** IR upper bound, required IC for a target Sharpe, cost-adjusted SR
- **R² out-of-sample:** Campbell & Thompson (2008) metric for forecast accuracy
- **VaR/CVaR:** Parametric, historical, Monte Carlo, and Cornish-Fisher methods with Kupiec backtesting
- **Distribution shift:** KS tests across all features to detect out-of-distribution regimes
- **Alpha decay:** IC at forward horizons with half-life estimation
- **Signal staleness:** Turnover-based detection of stale predictions

## Project Structure

```
dashboard/
  app.py                  # Streamlit entry point
  features.py             # Feature engineering (Tier 1 / Tier 2)
  core/
    models.py             # 7 alpha models
    backtest.py           # Walk-forward prediction engine
    portfolio.py          # Portfolio construction (5 methods)
    diagnostics.py        # IC, Fundamental Law, KS test, bootstrap CIs
    risk.py               # VaR/CVaR, factor exposure, risk contribution
    factor_models.py      # CAPM / FF3 / FF5 regressions, rolling beta
    data_loader.py        # Dataset loading and generation
  components/
    theme.py              # Design tokens, Plotly template, font injection
    charts.py             # Plotly chart builders
    metrics.py            # Metric cards, banners, tables
    workflow.py           # Navigation stepper
    interpretations.py    # Context-aware metric interpretations
  pages/                  # Streamlit multipage views
pipeline/
  __init__.py             # Pipeline orchestrator (yfinance -> features -> dataset)
  config.py               # Paths, column schema, constants
  fetchers/               # Price, fundamental, factor, macro data fetchers
  features/               # Feature computation modules
  assembler.py            # Monthly assembly and dataset append
tests/                    # pytest suite (79 tests)
```

## Setup

```bash
pip install -r requirements.txt
```

Optional: set a FRED API key for macro features (VIX, credit spread, EPU):
```bash
export FRED_API_KEY=your_key_here        # Linux/Mac
set FRED_API_KEY=your_key_here           # Windows
```

## Run

```bash
python -m streamlit run dashboard/app.py
```

On first run, the app auto-fetches S&P 500 data via yfinance and builds the dataset. This takes a few minutes.

## Tests

```bash
python -m pytest tests/
```

## Limitations

- **Survivorship bias:** The universe is current S&P 500 constituents, not point-in-time membership. This overstates performance for strategies that would have held delisted stocks.
- **Flat transaction costs:** The cost model uses fixed bps per trade rather than market-impact-aware sizing (e.g., Almgren-Chriss). Capacity analysis is not yet implemented.
- **No short-selling constraints:** The long-short mode assumes unlimited borrow availability at zero cost.
- **Single asset class:** Equities only; no cross-asset signals or hedging.

These are documented intentionally — acknowledging limitations is part of rigorous research.
