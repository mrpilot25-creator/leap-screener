# LEAP Call Options Screener + Walk-Forward Backtest Suite

A Python-based screening and backtesting tool that evaluates stocks against a set of fundamental, technical, and options-specific criteria to identify the best LEAP call option opportunities, then validates each top candidate using a 5-year walk-forward Monte Carlo simulation.

Results are automatically saved as JSON files and can be displayed on a live website powered by Base44.

---

## What It Does

### Phase 1 - Screening

For every ticker in the watchlist, the screener:

1. Fetches one year of daily price history
2. Pulls fundamental data from Yahoo Finance
3. Applies six fundamental filters - tickers that fail are skipped immediately
4. Computes RSI, Bollinger Bands, and 200-day Moving Average
5. Identifies the longest available call option expiry (the LEAP)
6. Calculates Black-Scholes Delta and Theta for every strike
7. Estimates IV Percentile from rolling realised volatility
8. Calculates additional metrics including leverage ratio, break-even price, and required stock move
9. Scores every option across 11 parameters and produces a weighted composite score
10. Saves the top 20 options per stock and the top 20 overall as JSON files

### Phase 2 - Walk-Forward Backtest

For the top 5 candidates by composite score, the backtest engine:

1. Fetches 5 years of daily price and dividend history
2. Splits data into a 3-year Training Window and a 2-year Validation Window
3. Calibrates drift (mu) adjusted for dividend yield and a risk-free rate of 3.88%
4. Applies a mean-reverting IV overlay blending training and validation volatility
5. Runs 1,000 vectorized Monte Carlo simulations using Geometric Brownian Motion
6. Compares simulated price paths against actual validation prices
7. Calculates a Reality Option Premium using actual historical volatility plus a 3% IV-HV spread
8. Produces Greek attribution (Delta, Theta, Rho) at entry, midpoint, and near expiry
9. Generates a Residual Variance Report including P(ITM), Expected Value, terminal price error, and drawdown error

---

## Screening Criteria

### Fundamental Filters

| Parameter | Threshold | Notes |
|---|---|---|
| Market Cap | Greater than $25B | Hard fail if missing or below threshold |
| Average Daily Volume | Greater than 1M shares | Only fails if data exists and is below threshold |
| Institutional Ownership | Greater than 50% | Only fails if data exists and is below threshold |
| EPS Growth | Greater than +15% | Only fails if data exists and is below threshold |
| Revenue Growth | Greater than +8% | Only fails if data exists and is below threshold |
| EBITDA Margin | Greater than 15% | Only fails if data exists and is below threshold |

### Technical Filters (used for scoring, not hard exclusion)

| Parameter | Target |
|---|---|
| RSI (14-day) | Below 40 |
| Price vs 200-Day MA | Near or just above |
| Bollinger Band Position | Near lower band |

### Option Filters (used for scoring, not hard exclusion)

| Parameter | Target |
|---|---|
| Delta (Black-Scholes) | 0.70 to 0.85 |
| IV Percentile | Below 30% |

---

## Composite Score Weights

Each parameter is scored from 0 to 100 and combined into a weighted composite score. The composite score is also used as the entry signal for the backtest engine.

| Parameter | Weight |
|---|---|
| Delta | 20% |
| RSI | 12% |
| IV Percentile | 12% |
| Price vs 200-Day MA | 10% |
| Bollinger Band Position | 10% |
| EPS Growth | 7% |
| Market Cap | 6% |
| Institutional Ownership | 6% |
| Revenue Growth | 6% |
| EBITDA Margin | 6% |
| Average Volume | 5% |

---

## Backtest Engine Details

### Data Partitioning

| Window | Length | Purpose |
|---|---|---|
| Training | 3 years | Calibrates drift, volatility, and dividend yield |
| Validation | 2 years | Compares Monte Carlo projections against actual prices |

### Monte Carlo Simulation

- Model: Geometric Brownian Motion (GBM)
- Paths: 1,000 simulations, fully vectorized using NumPy
- Drift: Historical mean return minus dividend yield
- Volatility: Training sigma blended with validation sigma using an Ornstein-Uhlenbeck mean-reversion decay
- Risk-Free Rate: 3.88% (2-Year Treasury proxy)

### Reality Check

Compares the simulated price distribution against actual historical prices over the 2-year validation window. The Reality Option Premium is calculated using actual historical volatility plus a 3% spread to simulate the IV-HV gap.

### Greek Attribution

Calculates Delta, Theta, and Rho at three checkpoints over the validation period: entry, midpoint, and near expiry. Also reports cumulative theta cost over the full validation window.

### Residual Variance Report

| Metric | Description |
|---|---|
| P(ITM) | Probability that the simulated terminal price exceeds the strike |
| Expected Value (EV) | P(ITM) times expected payoff minus P(OTM) times premium paid |
| Terminal Price Error | Percentage difference between median simulated and actual terminal price |
| Max Drawdown Error | Difference between simulated and actual maximum drawdown |
| 90% Confidence Band | Percentage of actual prices that fell within the simulated 5th-95th percentile range |

---

## Output Files

| File | Contents |
|---|---|
| top_20_per_stock.json | Best 20 call options for each ticker in the watchlist |
| top_20_overall.json | Best 20 call options ranked across all tickers |
| full_results.json | Every screened option with all metrics and scores |
| backtest_results.json | Walk-forward backtest report for the top 5 candidates |

---

## Screener Output Fields

### Identification
- ticker, expiry, strike, contract

### Stock Metrics
- current_price, market_cap, avg_volume, inst_ownership
- eps_growth_pct, revenue_growth_pct, ebitda_margin_pct

### Technical Metrics
- rsi, price_vs_200ma_pct
- bb_lower, bb_middle, bb_upper, bb_position_pct

### Option Metrics
- premium, implied_volatility_pct, iv_percentile, dte

### Greeks
- delta, theta_daily

### Derived Metrics
- required_stock_move, leverage_ratio, breakeven_price

### Scores
- scores (individual score for each parameter)
- composite_score (weighted total out of 100)

---

## Backtest Output Fields

Each backtest record contains:

- symbol, strike, entry_score, entry_price
- calibration: annualised mu, dividend yield, training sigma, validation sigma, blended sigma, risk-free rate
- reality_check: actual vs simulated terminal price, max drawdown comparison, reality premium
- greek_attribution: timeline of Delta, Theta, Rho at key checkpoints, cumulative theta cost
- residual_variance_report: P(ITM), EV, terminal price error, drawdown error, confidence band coverage

---

## Installation

```
pip install yfinance pandas numpy scipy
```

---

## Usage

1. Open leap_screener.py
2. Edit the WATCHLIST at the top of the file with your desired tickers
3. Run the script:

```
python leap_screener.py
```

The script runs Phase 1 (screening) followed by Phase 2 (backtest) automatically.

To disable the backtest, set this flag at the top of leap_screener.py:

```python
RUN_BACKTEST = False
```

---

## Editing the Watchlist

Open leap_screener.py and update the WATCHLIST variable near the top:

```python
WATCHLIST = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
    "META", "TSLA", "JPM", "UNH", "V",
]
```

Tickers must match the Yahoo Finance format (for example BRK-B not BRK.B).

---

## Editing the Thresholds

All thresholds are defined at the top of leap_screener.py:

```python
MIN_MARKET_CAP     = 25e9   # $25 Billion
MIN_AVG_VOLUME     = 1e6    # 1 Million shares/day
MIN_INST_OWNERSHIP = 0.50   # 50%
MIN_EPS_GROWTH     = 0.15   # +15%
MIN_REVENUE_GROWTH = 0.08   # +8%
MIN_EBITDA_MARGIN  = 0.15   # 15%
RSI_UPPER          = 40
DELTA_MIN          = 0.70
DELTA_MAX          = 0.85
MAX_IV_PERCENTILE  = 30
RISK_FREE_RATE     = 0.0388  # 2-Year Treasury proxy
```

Backtest settings can also be adjusted:

```python
RUN_BACKTEST         = True   # Set to False to skip backtest
BT_TRAINING_YEARS    = 3
BT_VALIDATION_YEARS  = 2
BT_N_SIMULATIONS     = 1000
BT_TOP_N_CANDIDATES  = 5      # Number of top picks to backtest
```

---

## Automated Refresh via GitHub Actions

The repository includes a workflow file at .github/workflows/refresh.yml that automatically runs the screener and backtest Monday to Friday at 12:00 UTC and commits the updated JSON files back to the repository.

### Setup

1. Go to your repository on GitHub
2. Click Settings then Actions then General
3. Under Workflow permissions, select Read and write permissions
4. Click Save

The workflow will then run on schedule automatically. You can also trigger it manually from the Actions tab at any time.

---

## Connecting to Base44

Point Base44 to the following raw GitHub URLs:

```
https://raw.githubusercontent.com/YOUR_USERNAME/YOUR_REPO/main/top_20_overall.json
https://raw.githubusercontent.com/YOUR_USERNAME/YOUR_REPO/main/top_20_per_stock.json
https://raw.githubusercontent.com/YOUR_USERNAME/YOUR_REPO/main/backtest_results.json
```

Replace YOUR_USERNAME and YOUR_REPO with your actual GitHub username and repository name.

---

## Data Source

All data is sourced from Yahoo Finance via the yfinance library. No API key is required.

Note: Yahoo Finance does not always populate every data field for every stock. When a field is missing, the ticker is still included in results but scores 0 for that parameter rather than being excluded. The only exception is market cap - if that is missing the ticker is skipped.

---

## Disclaimer

This tool is for research and educational purposes only. It does not constitute financial advice. Always conduct your own research before making any investment decisions.
