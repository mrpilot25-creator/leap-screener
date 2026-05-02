# LEAP Call Options Screener

A Python-based screening tool that evaluates stocks against a set of fundamental, technical, and options-specific criteria to identify the best LEAP call option opportunities.

Results are automatically saved as JSON files and can be displayed on a live website powered by Base44.

---

## What It Does

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
10. Saves the top 10 options per stock and the top 10 overall as JSON files

---

## Screening Criteria

### Fundamental Filters

| Parameter | Threshold | Notes |
|---|---|---|
| Market Cap | Greater than $2B | Hard fail if missing or below threshold |
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

Each parameter is scored from 0 to 100 and combined into a weighted composite score.

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

## Output Fields

Each screened option record contains the following fields:

### Identification
- ticker
- expiry
- strike
- contract

### Stock Metrics
- current_price
- market_cap
- avg_volume
- inst_ownership
- eps_growth_pct
- revenue_growth_pct
- ebitda_margin_pct

### Technical Metrics
- rsi
- price_vs_200ma_pct
- bb_lower
- bb_middle
- bb_upper
- bb_position_pct

### Option Metrics
- premium
- implied_volatility_pct
- iv_percentile
- dte (days to expiry)

### Greeks
- delta
- theta_daily

### Derived Metrics
- required_stock_move
- leverage_ratio
- breakeven_price

### Scores
- scores (individual score for each parameter)
- composite_score (weighted total out of 100)

---

## Output Files

| File | Contents |
|---|---|
| top_10_per_stock.json | Best 10 call options for each ticker in the watchlist |
| top_10_overall.json | Best 10 call options ranked across all tickers |
| full_results.json | Every screened option with all metrics and scores |

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

The script will print a live progress log showing each ticker, whether it passed or failed the fundamental screen, and the reason for any failure.

---

## Editing the Watchlist

Open leap_screener.py and update the WATCHLIST variable near the top of the file:

```python
WATCHLIST = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
    "META", "TSLA", "JPM", "UNH", "V",
]
```

Add or remove any ticker symbols as needed. Tickers must match the format used by Yahoo Finance (for example BRK-B not BRK.B).

---

## Editing the Thresholds

All thresholds are defined at the top of leap_screener.py and can be adjusted:

```python
MIN_MARKET_CAP     = 2e9   # $2 Billion
MIN_AVG_VOLUME     = 1e6   # 1 Million shares/day
MIN_INST_OWNERSHIP = 0.50  # 50%
MIN_EPS_GROWTH     = 0.15  # +15%
MIN_REVENUE_GROWTH = 0.08  # +8%
MIN_EBITDA_MARGIN  = 0.15  # 15%
RSI_UPPER          = 40
DELTA_MIN          = 0.70
DELTA_MAX          = 0.85
MAX_IV_PERCENTILE  = 30
```

---

## Automated Refresh via GitHub Actions

The repository includes a workflow file at .github/workflows/refresh.yml that automatically runs the screener Monday to Friday at 12:00 UTC and commits the updated JSON files back to the repository.

### Setup

1. Go to your repository on GitHub
2. Click Settings then Actions then General
3. Under Workflow permissions, select Read and write permissions
4. Click Save

The workflow will then run on schedule automatically. You can also trigger it manually from the Actions tab at any time.

---

## Connecting to Base44

The JSON output files are designed to be read directly by a Base44 web application. Once the workflow has run and the JSON files exist in the repository, point Base44 to the following raw GitHub URLs:

```
https://raw.githubusercontent.com/YOUR_USERNAME/YOUR_REPO/main/top_10_overall.json
https://raw.githubusercontent.com/YOUR_USERNAME/YOUR_REPO/main/top_10_per_stock.json
```

Replace YOUR_USERNAME and YOUR_REPO with your actual GitHub username and repository name.

The Base44 app fetches fresh data on every page load, so it will always reflect the most recent screener run.

---

## Data Source

All data is sourced from Yahoo Finance via the yfinance library. No API key is required.

Note: Yahoo Finance does not always populate every data field for every stock. When a field such as institutional ownership or EPS growth is missing, the ticker is still included in results but scores 0 for that parameter rather than being excluded. The only exception is market cap - if that is missing the ticker is skipped.

---

## Disclaimer

This tool is for research and educational purposes only. It does not constitute financial advice. Always conduct your own research before making any investment decisions.
