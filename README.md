# LEAP Call Options Screener

A Python script that screens a watchlist of stocks for the best **LEAP call options** based on a multi-factor scoring model.

---

## What it does

For every ticker in your watchlist, the screener:

1. Pulls **1 year of daily price history** and computes RSI, Bollinger Bands, and 200-day MA.
2. Fetches **fundamental data** (market cap, average volume, institutional ownership, EPS growth).
3. Selects the **longest available call option expiry** (the LEAP).
4. Computes **Black-Scholes Delta and Theta** for every strike in that expiry.
5. Approximates the **IV Percentile** from rolling realised volatility.
6. Calculates **additional metrics**: required stock move, leverage ratio (Omega), and break-even price.
7. Scores every option on **9 parameters** (0–100 each) and combines them into a **weighted composite score**.
8. Outputs the **top 10 options per stock** and **top 10 options overall** as JSON files.

---

## Screening Parameters

| Category | Parameter | Threshold |
|---|---|---|
| **Fundamental** | Market Cap | > $2 B |
| | Average Volume | > 1 M shares |
| | Institutional Ownership | > 50% |
| | EPS Growth | > +15% (trailing) |
| **Technical** | RSI (14-day) | < 40 |
| | Price vs 200-day MA | Near / slightly above |
| | Bollinger Bands | Touching / near lower band |
| **Option-Specific** | Delta (Black-Scholes) | 0.70 – 0.85 |
| | IV Percentile | < 30% |

---

## Additional Calculated Metrics

| Metric | Formula |
|---|---|
| Required Stock Move | `Premium / Delta` |
| Leverage Ratio (Omega) | `Delta × (Stock Price / Option Price)` |
| Break-Even Price | `Strike + (2 × Premium)` |
| Daily Theta | Black-Scholes Theta ÷ 365 |

---

## Score Weights

| Parameter | Weight |
|---|---|
| Delta | 20% |
| IV Percentile | 16% |
| RSI | 12% |
| EPS Growth | 10% |
| Price vs 200 MA | 10% |
| Bollinger Position | 10% |
| Market Cap | 8% |
| Institutional Ownership | 8% |
| Average Volume | 6% |

---

## Installation

```bash
pip install yfinance pandas numpy scipy
```

---

## Usage

1. Edit `WATCHLIST` at the top of `leap_screener.py`.
2. Optionally adjust `RISK_FREE_RATE` and the threshold constants.
3. Run:

```bash
python leap_screener.py
```

---

## Output Files

| File | Contents |
|---|---|
| `top_10_per_stock.json` | Best 10 call options for each ticker |
| `top_10_overall.json` | Best 10 call options across all tickers |
| `full_results.json` | Every screened option with all metrics |

---

## JSON Schema (per option record)

```json
{
  "ticker": "AAPL",
  "expiry": "2027-01-15",
  "strike": 170.00,
  "contract": "AAPL270115C00170000",
  "current_price": 182.50,
  "market_cap": 2800000000000,
  "avg_volume": 55000000,
  "inst_ownership": 61.2,
  "eps_growth_pct": 18.5,
  "rsi": 36.4,
  "price_vs_200ma_pct": 1.2,
  "bb_lower": 168.30,
  "bb_middle": 179.10,
  "bb_upper": 189.90,
  "bb_position_pct": 65.1,
  "premium": 22.40,
  "implied_volatility_pct": 28.6,
  "iv_percentile": 21.3,
  "dte": 630,
  "delta": 0.7312,
  "theta_daily": -0.0148,
  "required_stock_move": 30.63,
  "leverage_ratio": 5.95,
  "breakeven_price": 214.80,
  "scores": {
    "market_cap": 100.0,
    "avg_volume": 100.0,
    "inst_ownership": 37.3,
    "eps_growth": 10.0,
    "rsi": 17.8,
    "price_vs_200ma": 76.0,
    "bb_position": 0.0,
    "delta": 89.2,
    "iv_percentile": 28.9
  },
  "composite_score": 62.4
}
```

---

## Deploying to Base44 via GitHub

1. Push this repository to GitHub.
2. In **Base44**, create a new project and connect it to your GitHub repo.
3. Point Base44 to the JSON files (`top_10_overall.json`, `top_10_per_stock.json`) as your data source.
4. Build your UI components to read and display the JSON fields listed above.

> **Tip:** Run `leap_screener.py` on a schedule (e.g. GitHub Actions cron, daily before market open) to keep the JSON files fresh. Commit and push the updated JSON; Base44 will pick up the changes automatically.

---

## GitHub Actions – Auto-Refresh (Optional)

Create `.github/workflows/refresh.yml`:

```yaml
name: Refresh LEAP Data

on:
  schedule:
    - cron: "0 12 * * 1-5"   # 12:00 UTC Mon–Fri (pre-market US)
  workflow_dispatch:

jobs:
  run:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install yfinance pandas numpy scipy
      - run: python leap_screener.py
      - name: Commit updated data
        run: |
          git config user.name  "github-actions[bot]"
          git config user.email "github-actions[bot]@users.noreply.github.com"
          git add top_10_per_stock.json top_10_overall.json full_results.json
          git diff --staged --quiet || git commit -m "chore: refresh LEAP data $(date -u +%Y-%m-%d)"
          git push
```

---

## Notes

- **Data source:** [yfinance](https://github.com/ranaroussi/yfinance) (Yahoo Finance). No API key required.
- **IV Percentile** is approximated from 30-day rolling realised volatility — a proxy for the true IV percentile.
- This tool is for **research and educational purposes only** and does not constitute financial advice.
