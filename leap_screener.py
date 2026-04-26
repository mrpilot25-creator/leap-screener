"""
LEAP Call Options Screener
==========================
Automatically fetches every ticker in the S&P 500, screens them against
fundamental, technical, and option-specific parameters, and outputs:

  - top_10_per_stock.json   → best 10 options for each ticker
  - top_10_overall.json     → best 10 options across all tickers
  - full_results.json       → every screened option with all metrics

Dependencies:
    pip install yfinance pandas numpy scipy requests

Usage:
    python leap_screener.py
"""

import json
import math
import time
import warnings
from datetime import datetime, date

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from scipy.stats import norm

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

RISK_FREE_RATE = 0.05          # Approximate annualised risk-free rate
OUTPUT_DIR     = "."           # Folder where JSON files are written
DELAY_SECONDS  = 0.3           # Pause between Yahoo Finance calls

# ── Screening thresholds ──────────────────────────────────────────────
MIN_MARKET_CAP        = 2e9    # $2 B
MIN_AVG_VOLUME        = 1e6    # 1 M shares/day
MIN_INST_OWNERSHIP    = 0.50   # 50 %
MIN_EPS_GROWTH        = 0.15   # +15 %
RSI_UPPER             = 40     # RSI must be BELOW this
PRICE_VS_200MA_RANGE  = 0.05   # Within ±5 % of 200-day MA
BB_NEAR_LOWER_THRESH  = 0.10   # Within 10 % of lower Bollinger Band
DELTA_MIN             = 0.70
DELTA_MAX             = 0.85
MAX_IV_PERCENTILE     = 30     # IV-percentile must be BELOW this


# ─────────────────────────────────────────────
# FETCH S&P 500 TICKERS
# ─────────────────────────────────────────────

def get_sp500_tickers() -> list[str]:
    """
    Scrapes the current S&P 500 constituents from Wikipedia.
    Returns a sorted list of ticker symbols.
    Falls back to a hardcoded list of the 50 largest constituents
    if the Wikipedia fetch fails.
    """
    print("Fetching S&P 500 tickers from Wikipedia...")
    try:
        url    = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        tables = pd.read_html(url)
        df     = tables[0]
        tickers = df["Symbol"].tolist()
        # Yahoo Finance uses hyphens, not dots (e.g. BRK-B not BRK.B)
        tickers = [t.replace(".", "-") for t in tickers]
        tickers = sorted(set(tickers))
        print(f"  Loaded {len(tickers)} tickers from Wikipedia.\n")
        return tickers
    except Exception as e:
        print(f"  [WARN] Wikipedia fetch failed ({e}). Using fallback list.\n")
        return [
            "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "GOOG", "TSLA",
            "BRK-B", "JPM", "LLY", "V", "UNH", "XOM", "MA", "JNJ", "PG",
            "AVGO", "HD", "MRK", "COST", "ABBV", "CVX", "KO", "PEP", "ADBE",
            "WMT", "BAC", "MCD", "CSCO", "CRM", "TMO", "ABT", "ACN", "LIN",
            "NFLX", "DHR", "TXN", "CMCSA", "VZ", "NEE", "PM", "INTC", "RTX",
            "ORCL", "QCOM", "HON", "UPS", "AMGN", "BMY",
        ]


# ─────────────────────────────────────────────
# TECHNICAL INDICATORS
# ─────────────────────────────────────────────

def calculate_rsi(prices: pd.Series, period: int = 14) -> float:
    """Wilder's RSI."""
    delta    = prices.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    rsi      = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1])


def calculate_bollinger_bands(
    prices: pd.Series, period: int = 20, num_std: float = 2.0
) -> tuple[float, float, float]:
    """Returns (upper, middle, lower) Bollinger Bands for the latest bar."""
    middle = prices.rolling(period).mean().iloc[-1]
    std    = prices.rolling(period).std().iloc[-1]
    return (
        float(middle + num_std * std),
        float(middle),
        float(middle - num_std * std),
    )


def calculate_iv_percentile(ticker_symbol: str, current_iv: float) -> float:
    """
    Approximates IV percentile using a year of 30-day rolling realised
    volatility as a proxy for the IV distribution.
    Returns a value in [0, 100].
    """
    try:
        hist     = yf.Ticker(ticker_symbol).history(period="1y")["Close"]
        if len(hist) < 30:
            return 50.0
        log_rets = np.log(hist / hist.shift(1)).dropna()
        roll_vol = log_rets.rolling(30).std() * np.sqrt(252)
        roll_vol = roll_vol.dropna()
        return round(float(np.mean(roll_vol < current_iv) * 100), 1)
    except Exception:
        return 50.0


# ─────────────────────────────────────────────
# BLACK-SCHOLES GREEKS
# ─────────────────────────────────────────────

def black_scholes_delta(
    S: float, K: float, T: float, r: float, sigma: float
) -> float:
    """Call-option Delta from Black-Scholes."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return float("nan")
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    return float(norm.cdf(d1))


def black_scholes_theta(
    S: float, K: float, T: float, r: float, sigma: float
) -> float:
    """Call-option Theta (per calendar day, in $ terms)."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return float("nan")
    d1    = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2    = d1 - sigma * math.sqrt(T)
    term1 = -(S * norm.pdf(d1) * sigma) / (2 * math.sqrt(T))
    term2 = r * K * math.exp(-r * T) * norm.cdf(d2)
    return float((term1 - term2) / 365)


# ─────────────────────────────────────────────
# SCORING ENGINE
# ─────────────────────────────────────────────

def _clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


def score_parameter(value, kind: str) -> float:
    """Converts a raw metric value to a score in [0, 100]."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return 0.0

    if kind == "market_cap":
        return _clamp((value - MIN_MARKET_CAP) / 8e9 * 100, 0, 100)

    if kind == "avg_volume":
        return _clamp((value - MIN_AVG_VOLUME) / 9e6 * 100, 0, 100)

    if kind == "inst_ownership":
        return _clamp((value - MIN_INST_OWNERSHIP) / 0.30 * 100, 0, 100)

    if kind == "eps_growth":
        return _clamp((value - MIN_EPS_GROWTH) / 0.35 * 100, 0, 100)

    if kind == "rsi":
        if value > RSI_UPPER:
            return 0.0
        return _clamp((RSI_UPPER - value) / 20.0 * 100, 0, 100)

    if kind == "price_vs_200ma":
        if abs(value) > PRICE_VS_200MA_RANGE:
            return max(0.0, 100 - abs(value) / PRICE_VS_200MA_RANGE * 50)
        return 100 - abs(value) / PRICE_VS_200MA_RANGE * 30

    if kind == "bb_position":
        return _clamp((1 - value) * 100, 0, 100) if value <= 0.5 else 0.0

    if kind == "delta":
        if DELTA_MIN <= value <= DELTA_MAX:
            mid = (DELTA_MIN + DELTA_MAX) / 2
            return 100 - abs(value - mid) / 0.075 * 40
        dist = min(abs(value - DELTA_MIN), abs(value - DELTA_MAX))
        return _clamp(100 - dist / 0.10 * 100, 0, 100)

    if kind == "iv_percentile":
        if value > MAX_IV_PERCENTILE:
            return 0.0
        return _clamp((MAX_IV_PERCENTILE - value) / MAX_IV_PERCENTILE * 100, 0, 100)

    return 0.0


# Weights for each parameter in the composite score (must sum to 1.0)
WEIGHTS = {
    "market_cap":      0.08,
    "avg_volume":      0.06,
    "inst_ownership":  0.08,
    "eps_growth":      0.10,
    "rsi":             0.12,
    "price_vs_200ma":  0.10,
    "bb_position":     0.10,
    "delta":           0.20,
    "iv_percentile":   0.16,
}


def compute_composite_score(scores: dict) -> float:
    return round(sum(WEIGHTS[k] * scores.get(k, 0) for k in WEIGHTS), 2)


# ─────────────────────────────────────────────
# FUNDAMENTAL DATA
# ─────────────────────────────────────────────

def fetch_fundamental_data(ticker: yf.Ticker) -> dict:
    """Pulls fundamental metrics from yfinance with safe fallbacks."""
    info = ticker.info or {}

    market_cap = info.get("marketCap")
    avg_volume = info.get("averageVolume")
    inst_own   = info.get("heldPercentInstitutions")

    eps_growth = None
    try:
        eh = ticker.earnings_history
        if eh is not None and len(eh) >= 8:
            recent_4q = eh["epsActual"].iloc[:4].sum()
            prior_4q  = eh["epsActual"].iloc[4:8].sum()
            if prior_4q and abs(prior_4q) > 0.001:
                eps_growth = (recent_4q - prior_4q) / abs(prior_4q)
    except Exception:
        pass

    if eps_growth is None:
        eps_growth = info.get("earningsGrowth")

    return {
        "market_cap":    market_cap,
        "avg_volume":    avg_volume,
        "inst_ownership": inst_own,
        "eps_growth":    eps_growth,
    }


def passes_fundamental_screen(fund: dict) -> bool:
    """
    Returns True only if ALL four fundamental criteria are met.
    Tickers that fail are skipped before fetching options data,
    saving significant time and API calls.
    """
    mc = fund.get("market_cap")
    av = fund.get("avg_volume")
    io = fund.get("inst_ownership")
    eg = fund.get("eps_growth")

    if mc is None or mc < MIN_MARKET_CAP:
        return False
    if av is None or av < MIN_AVG_VOLUME:
        return False
    if io is None or io < MIN_INST_OWNERSHIP:
        return False
    if eg is None or eg < MIN_EPS_GROWTH:
        return False
    return True


# ─────────────────────────────────────────────
# MAIN ANALYSIS FUNCTION
# ─────────────────────────────────────────────

def analyze_ticker(symbol: str, idx: int, total: int) -> list[dict]:
    """
    Fetches data for one ticker and returns a list of option records.
    Returns [] if the ticker fails the fundamental screen or on any error.
    """
    print(f"  [{idx:>3}/{total}] {symbol:<8}", end="", flush=True)
    results = []

    try:
        ticker = yf.Ticker(symbol)

        # ── Fundamentals first — skip early if they don't pass ─────────
        fund = fetch_fundamental_data(ticker)
        if not passes_fundamental_screen(fund):
            mc = fund.get("market_cap")
            io = fund.get("inst_ownership")
            eg = fund.get("eps_growth")
            mc_s = f"${mc/1e9:.0f}B" if mc else "?"
            io_s = f"{io*100:.0f}%" if io else "?"
            eg_s = f"+{eg*100:.0f}%" if eg else "?"
            print(f" ✗  MCap={mc_s} InstOwn={io_s} EPSGrowth={eg_s}")
            return []

        # ── Price history ──────────────────────────────────────────────
        hist = ticker.history(period="1y")
        if hist.empty or len(hist) < 30:
            print(" ✗  Insufficient price history")
            return []
        close         = hist["Close"]
        current_price = float(close.iloc[-1])

        # ── Technical indicators ───────────────────────────────────────
        rsi_val = calculate_rsi(close)
        bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(close)
        ma_200  = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else float(close.mean())
        bb_width = (bb_upper - bb_lower) or 1
        bb_pos   = (current_price - bb_lower) / bb_width
        price_vs_200ma = (current_price - ma_200) / ma_200

        # ── Options chain (longest expiry = LEAP) ─────────────────────
        expirations = ticker.options
        if not expirations:
            print(" ✗  No options data")
            return []

        expiry = expirations[-1]
        chain  = ticker.option_chain(expiry)
        calls  = chain.calls.copy()

        if calls.empty:
            print(f" ✗  No calls at {expiry}")
            return []

        exp_date = datetime.strptime(expiry, "%Y-%m-%d").date()
        dte      = max((exp_date - date.today()).days, 1)
        T        = dte / 365.0

        mid_iv        = float(calls["impliedVolatility"].median())
        iv_percentile = calculate_iv_percentile(symbol, mid_iv)

        # ── Base scores (same for every strike on this ticker) ─────────
        param_scores_base = {
            "market_cap":     score_parameter(fund["market_cap"],    "market_cap"),
            "avg_volume":     score_parameter(fund["avg_volume"],     "avg_volume"),
            "inst_ownership": score_parameter(fund["inst_ownership"], "inst_ownership"),
            "eps_growth":     score_parameter(fund["eps_growth"],     "eps_growth"),
            "rsi":            score_parameter(rsi_val,                "rsi"),
            "price_vs_200ma": score_parameter(price_vs_200ma,         "price_vs_200ma"),
            "bb_position":    score_parameter(bb_pos,                 "bb_position"),
        }

        # ── Process each call strike ────────────────────────────────────
        for _, opt in calls.iterrows():
            strike = float(opt["strike"])
            iv     = float(opt["impliedVolatility"])

            bid  = float(opt.get("bid", 0) or 0)
            ask  = float(opt.get("ask", 0) or 0)
            prem = (bid + ask) / 2 if (bid + ask) > 0 else float(opt.get("lastPrice", 0) or 0)
            if prem <= 0:
                continue

            delta = black_scholes_delta(current_price, strike, T, RISK_FREE_RATE, iv)
            theta = black_scholes_theta(current_price, strike, T, RISK_FREE_RATE, iv)
            if math.isnan(delta):
                continue

            required_move  = prem / delta if delta != 0 else float("nan")
            leverage_ratio = delta * (current_price / prem) if prem != 0 else float("nan")
            breakeven      = strike + 2 * prem

            opt_iv_pct = (
                calculate_iv_percentile(symbol, iv)
                if abs(iv - mid_iv) > 0.01
                else iv_percentile
            )

            param_scores = {
                **param_scores_base,
                "delta":        score_parameter(delta,       "delta"),
                "iv_percentile": score_parameter(opt_iv_pct, "iv_percentile"),
            }
            composite = compute_composite_score(param_scores)

            results.append({
                # Identification
                "ticker":   symbol,
                "expiry":   expiry,
                "strike":   round(strike, 2),
                "contract": opt.get("contractSymbol", ""),

                # Stock metrics
                "current_price":   round(current_price, 2),
                "market_cap":      fund["market_cap"],
                "avg_volume":      fund["avg_volume"],
                "inst_ownership":  round(fund["inst_ownership"] * 100, 1) if fund["inst_ownership"] else None,
                "eps_growth_pct":  round(fund["eps_growth"] * 100, 1)     if fund["eps_growth"]     else None,

                # Technical metrics
                "rsi":                round(rsi_val, 2),
                "price_vs_200ma_pct": round(price_vs_200ma * 100, 2),
                "bb_lower":           round(bb_lower, 2),
                "bb_middle":          round(bb_middle, 2),
                "bb_upper":           round(bb_upper, 2),
                "bb_position_pct":    round(bb_pos * 100, 1),

                # Option metrics
                "premium":                   round(prem, 2),
                "implied_volatility_pct":    round(iv * 100, 2),
                "iv_percentile":             round(opt_iv_pct, 1),
                "dte":                       dte,

                # Greeks
                "delta":       round(delta, 4),
                "theta_daily": round(theta, 4),

                # Derived metrics
                "required_stock_move": round(required_move, 2)  if not math.isnan(required_move)  else None,
                "leverage_ratio":      round(leverage_ratio, 2) if not math.isnan(leverage_ratio) else None,
                "breakeven_price":     round(breakeven, 2),

                # Scores
                "scores":          {k: round(v, 1) for k, v in param_scores.items()},
                "composite_score": composite,
            })

        results.sort(key=lambda x: x["composite_score"], reverse=True)
        mc_s = f"${fund['market_cap']/1e9:.0f}B" if fund['market_cap'] else "?"
        print(f" ✓  {len(results)} options  MCap={mc_s}  Expiry={expiry}")
        time.sleep(DELAY_SECONDS)
        return results

    except Exception as e:
        print(f" ✗  Error: {e}")
        return []


# ─────────────────────────────────────────────
# OUTPUT HELPERS
# ─────────────────────────────────────────────

def save_json(data, filename: str):
    path = f"{OUTPUT_DIR}/{filename}"
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  Saved → {path}")


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  LEAP CALL OPTIONS SCREENER  —  Full S&P 500")
    print(f"  Run date: {date.today()}")
    print("=" * 65)

    # ── Fetch tickers ────────────────────────────────────────────────
    watchlist = get_sp500_tickers()
    total     = len(watchlist)

    print(f"Screening {total} tickers...\n")

    all_results: list[dict]         = []
    per_stock:   dict[str, list]    = {}
    passed_tickers: list[str]       = []

    for idx, symbol in enumerate(watchlist, 1):
        options = analyze_ticker(symbol, idx, total)
        if options:
            passed_tickers.append(symbol)
            per_stock[symbol] = options[:10]
            all_results.extend(options)

    # ── Sort and slice ───────────────────────────────────────────────
    all_results.sort(key=lambda x: x["composite_score"], reverse=True)
    top_10_overall = all_results[:10]

    # ── Build output payloads ────────────────────────────────────────
    run_meta = {
        "generated_at":   datetime.utcnow().isoformat() + "Z",
        "sp500_universe": total,
        "tickers_passed": len(passed_tickers),
        "tickers_list":   passed_tickers,
        "screening_parameters": {
            "min_market_cap_usd":     MIN_MARKET_CAP,
            "min_avg_volume_shares":  MIN_AVG_VOLUME,
            "min_inst_ownership_pct": MIN_INST_OWNERSHIP * 100,
            "min_eps_growth_pct":     MIN_EPS_GROWTH * 100,
            "rsi_upper":              RSI_UPPER,
            "price_vs_200ma_range":   f"±{PRICE_VS_200MA_RANGE*100}%",
            "delta_range":            [DELTA_MIN, DELTA_MAX],
            "max_iv_percentile":      MAX_IV_PERCENTILE,
        },
        "score_weights": WEIGHTS,
    }

    print(f"\n{len(passed_tickers)}/{total} tickers passed fundamental screen.")
    print(f"Saving output files...")

    save_json({"meta": run_meta, "top_10_per_stock":      per_stock},      "top_10_per_stock.json")
    save_json({"meta": run_meta, "top_10_overall":        top_10_overall}, "top_10_overall.json")
    save_json({"meta": run_meta, "all_screened_options":  all_results},    "full_results.json")

    # ── Console summary ──────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  TOP 10 OVERALL (by composite score)")
    print("=" * 65)
    print(f"{'#':<3} {'Ticker':<7} {'Strike':>8} {'Expiry':<12} "
          f"{'Prem':>7} {'Delta':>7} {'IV%':>6} {'IVPct':>6} "
          f"{'RSI':>5} {'Score':>7}")
    print("-" * 65)
    for i, opt in enumerate(top_10_overall, 1):
        print(
            f"{i:<3} {opt['ticker']:<7} {opt['strike']:>8.2f} {opt['expiry']:<12} "
            f"${opt['premium']:>6.2f} {opt['delta']:>7.4f} "
            f"{opt['implied_volatility_pct']:>5.1f}% {opt['iv_percentile']:>5.1f}% "
            f"{opt['rsi']:>5.1f} {opt['composite_score']:>7.1f}"
        )
    print("=" * 65)
    print("Done.\n")


if __name__ == "__main__":
    main()
