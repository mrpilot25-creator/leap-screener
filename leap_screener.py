"""
LEAP Call Options Screener
==========================
Screens stocks and their longest-dated call options against a set of
fundamental, technical, and option-specific parameters. Outputs:
  - top_10_per_stock.json   → best 10 options for each ticker
  - top_10_overall.json     → best 10 options across all tickers
  - full_results.json       → every screened option with all metrics

Dependencies:
    pip install yfinance pandas numpy scipy

Usage:
    python leap_screener.py
    # Edit WATCHLIST below before running.
"""

import json
import math
import warnings
from datetime import datetime, date

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIG  –  Edit these values
# ─────────────────────────────────────────────

WATCHLIST = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
    "META", "TSLA", "JPM", "UNH", "V",
]

RISK_FREE_RATE = 0.05          # Approximate annualised risk-free rate
OUTPUT_DIR = "."               # Folder where JSON files are written

# ── Screening thresholds (change to match your preferences) ──
MIN_MARKET_CAP        = 2e9    # $2 B
MIN_AVG_VOLUME        = 1e6    # 1 M shares
MIN_INST_OWNERSHIP    = 0.50   # 50 %
MIN_EPS_GROWTH        = 0.15   # 15 % over trailing period
RSI_UPPER             = 40     # RSI must be BELOW this
PRICE_VS_200MA_RANGE  = 0.05   # Within ±5 % of 200-day MA
BB_NEAR_LOWER_THRESH  = 0.10   # Within 10 % of lower Bollinger Band
DELTA_MIN             = 0.70
DELTA_MAX             = 0.85
MAX_IV_PERCENTILE     = 30     # IV-percentile must be BELOW this


# ─────────────────────────────────────────────
# TECHNICAL INDICATORS
# ─────────────────────────────────────────────

def calculate_rsi(prices: pd.Series, period: int = 14) -> float:
    """Wilder's RSI."""
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1])


def calculate_bollinger_bands(
    prices: pd.Series, period: int = 20, num_std: float = 2.0
) -> tuple[float, float, float]:
    """Returns (upper, middle, lower) Bollinger Bands for the latest bar."""
    middle = prices.rolling(period).mean().iloc[-1]
    std    = prices.rolling(period).std().iloc[-1]
    return float(middle + num_std * std), float(middle), float(middle - num_std * std)


def calculate_iv_percentile(ticker_symbol: str, current_iv: float) -> float:
    """
    Approximates IV percentile using a year of daily realised volatility
    (30-day rolling) as a proxy for the IV distribution.
    Returns a value in [0, 100].
    """
    try:
        hist = yf.Ticker(ticker_symbol).history(period="1y")["Close"]
        if len(hist) < 30:
            return 50.0
        log_rets = np.log(hist / hist.shift(1)).dropna()
        rolling_vol = log_rets.rolling(30).std() * np.sqrt(252)
        rolling_vol = rolling_vol.dropna()
        pct = float(np.mean(rolling_vol < current_iv) * 100)
        return round(pct, 1)
    except Exception:
        return 50.0


# ─────────────────────────────────────────────
# BLACK-SCHOLES GREEKS
# ─────────────────────────────────────────────

def black_scholes_delta(
    S: float, K: float, T: float, r: float, sigma: float
) -> float:
    """
    Call-option Delta from Black-Scholes.
    Returns NaN if inputs are invalid.
    """
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return float("nan")
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    return float(norm.cdf(d1))


def black_scholes_theta(
    S: float, K: float, T: float, r: float, sigma: float
) -> float:
    """
    Call-option Theta (per calendar day, in $ terms).
    """
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return float("nan")
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    term1 = -(S * norm.pdf(d1) * sigma) / (2 * math.sqrt(T))
    term2 = r * K * math.exp(-r * T) * norm.cdf(d2)
    return float((term1 - term2) / 365)


# ─────────────────────────────────────────────
# SCORING ENGINE
# ─────────────────────────────────────────────

def _clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


def score_parameter(value, kind: str) -> float:
    """
    Converts a raw metric value to a score in [0, 100].
    Higher score = better match to the screening criterion.
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return 0.0

    if kind == "market_cap":
        # Anything above $2 B scores; scales up to $10 B
        return _clamp((value - MIN_MARKET_CAP) / (8e9) * 100, 0, 100)

    if kind == "avg_volume":
        # Above 1 M; scales up to 10 M
        return _clamp((value - MIN_AVG_VOLUME) / (9e6) * 100, 0, 100)

    if kind == "inst_ownership":
        # Above 50 %; perfect at 80 %+
        return _clamp((value - MIN_INST_OWNERSHIP) / 0.30 * 100, 0, 100)

    if kind == "eps_growth":
        # Above +15 %; perfect at +50 %+
        return _clamp((value - MIN_EPS_GROWTH) / 0.35 * 100, 0, 100)

    if kind == "rsi":
        # Below 40 is ideal; perfect score at RSI = 20
        if value > RSI_UPPER:
            return 0.0
        return _clamp((RSI_UPPER - value) / 20.0 * 100, 0, 100)

    if kind == "price_vs_200ma":
        # Ideally at or just above 200MA; penalise too far above or below
        if abs(value) > PRICE_VS_200MA_RANGE:
            return max(0.0, 100 - abs(value) / PRICE_VS_200MA_RANGE * 50)
        return 100 - abs(value) / PRICE_VS_200MA_RANGE * 30

    if kind == "bb_position":
        # value = (price - lower_band) / (upper_band - lower_band)  [0 = lower, 1 = upper]
        # Best score when touching lower band (≈ 0)
        return _clamp((1 - value) * 100, 0, 100) if value <= 0.5 else 0.0

    if kind == "delta":
        # Best in [0.70, 0.85]; penalise outside
        if DELTA_MIN <= value <= DELTA_MAX:
            mid = (DELTA_MIN + DELTA_MAX) / 2
            return 100 - abs(value - mid) / (0.075) * 40
        dist = min(abs(value - DELTA_MIN), abs(value - DELTA_MAX))
        return _clamp(100 - dist / 0.10 * 100, 0, 100)

    if kind == "iv_percentile":
        # Below 30; perfect at 0
        if value > MAX_IV_PERCENTILE:
            return 0.0
        return _clamp((MAX_IV_PERCENTILE - value) / MAX_IV_PERCENTILE * 100, 0, 100)

    return 0.0


# Weights for each parameter in the composite score
WEIGHTS = {
    "market_cap":     0.08,
    "avg_volume":     0.06,
    "inst_ownership": 0.08,
    "eps_growth":     0.10,
    "rsi":            0.12,
    "price_vs_200ma": 0.10,
    "bb_position":    0.10,
    "delta":          0.20,
    "iv_percentile":  0.16,
}
assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-6, "Weights must sum to 1"


def compute_composite_score(scores: dict) -> float:
    total = sum(WEIGHTS[k] * scores.get(k, 0) for k in WEIGHTS)
    return round(total, 2)


# ─────────────────────────────────────────────
# FUNDAMENTAL DATA
# ─────────────────────────────────────────────

def fetch_fundamental_data(ticker: yf.Ticker) -> dict:
    """Pulls fundamental metrics from yfinance with safe fallbacks."""
    info = ticker.info or {}

    market_cap    = info.get("marketCap", None)
    avg_volume    = info.get("averageVolume", None)
    inst_own      = info.get("heldPercentInstitutions", None)

    # EPS growth: compare trailing EPS to a year ago
    eps_growth = None
    try:
        earnings = ticker.earnings_history
        if earnings is not None and len(earnings) >= 4:
            # Quarterly EPS – compare most-recent 4Q sum vs prior 4Q sum
            recent_4q = earnings["epsActual"].iloc[:4].sum()
            prior_4q  = earnings["epsActual"].iloc[4:8].sum()
            if prior_4q and prior_4q != 0:
                eps_growth = (recent_4q - prior_4q) / abs(prior_4q)
    except Exception:
        pass

    if eps_growth is None:
        eps_growth = info.get("earningsGrowth", None)

    return {
        "market_cap":     market_cap,
        "avg_volume":     avg_volume,
        "inst_ownership": inst_own,
        "eps_growth":     eps_growth,
    }


# ─────────────────────────────────────────────
# MAIN ANALYSIS FUNCTION
# ─────────────────────────────────────────────

def analyze_ticker(symbol: str) -> list[dict]:
    """
    Fetches data for one ticker and returns a list of option records,
    each containing all screening metrics, greeks, and a composite score.
    Returns [] on any failure.
    """
    print(f"  Analysing {symbol}...")
    results = []

    try:
        ticker = yf.Ticker(symbol)

        # ── Price history ──────────────────────────────────────────────
        hist = ticker.history(period="1y")
        if hist.empty or len(hist) < 30:
            print(f"    [SKIP] Insufficient price history for {symbol}")
            return []
        close = hist["Close"]
        current_price = float(close.iloc[-1])

        # ── Technical indicators ───────────────────────────────────────
        rsi_val    = calculate_rsi(close)
        bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(close)
        ma_200     = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else float(close.mean())
        bb_width   = bb_upper - bb_lower if (bb_upper - bb_lower) != 0 else 1
        bb_pos     = (current_price - bb_lower) / bb_width   # 0 = lower, 1 = upper
        price_vs_200ma = (current_price - ma_200) / ma_200   # signed %

        # ── Fundamentals ───────────────────────────────────────────────
        fundamentals = fetch_fundamental_data(ticker)

        # ── Options chain (longest expiry) ─────────────────────────────
        expirations = ticker.options
        if not expirations:
            print(f"    [SKIP] No options data for {symbol}")
            return []

        expiry = expirations[-1]   # Furthest-dated expiry = LEAP
        chain  = ticker.option_chain(expiry)
        calls  = chain.calls.copy()

        if calls.empty:
            print(f"    [SKIP] No calls at expiry {expiry} for {symbol}")
            return []

        # Days to expiration
        exp_date = datetime.strptime(expiry, "%Y-%m-%d").date()
        dte      = max((exp_date - date.today()).days, 1)
        T        = dte / 365.0

        # IV percentile (compute once per ticker using mid-chain IV)
        mid_iv         = float(calls["impliedVolatility"].median())
        iv_percentile  = calculate_iv_percentile(symbol, mid_iv)

        # ── Score fundamentals & technicals (same for all options) ─────
        param_scores_base = {
            "market_cap":     score_parameter(fundamentals["market_cap"],    "market_cap"),
            "avg_volume":     score_parameter(fundamentals["avg_volume"],     "avg_volume"),
            "inst_ownership": score_parameter(fundamentals["inst_ownership"], "inst_ownership"),
            "eps_growth":     score_parameter(fundamentals["eps_growth"],     "eps_growth"),
            "rsi":            score_parameter(rsi_val,                        "rsi"),
            "price_vs_200ma": score_parameter(price_vs_200ma,                 "price_vs_200ma"),
            "bb_position":    score_parameter(bb_pos,                         "bb_position"),
        }

        # ── Process each call option ────────────────────────────────────
        for _, opt in calls.iterrows():
            strike = float(opt["strike"])
            iv     = float(opt["impliedVolatility"])

            # Prefer mid-market price; fall back to lastPrice
            bid  = float(opt.get("bid", 0) or 0)
            ask  = float(opt.get("ask", 0) or 0)
            prem = (bid + ask) / 2 if (bid + ask) > 0 else float(opt.get("lastPrice", 0) or 0)
            if prem <= 0:
                continue

            # ── Greeks (Black-Scholes) ───────────────────────────────
            delta = black_scholes_delta(current_price, strike, T, RISK_FREE_RATE, iv)
            theta = black_scholes_theta(current_price, strike, T, RISK_FREE_RATE, iv)

            if math.isnan(delta):
                continue

            # ── Additional metrics ───────────────────────────────────
            required_move   = prem / delta if delta != 0 else float("nan")
            leverage_ratio  = delta * (current_price / prem) if prem != 0 else float("nan")
            breakeven_price = strike + 2 * prem

            # Per-option IV-percentile score uses the option's own IV
            opt_iv_pct  = calculate_iv_percentile(symbol, iv) if abs(iv - mid_iv) > 0.01 else iv_percentile

            # ── Score option-specific parameters ─────────────────────
            param_scores = {
                **param_scores_base,
                "delta":        score_parameter(delta,       "delta"),
                "iv_percentile": score_parameter(opt_iv_pct, "iv_percentile"),
            }
            composite = compute_composite_score(param_scores)

            results.append({
                # Identification
                "ticker":        symbol,
                "expiry":        expiry,
                "strike":        round(strike, 2),
                "contract":      opt.get("contractSymbol", ""),

                # Stock metrics
                "current_price": round(current_price, 2),
                "market_cap":    fundamentals["market_cap"],
                "avg_volume":    fundamentals["avg_volume"],
                "inst_ownership": round(fundamentals["inst_ownership"] * 100, 1) if fundamentals["inst_ownership"] else None,
                "eps_growth_pct": round(fundamentals["eps_growth"] * 100, 1) if fundamentals["eps_growth"] else None,

                # Technical metrics
                "rsi":              round(rsi_val, 2),
                "price_vs_200ma_pct": round(price_vs_200ma * 100, 2),
                "bb_lower":         round(bb_lower, 2),
                "bb_middle":        round(bb_middle, 2),
                "bb_upper":         round(bb_upper, 2),
                "bb_position_pct":  round(bb_pos * 100, 1),

                # Option metrics
                "premium":          round(prem, 2),
                "implied_volatility_pct": round(iv * 100, 2),
                "iv_percentile":    round(opt_iv_pct, 1),
                "dte":              dte,

                # Greeks
                "delta":            round(delta, 4),
                "theta_daily":      round(theta, 4),

                # Derived / additional metrics
                "required_stock_move": round(required_move, 2) if not math.isnan(required_move) else None,
                "leverage_ratio":      round(leverage_ratio, 2) if not math.isnan(leverage_ratio) else None,
                "breakeven_price":     round(breakeven_price, 2),

                # Per-parameter scores (0–100)
                "scores": {k: round(v, 1) for k, v in param_scores.items()},

                # Composite score (0–100)
                "composite_score": composite,
            })

        # Sort by composite score descending
        results.sort(key=lambda x: x["composite_score"], reverse=True)
        print(f"    → {len(results)} options analysed for {symbol}")
        return results

    except Exception as e:
        print(f"    [ERROR] {symbol}: {e}")
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
    print("=" * 60)
    print("  LEAP CALL OPTIONS SCREENER")
    print(f"  Run date : {date.today()}")
    print(f"  Watchlist: {', '.join(WATCHLIST)}")
    print("=" * 60)

    all_results: list[dict] = []
    per_stock: dict[str, list[dict]] = {}

    for symbol in WATCHLIST:
        options = analyze_ticker(symbol)
        if options:
            per_stock[symbol] = options[:10]   # top 10 per stock
            all_results.extend(options)

    # Global top 10 across all tickers
    all_results.sort(key=lambda x: x["composite_score"], reverse=True)
    top_10_overall = all_results[:10]

    # ── Build output payloads ──────────────────────────────────────────
    run_meta = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "watchlist": WATCHLIST,
        "screening_parameters": {
            "min_market_cap_usd":     MIN_MARKET_CAP,
            "min_avg_volume_shares":  MIN_AVG_VOLUME,
            "min_inst_ownership_pct": MIN_INST_OWNERSHIP * 100,
            "min_eps_growth_pct":     MIN_EPS_GROWTH * 100,
            "rsi_upper":              RSI_UPPER,
            "price_vs_200ma_range":   f"±{PRICE_VS_200MA_RANGE*100}%",
            "bb_near_lower_thresh":   f"{BB_NEAR_LOWER_THRESH*100}%",
            "delta_range":            [DELTA_MIN, DELTA_MAX],
            "max_iv_percentile":      MAX_IV_PERCENTILE,
        },
        "score_weights": WEIGHTS,
    }

    top_10_per_stock_output = {
        "meta": run_meta,
        "top_10_per_stock": per_stock,
    }

    top_10_overall_output = {
        "meta": run_meta,
        "top_10_overall": top_10_overall,
    }

    full_output = {
        "meta": run_meta,
        "all_screened_options": all_results,
    }

    print("\nSaving output files...")
    save_json(top_10_per_stock_output, "top_10_per_stock.json")
    save_json(top_10_overall_output,   "top_10_overall.json")
    save_json(full_output,             "full_results.json")

    # ── Console summary ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  TOP 10 OVERALL (by composite score)")
    print("=" * 60)
    print(f"{'#':<3} {'Ticker':<7} {'Strike':>8} {'Expiry':<12} "
          f"{'Prem':>7} {'Delta':>7} {'IV%':>6} {'IV Pct':>7} "
          f"{'RSI':>5} {'Score':>7}")
    print("-" * 72)
    for i, opt in enumerate(top_10_overall, 1):
        print(
            f"{i:<3} {opt['ticker']:<7} {opt['strike']:>8.2f} {opt['expiry']:<12} "
            f"${opt['premium']:>6.2f} {opt['delta']:>7.4f} "
            f"{opt['implied_volatility_pct']:>5.1f}% {opt['iv_percentile']:>6.1f}% "
            f"{opt['rsi']:>5.1f} {opt['composite_score']:>7.1f}"
        )
    print("=" * 60)
    print("Done.\n")


if __name__ == "__main__":
    main()
