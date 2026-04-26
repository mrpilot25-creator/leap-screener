"""
LEAP Call Options Screener
==========================
Automatically fetches every ticker in the S&P 500, screens them against
fundamental, technical, and option-specific parameters, and outputs:

  - top_10_per_stock.json   → best 10 options for each ticker
  - top_10_overall.json     → best 10 options across all tickers
  - full_results.json       → every screened option with all metrics

Dependencies:
    pip install yfinance pandas numpy scipy requests lxml

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

RISK_FREE_RATE = 0.05
OUTPUT_DIR     = "."
DELAY_SECONDS  = 0.3

# ── Screening thresholds ──────────────────────────────────────────────
MIN_MARKET_CAP        = 2e9
MIN_AVG_VOLUME        = 1e6
MIN_INST_OWNERSHIP    = 0.50
MIN_EPS_GROWTH        = 0.15
RSI_UPPER             = 40
PRICE_VS_200MA_RANGE  = 0.05
BB_NEAR_LOWER_THRESH  = 0.10
DELTA_MIN             = 0.70
DELTA_MAX             = 0.85
MAX_IV_PERCENTILE     = 30


# ─────────────────────────────────────────────
# FETCH S&P 500 TICKERS
# ─────────────────────────────────────────────

def get_sp500_tickers() -> list:
    """
    Fetches the current S&P 500 constituents using three methods in order:

      1. Wikipedia via requests + lxml (most reliable)
      2. Wikipedia via requests + html.parser (fallback parser)
      3. Wikipedia REST API (JSON endpoint, no HTML parsing needed)

    Returns a sorted list of ticker symbols.
    """
    print("Fetching S&P 500 tickers...")

    url     = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }

    # ── Method 1: requests + lxml ─────────────────────────────────────
    try:
        resp = requests.get(url, headers=headers, timeout=20)
        resp.raise_for_status()
        tables  = pd.read_html(resp.text, flavor="lxml")
        df      = tables[0]
        tickers = _extract_tickers(df)
        if tickers:
            print(f"  ✓ Loaded {len(tickers)} tickers (method: requests + lxml)\n")
            return tickers
    except Exception as e:
        print(f"  Method 1 failed: {e}")

    # ── Method 2: requests + html.parser ─────────────────────────────
    try:
        resp = requests.get(url, headers=headers, timeout=20)
        resp.raise_for_status()
        tables  = pd.read_html(resp.text, flavor="html5lib")
        df      = tables[0]
        tickers = _extract_tickers(df)
        if tickers:
            print(f"  ✓ Loaded {len(tickers)} tickers (method: html5lib)\n")
            return tickers
    except Exception as e:
        print(f"  Method 2 failed: {e}")

    # ── Method 3: Wikipedia REST API (JSON, no HTML parsing) ──────────
    try:
        api_url = (
            "https://en.wikipedia.org/api/rest_v1/page/html/"
            "List_of_S%26P_500_companies"
        )
        resp    = requests.get(api_url, headers=headers, timeout=20)
        tables  = pd.read_html(resp.text, flavor="lxml")
        df      = tables[0]
        tickers = _extract_tickers(df)
        if tickers:
            print(f"  ✓ Loaded {len(tickers)} tickers (method: Wikipedia REST API)\n")
            return tickers
    except Exception as e:
        print(f"  Method 3 failed: {e}")

    # ── Method 4: slickcharts.com (independent source) ────────────────
    try:
        slick_url = "https://www.slickcharts.com/sp500"
        resp      = requests.get(slick_url, headers=headers, timeout=20)
        tables    = pd.read_html(resp.text)
        for t in tables:
            cols_lower = [str(c).lower() for c in t.columns]
            if "symbol" in cols_lower:
                col     = t.columns[cols_lower.index("symbol")]
                tickers = _clean_tickers(t[col].tolist())
                if len(tickers) > 400:
                    print(f"  ✓ Loaded {len(tickers)} tickers (method: slickcharts)\n")
                    return tickers
    except Exception as e:
        print(f"  Method 4 failed: {e}")

    # ── Final fallback: hardcoded top 100 S&P 500 constituents ────────
    print("  ⚠ All fetch methods failed. Using hardcoded top-100 fallback.\n")
    return [
        "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "GOOG", "TSLA",
        "BRK-B", "JPM", "LLY", "V", "UNH", "XOM", "MA", "JNJ", "PG",
        "AVGO", "HD", "MRK", "COST", "ABBV", "CVX", "KO", "PEP", "ADBE",
        "WMT", "BAC", "MCD", "CSCO", "CRM", "TMO", "ABT", "ACN", "LIN",
        "NFLX", "DHR", "TXN", "CMCSA", "VZ", "NEE", "PM", "RTX", "ORCL",
        "QCOM", "HON", "UPS", "AMGN", "BMY", "IBM", "GE", "CAT", "SBUX",
        "GS", "BLK", "SPGI", "AXP", "SYK", "PLD", "NOW", "ISRG", "DE",
        "MDLZ", "ADI", "REGN", "MMC", "PGR", "MO", "DUK", "SO", "CL",
        "ELV", "BSX", "ZTS", "CI", "WM", "MCO", "ITW", "EOG", "AON",
        "FI", "HUM", "NOC", "GD", "ANET", "APD", "KLAC", "LRCX", "SNPS",
        "CDNS", "PANW", "MCHP", "PAYX", "ADP", "MSCI", "ICE", "CME",
        "TDG", "CTAS",
    ]


def _extract_tickers(df: pd.DataFrame) -> list:
    """
    Finds the Symbol column in a DataFrame regardless of exact column name.
    """
    cols_lower = [str(c).lower() for c in df.columns]
    for keyword in ["symbol", "ticker"]:
        if keyword in cols_lower:
            col = df.columns[cols_lower.index(keyword)]
            return _clean_tickers(df[col].tolist())
    return []


def _clean_tickers(raw: list) -> list:
    """Cleans and deduplicates a list of ticker strings."""
    cleaned = []
    for t in raw:
        t = str(t).strip().replace(".", "-")
        if t and t != "nan" and len(t) <= 6:
            cleaned.append(t)
    return sorted(set(cleaned))


# ─────────────────────────────────────────────
# TECHNICAL INDICATORS
# ─────────────────────────────────────────────

def calculate_rsi(prices: pd.Series, period: int = 14) -> float:
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
) -> tuple:
    middle = prices.rolling(period).mean().iloc[-1]
    std    = prices.rolling(period).std().iloc[-1]
    return (
        float(middle + num_std * std),
        float(middle),
        float(middle - num_std * std),
    )


def calculate_iv_percentile(ticker_symbol: str, current_iv: float) -> float:
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
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return float("nan")
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    return float(norm.cdf(d1))


def black_scholes_theta(
    S: float, K: float, T: float, r: float, sigma: float
) -> float:
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
        "market_cap":     market_cap,
        "avg_volume":     avg_volume,
        "inst_ownership": inst_own,
        "eps_growth":     eps_growth,
    }


def passes_fundamental_screen(fund: dict) -> bool:
    mc = fund.get("market_cap")
    av = fund.get("avg_volume")
    io = fund.get("inst_ownership")
    eg = fund.get("eps_growth")

    if mc is None or mc < MIN_MARKET_CAP:   return False
    if av is None or av < MIN_AVG_VOLUME:   return False
    if io is None or io < MIN_INST_OWNERSHIP: return False
    if eg is None or eg < MIN_EPS_GROWTH:   return False
    return True


# ─────────────────────────────────────────────
# MAIN ANALYSIS FUNCTION
# ─────────────────────────────────────────────

def analyze_ticker(symbol: str, idx: int, total: int) -> list:
    print(f"  [{idx:>3}/{total}] {symbol:<8}", end="", flush=True)
    results = []

    try:
        ticker = yf.Ticker(symbol)

        # ── Fundamentals first ─────────────────────────────────────────
        fund = fetch_fundamental_data(ticker)
        if not passes_fundamental_screen(fund):
            mc = fund.get("market_cap")
            io = fund.get("inst_ownership")
            eg = fund.get("eps_growth")
            mc_s = f"${mc/1e9:.0f}B" if mc else "?"
            io_s = f"{io*100:.0f}%" if io else "?"
            eg_s = f"+{eg*100:.0f}%" if eg else "?"
            print(f" ✗  MCap={mc_s}  InstOwn={io_s}  EPSGrowth={eg_s}")
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

        param_scores_base = {
            "market_cap":     score_parameter(fund["market_cap"],    "market_cap"),
            "avg_volume":     score_parameter(fund["avg_volume"],     "avg_volume"),
            "inst_ownership": score_parameter(fund["inst_ownership"], "inst_ownership"),
            "eps_growth":     score_parameter(fund["eps_growth"],     "eps_growth"),
            "rsi":            score_parameter(rsi_val,                "rsi"),
            "price_vs_200ma": score_parameter(price_vs_200ma,         "price_vs_200ma"),
            "bb_position":    score_parameter(bb_pos,                 "bb_position"),
        }

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
                "delta":         score_parameter(delta,       "delta"),
                "iv_percentile": score_parameter(opt_iv_pct, "iv_percentile"),
            }
            composite = compute_composite_score(param_scores)

            results.append({
                "ticker":   symbol,
                "expiry":   expiry,
                "strike":   round(strike, 2),
                "contract": opt.get("contractSymbol", ""),

                "current_price":   round(current_price, 2),
                "market_cap":      fund["market_cap"],
                "avg_volume":      fund["avg_volume"],
                "inst_ownership":  round(fund["inst_ownership"] * 100, 1) if fund["inst_ownership"] else None,
                "eps_growth_pct":  round(fund["eps_growth"] * 100, 1)     if fund["eps_growth"]     else None,

                "rsi":                round(rsi_val, 2),
                "price_vs_200ma_pct": round(price_vs_200ma * 100, 2),
                "bb_lower":           round(bb_lower, 2),
                "bb_middle":          round(bb_middle, 2),
                "bb_upper":           round(bb_upper, 2),
                "bb_position_pct":    round(bb_pos * 100, 1),

                "premium":                round(prem, 2),
                "implied_volatility_pct": round(iv * 100, 2),
                "iv_percentile":          round(opt_iv_pct, 1),
                "dte":                    dte,

                "delta":       round(delta, 4),
                "theta_daily": round(theta, 4),

                "required_stock_move": round(required_move, 2)  if not math.isnan(required_move)  else None,
                "leverage_ratio":      round(leverage_ratio, 2) if not math.isnan(leverage_ratio) else None,
                "breakeven_price":     round(breakeven, 2),

                "scores":          {k: round(v, 1) for k, v in param_scores.items()},
                "composite_score": composite,
            })

        results.sort(key=lambda x: x["composite_score"], reverse=True)
        mc_s = f"${fund['market_cap']/1e9:.0f}B" if fund["market_cap"] else "?"
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
    print("=" * 65 + "\n")

    watchlist = get_sp500_tickers()
    total     = len(watchlist)

    print(f"Screening {total} tickers...\n")

    all_results    = []
    per_stock      = {}
    passed_tickers = []

    for idx, symbol in enumerate(watchlist, 1):
        options = analyze_ticker(symbol, idx, total)
        if options:
            passed_tickers.append(symbol)
            per_stock[symbol] = options[:10]
            all_results.extend(options)

    all_results.sort(key=lambda x: x["composite_score"], reverse=True)
    top_10_overall = all_results[:10]

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
    print("Saving output files...")

    save_json({"meta": run_meta, "top_10_per_stock":     per_stock},      "top_10_per_stock.json")
    save_json({"meta": run_meta, "top_10_overall":       top_10_overall}, "top_10_overall.json")
    save_json({"meta": run_meta, "all_screened_options": all_results},    "full_results.json")

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
