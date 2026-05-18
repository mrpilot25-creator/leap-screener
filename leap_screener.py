# LEAP Call Options Screener + Walk-Forward Backtesting Suite
# ===========================================================
# Screens a manual watchlist of stocks for the best LEAP call options,
# then runs a 5-year walk-forward backtest on the top candidates.
#
# Outputs:
# top_20_per_stock.json   - best 20 options for each ticker
# top_20_overall.json     - best 20 options across all tickers
# full_results.json       - every screened option with all metrics
# backtest_results.json   - walk-forward backtest report
# forward_projections.json- forward price projections to expiry
#
# Install: pip install yfinance pandas numpy scipy
# Run:     python leap_screener.py

import json
import math
import time
import warnings
from datetime import datetime, date, timedelta
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
warnings.filterwarnings("ignore")

# ------------------------------------------------------------------
# WATCHLIST - Edit this list with your tickers
# ------------------------------------------------------------------
WATCHLIST = [
    "A", "AAPL", "ABBV", "ABNB", "ABT", "ACGL", "ACN", "ADBE", "ADI", "ADM", "ADP", "ADSK", "AEE", "AEP", "AES", "AFL", "AIG", "AIZ", "AJG", "AKAM", 
    "ALB", "ALGN", "ALL", "ALLE", "AMAT", "AMCR", "AMD", "AME", "AMGN", "AMP", "AMT", "AMZN", "ANET", "ANSS", "AON", "AOS", "APA", "APD", "APH", "APO", 
    "ARE", "ATO", "AVB", "AVGO", "AVY", "AWK", "AXON", "AXP", "AYI", "AZO", "BA", "BAC", "BALL", "BAX", "BBWI", "BBY", "BDX", "BEN", "BF-B", "BG", 
    "BIIB", "BIO", "BK", "BKNG", "BKR", "BLK", "BMY", "BR", "BRK-B", "BSX", "BWA", "BX", "BXP", "C", "CAG", "CAH", "CARR", "CAT", "CB", "CBOE", 
    "CBRE", "CCI", "CCL", "CDNS", "CDW", "CE", "CEG", "CF", "CFG", "CHD", "CHRW", "CHTR", "CI", "CINF", "CL", "CLX", "CMA", "CMCSA", "CME", "CMG", 
    "CMI", "CMS", "CNC", "CNP", "COF", "COO", "COP", "COR", "COST", "CPAY", "CPB", "CPRT", "CPT", "CRL", "CRM", "CSCO", "CSGP", "CSX", "CTAS", "CTRA", 
    "CTSH", "CTVA", "CVS", "CVX", "CZR", "D", "DAL", "DAY", "DD", "DE", "DECK", "DELL", "DFS", "DG", "DGX", "DHI", "DHR", "DIS", "DLR", "DLTR", 
    "DOC", "DOV", "DOW", "DPZ", "DRI", "DTE", "DUK", "DVA", "DVN", "DXCM", "EA", "EBAY", "ECL", "ED", "EFX", "EG", "EIX", "EL", "ELV", "EMN", 
    "EMR", "ENPH", "EOG", "EPAM", "EQIX", "EQR", "EQT", "ES", "ESS", "ETN", "ETR", "ETSY", "EVRG", "EW", "EXC", "EXPD", "EXPE", "EXR", "F", "FANG", 
    "FAST", "FCX", "FDS", "FDX", "FE", "FFIV", "FI", "FICO", "FIS", "FITB", "FMC", "FOX", "FOXA", "FRT", "FSLR", "FTNT", "FTV", "GD", "GE", "GEF", 
    "GEHC", "GEN", "GESV", "GEV", "GILD", "GIS", "GL", "GLW", "GM", "GNRC", "GOOG", "GOOGL", "GPC", "GPN", "GRMN", "GS", "GWRE", "GWW", "HAL", "HAS", 
    "HBAN", "HCA", "HD", "HES", "HIG", "HII", "HLT", "HOLX", "HON", "HPE", "HPQ", "HRL", "HSIC", "HST", "HSY", "HUBB", "HUM", "HWM", "IBM", "ICE", 
    "IDXX", "IEX", "IFF", "ILMN", "INCY", "INTC", "INTU", "INVH", "IP", "IPG", "IQV", "IR", "IRM", "ISRG", "IT", "ITW", "IVZ", "J", "JBHT", "JBL", 
    "JCI", "JKHY", "JNJ", "JNPR", "JPM", "K", "KDP", "KEY", "KEYS", "KHC", "KIM", "KLAC", "KMB", "KMI", "KMX", "KO", "KR", "KVUE", "L", "LDOS", 
    "LEN", "LH", "LHX", "LIN", "LKQ", "LLY", "LMT", "LNT", "LOW", "LRCX", "LULU", "LUV", "LVS", "LW", "LYB", "LYV", "MA", "MAA", "MAR", "MAS", 
    "MCD", "MCHP", "MCK", "MCO", "MDLZ", "MDT", "MET", "META", "MGM", "MHK", "MKC", "MKTX", "MLM", "MMC", "MMM", "MNST", "MO", "MOH", "MOS", "MPC", 
    "MPWR", "MRK", "MRNA", "MS", "MSI", "MSFT", "MTB", "MTCH", "MTD", "MU", "NCLH", "NDAQ", "NDSN", "NEE", "NEM", "NFLX", "NI", "NKE", "NOC", "NOW", 
    "NRG", "NSC", "NTRS", "NUE", "NVDA", "NVR", "NWS", "NWSA", "NXPI", "O", "ODFL", "OKE", "OMC", "ON", "ORCL", "ORLY", "OTIS", "OXY", "PANW", "PARA", 
    "PAYC", "PAYX", "PCAR", "PCG", "PEG", "PEP", "PFE", "PFG", "PG", "PGR", "PH", "PHM", "PKG", "PLD", "PLTR", "PM", "PNC", "PNW", "POOL", "PPG", 
    "PPL", "PRU", "PSA", "PSX", "PTC", "PWR", "PXD", "PYPL", "QCOM", "QRVO", "RCL", "REG", "REGN", "RF", "RHI", "RJF", "RL", "RMD", "ROK", "ROL", 
    "ROP", "ROST", "RSG", "RTX", "RVTY", "SBAC", "SBUX", "SCHW", "SHW", "SJM", "SNA", "SNPS", "SO", "SPG", "SPGI", "SRE", "STE", "STLD", "STT", "STX", 
    "SYK", "SYY", "T", "TAP", "TDG", "TDY", "TECH", "TEL", "TER", "TFC", "TFX", "TGT", "TJX", "TMO", "TMUS", "TPR", "TRGP", "TRMB", "TROW", "TRV", 
    "TSCO", "TSLA", "TSN", "TT", "TTWO", "TXN", "TXT", "TYL", "UAL", "UDR", "UHS", "ULTA", "UNH", "UNP", "UPS", "URI", "USB", "V", "VICI", "VLO", 
    "VMC", "VRSK", "VRSN", "VRTX", "VST", "VTR", "VZ", "WAB", "WAT", "WBA", "WBD", "WDC", "WEC", "WELL", "WFC", "WHR", "WM", "WMB", "WMT", "WRB", 
    "WST", "WTW", "WY", "WYNN", "XEL", "XOM", "XYL", "YUM", "ZBH", "ZBRA", "ZTS"
]

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------
RISK_FREE_RATE = 0.0388          # 2-Year Treasury proxy
OUTPUT_DIR = "."
DELAY_SECONDS = 0.2

RUN_BACKTEST = True              # Set to False to skip the backtest module
RUN_FORWARD_PROJECTION = True    # Set to False to skip forward projections

# Backtest config
BT_TRAINING_YEARS   = 4
BT_VALIDATION_YEARS = 2
BT_N_SIMULATIONS    = 5500
BT_TOP_N_CANDIDATES = 999        # Backtest ALL passing tickers
BT_USE_ATM_STRIKE   = True       # Use ATM strike at historical entry date

FP_N_SIMULATIONS = 1000          # Monte Carlo paths for forward projection

# Fundamental thresholds
MIN_MARKET_CAP     = 25e9
MIN_AVG_VOLUME     = 1e6
MIN_INST_OWNERSHIP = 0.50
MIN_EPS_GROWTH     = 0.15
MIN_REVENUE_GROWTH = 0.08
MIN_EBITDA_MARGIN  = 0.15

# Technical thresholds
RSI_UPPER            = 40
PRICE_VS_200MA_RANGE = 0.05
BB_NEAR_LOWER_THRESH = 0.10

# Option thresholds
# ATM LEAP calls typically carry delta ~0.50-0.65 (1-2yr expiry).
# Deep ITM (0.70-0.85) adds intrinsic value to the premium unnecessarily.
# Targeting ATM minimises premium paid while maximising leverage per dollar.
DELTA_MIN       = 0.45
DELTA_MAX       = 0.65
MAX_IV_PERCENTILE = 30

# ------------------------------------------------------------------
# TECHNICAL INDICATORS
# ------------------------------------------------------------------
def calculate_rsi(prices, period=14):
    delta    = prices.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    return float((100 - (100 / (1 + rs))).iloc[-1])


def calculate_bollinger_bands(prices, period=20, num_std=2.0):
    middle = prices.rolling(period).mean().iloc[-1]
    std    = prices.rolling(period).std().iloc[-1]
    return (
        float(middle + num_std * std),
        float(middle),
        float(middle - num_std * std),
    )


def calculate_iv_percentile(close_prices, current_iv):
    try:
        if len(close_prices) < 30:
            return 50.0
        log_rets = np.log(close_prices / close_prices.shift(1)).dropna()
        roll_vol = log_rets.rolling(30).std() * np.sqrt(252)
        return round(float(np.mean(roll_vol.dropna() < current_iv) * 100), 1)
    except Exception:
        return 50.0


# ------------------------------------------------------------------
# EXTENDED TECHNICAL INDICATORS (ABCD Strategy Framework)
# ------------------------------------------------------------------
def resample_to_weekly(hist):
    weekly = hist.resample("W").agg({
        "Open":   "first",
        "High":   "max",
        "Low":    "min",
        "Close":  "last",
        "Volume": "sum",
    }).dropna()
    return weekly


def calculate_macd(prices, fast=12, slow=26, signal=9):
    try:
        ema_fast    = prices.ewm(span=fast,   adjust=False).mean()
        ema_slow    = prices.ewm(span=slow,   adjust=False).mean()
        macd_line   = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram   = macd_line - signal_line
        macd_now    = float(macd_line.iloc[-1])
        signal_now  = float(signal_line.iloc[-1])
        macd_prev   = float(macd_line.iloc[-2])   if len(macd_line)   > 1 else macd_now
        signal_prev = float(signal_line.iloc[-2]) if len(signal_line) > 1 else signal_now
        bullish_crossover  = (macd_prev < signal_prev) and (macd_now >= signal_now)
        macd_above_signal  = macd_now > signal_now
        return {
            "macd":               round(macd_now,              4),
            "signal":             round(signal_now,            4),
            "histogram":          round(float(histogram.iloc[-1]), 4),
            "bullish_crossover":  bullish_crossover,
            "macd_above_signal":  macd_above_signal,
        }
    except Exception:
        return {
            "macd": None, "signal": None, "histogram": None,
            "bullish_crossover": False, "macd_above_signal": False,
        }


def calculate_stoch_rsi(prices, rsi_period=14, stoch_period=14, k=3, d=3):
    try:
        delta    = prices.diff()
        gain     = delta.clip(lower=0)
        loss     = -delta.clip(upper=0)
        avg_gain = gain.ewm(com=rsi_period - 1, min_periods=rsi_period).mean()
        avg_loss = loss.ewm(com=rsi_period - 1, min_periods=rsi_period).mean()
        rs       = avg_gain / avg_loss.replace(0, np.nan)
        rsi_ser  = 100 - (100 / (1 + rs))
        rsi_min  = rsi_ser.rolling(stoch_period).min()
        rsi_max  = rsi_ser.rolling(stoch_period).max()
        rsi_rng  = (rsi_max - rsi_min).replace(0, np.nan)
        raw_k    = ((rsi_ser - rsi_min) / rsi_rng * 100)
        pct_k    = raw_k.rolling(k).mean()
        pct_d    = pct_k.rolling(d).mean()
        k_now    = float(pct_k.iloc[-1])
        d_now    = float(pct_d.iloc[-1])
        k_prev   = float(pct_k.iloc[-2]) if len(pct_k) > 1 else k_now
        d_prev   = float(pct_d.iloc[-2]) if len(pct_d) > 1 else d_now
        bullish_cross = (k_prev < d_prev) and (k_now >= d_now)
        oversold      = k_now < 20
        return {
            "stoch_k":      round(k_now, 2),
            "stoch_d":      round(d_now, 2),
            "bullish_cross": bullish_cross,
            "oversold":      oversold,
        }
    except Exception:
        return {"stoch_k": None, "stoch_d": None,
                "bullish_cross": False, "oversold": False}


def calculate_obv(close, volume):
    try:
        direction = np.sign(close.diff().fillna(0))
        obv       = (direction * volume).cumsum()
        obv_now   = float(obv.iloc[-1])
        obv_20    = float(obv.rolling(20).mean().iloc[-1])
        price_low_20 = float(close.tail(20).min())
        price_low_10 = float(close.tail(10).min())
        obv_low_20   = float(obv.tail(20).min())
        obv_low_10   = float(obv.tail(10).min())
        bullish_divergence = (price_low_10 <= price_low_20) and (obv_low_10 > obv_low_20)
        obv_rising = obv_now > obv_20
        return {
            "obv":                round(obv_now, 0),
            "obv_rising":         obv_rising,
            "bullish_divergence": bullish_divergence,
        }
    except Exception:
        return {"obv": None, "obv_rising": False, "bullish_divergence": False}


def calculate_atr(high, low, close, period=14):
    try:
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low  - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        atr = float(tr.ewm(span=period, adjust=False).mean().iloc[-1])
        return round(atr, 4)
    except Exception:
        return None


def calculate_weekly_mas(weekly_close):
    try:
        ma50  = weekly_close.rolling(50).mean()
        ma200 = weekly_close.rolling(200).mean()
        price = float(weekly_close.iloc[-1])
        ma50_val  = float(ma50.iloc[-1])  if not ma50.isna().all()  else None
        ma200_val = float(ma200.iloc[-1]) if not ma200.isna().all() else None
        return {
            "ma50_weekly":       round(ma50_val,  2) if ma50_val  else None,
            "ma200_weekly":      round(ma200_val, 2) if ma200_val else None,
            "above_ma50_weekly": price > ma50_val  if ma50_val  else None,
            "above_ma200_weekly":price > ma200_val if ma200_val else None,
        }
    except Exception:
        return {"ma50_weekly": None, "ma200_weekly": None,
                "above_ma50_weekly": None, "above_ma200_weekly": None}


def detect_swing_points(prices, window=5):
    highs, lows = [], []
    arr    = prices.values
    labels = list(prices.index)
    for i in range(window, len(arr) - window):
        left  = arr[i - window: i]
        right = arr[i + 1: i + window + 1]
        label = str(labels[i])[:10]
        if arr[i] >= max(left) and arr[i] >= max(right):
            highs.append((i, float(arr[i]), label))
        if arr[i] <= min(left) and arr[i] <= min(right):
            lows.append((i, float(arr[i]), label))
    return highs, lows


def detect_abcd_pattern(weekly_close, current_price):
    try:
        if len(weekly_close) < 30:
            return {"status": "insufficient_data"}
        highs, lows = detect_swing_points(weekly_close, window=3)
        if len(lows) < 2 or len(highs) < 1:
            return {"status": "no_pattern_found"}
        best = None
        for li, (a_idx, a_price, a_date) in enumerate(lows[:-1]):
            b_candidates = [(i, p, d) for i, p, d in highs if i > a_idx]
            if not b_candidates:
                continue
            b_idx, b_price, b_date = max(b_candidates, key=lambda x: x[1])
            c_candidates = [(i, p, d) for i, p, d in lows
                            if i > b_idx and p > a_price]
            if not c_candidates:
                continue
            c_idx, c_price, c_date = c_candidates[-1]
            ab_dist = b_price - a_price
            bc_dist = b_price - c_price
            if ab_dist <= 0:
                continue
            bc_retracement = bc_dist / ab_dist
            pattern = {
                "A_price":            round(a_price, 2),
                "A_date":             a_date,
                "B_price":            round(b_price, 2),
                "B_date":             b_date,
                "C_price":            round(c_price, 2),
                "C_date":             c_date,
                "AB_distance":        round(ab_dist, 2),
                "BC_retracement_pct": round(bc_retracement * 100, 1),
                "D_targets": {
                    "1.272x": round(c_price + ab_dist * 1.272, 2),
                    "1.618x": round(c_price + ab_dist * 1.618, 2),
                    "2.000x": round(c_price + ab_dist * 2.000, 2),
                    "2.618x": round(c_price + ab_dist * 2.618, 2),
                    "4.236x": round(c_price + ab_dist * 4.236, 2),
                },
                "pattern_valid": True,
                "C_above_A":     c_price > a_price,
            }
            if best is None or c_idx > best[0]:
                best = (c_idx, pattern)
        if best is None:
            return {"status": "no_pattern_found"}
        _, result = best
        c_price  = result["C_price"]
        near_c   = abs(current_price - c_price) / c_price < 0.10
        result["price_near_C"]  = near_c
        result["current_price"] = round(current_price, 2)
        result["status"]        = "found"
        return result
    except Exception as e:
        return {"status": "error", "error": str(e)}


def leaps_entry_checklist(
    current_price, rsi_weekly, macd_data, obv_data,
    bb_pos, stoch_data, weekly_mas, iv_percentile,
    abcd_data, volume_data
):
    checks = {}
    checks["near_abcd_c"] = (
        abcd_data.get("status") == "found" and
        abcd_data.get("price_near_C", False)
    )
    checks["c_above_a"] = (
        abcd_data.get("status") == "found" and
        abcd_data.get("C_above_A", False)
    )
    checks["weekly_rsi_oversold"] = (
        rsi_weekly is not None and rsi_weekly < 40
    )
    checks["obv_bullish_divergence"] = obv_data.get("bullish_divergence", False)
    checks["at_lower_bollinger"]     = bb_pos < 0.15
    checks["macd_bullish"] = (
        macd_data.get("bullish_crossover", False) or
        macd_data.get("macd_above_signal", False)
    )
    checks["volume_capitulation"] = obv_data.get("obv_rising", False)
    above_50  = weekly_mas.get("above_ma50_weekly")
    above_200 = weekly_mas.get("above_ma200_weekly")
    checks["weekly_ma_bullish"] = (
        (above_50 is True) or (above_200 is True)
    )
    checks["iv_low"] = iv_percentile < 30
    confluence = sum(1 for v in checks.values() if v is True)
    return {
        "checklist":       checks,
        "confluence_score": confluence,
        "max_score":        9,
        "confluence_pct":   round(confluence / 9 * 100, 1),
    }


# ------------------------------------------------------------------
# BLACK-SCHOLES GREEKS
# ------------------------------------------------------------------
def bs_d1_d2(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return float("nan"), float("nan")
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return d1, d2


def black_scholes_price(S, K, T, r, sigma):
    d1, d2 = bs_d1_d2(S, K, T, r, sigma)
    if math.isnan(d1):
        return float("nan")
    return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)


def black_scholes_delta(S, K, T, r, sigma):
    d1, _ = bs_d1_d2(S, K, T, r, sigma)
    if math.isnan(d1):
        return float("nan")
    return float(norm.cdf(d1))


def black_scholes_theta(S, K, T, r, sigma):
    d1, d2 = bs_d1_d2(S, K, T, r, sigma)
    if math.isnan(d1):
        return float("nan")
    term1 = -(S * norm.pdf(d1) * sigma) / (2 * math.sqrt(T))
    term2 =  r * K * math.exp(-r * T) * norm.cdf(d2)
    return float((term1 - term2) / 365)


def black_scholes_rho(S, K, T, r, sigma):
    _, d2 = bs_d1_d2(S, K, T, r, sigma)
    if math.isnan(d2):
        return float("nan")
    return float(K * T * math.exp(-r * T) * norm.cdf(d2) / 100)


def black_scholes_gamma(S, K, T, r, sigma):
    d1, _ = bs_d1_d2(S, K, T, r, sigma)
    if math.isnan(d1):
        return float("nan")
    return float(norm.pdf(d1) / (S * sigma * math.sqrt(T)))


def black_scholes_vega(S, K, T, r, sigma):
    d1, _ = bs_d1_d2(S, K, T, r, sigma)
    if math.isnan(d1):
        return float("nan")
    return float(S * norm.pdf(d1) * math.sqrt(T) / 100)


# ------------------------------------------------------------------
# SCORING ENGINE
# ------------------------------------------------------------------
def clamp(val, lo, hi):
    return max(lo, min(hi, val))


def score_parameter(value, kind):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return 0.0
    if kind == "market_cap":
        return clamp((value - MIN_MARKET_CAP) / 8e9 * 100, 0, 100)
    if kind == "avg_volume":
        return clamp((value - MIN_AVG_VOLUME) / 9e6 * 100, 0, 100)
    if kind == "inst_ownership":
        return clamp((value - MIN_INST_OWNERSHIP) / 0.30 * 100, 0, 100)
    if kind == "eps_growth":
        return clamp((value - MIN_EPS_GROWTH) / 0.35 * 100, 0, 100)
    if kind == "revenue_growth":
        return clamp((value - MIN_REVENUE_GROWTH) / 0.32 * 100, 0, 100)
    if kind == "ebitda_margin":
        return clamp((value - MIN_EBITDA_MARGIN) / 0.30 * 100, 0, 100)
    if kind == "rsi":
        if value > RSI_UPPER:
            return 0.0
        return clamp((RSI_UPPER - value) / 20.0 * 100, 0, 100)
    if kind == "price_vs_200ma":
        if abs(value) > PRICE_VS_200MA_RANGE:
            return max(0.0, 100 - abs(value) / PRICE_VS_200MA_RANGE * 50)
        return 100 - abs(value) / PRICE_VS_200MA_RANGE * 30
    if kind == "bb_position":
        if value <= 0.5:
            return clamp((1 - value) * 100, 0, 100)
        return 0.0
    if kind == "delta":
        if DELTA_MIN <= value <= DELTA_MAX:
            mid = (DELTA_MIN + DELTA_MAX) / 2
            return 100 - abs(value - mid) / 0.075 * 40
        dist = min(abs(value - DELTA_MIN), abs(value - DELTA_MAX))
        return clamp(100 - dist / 0.10 * 100, 0, 100)
    if kind == "iv_percentile":
        if value > MAX_IV_PERCENTILE:
            return 0.0
        return clamp((MAX_IV_PERCENTILE - value) / MAX_IV_PERCENTILE * 100, 0, 100)
    return 0.0


# ------------------------------------------------------------------
# SCORING WEIGHTS
# ------------------------------------------------------------------
DEFAULT_WEIGHTS = {
    "market_cap":     0.06,
    "avg_volume":     0.05,
    "inst_ownership": 0.06,
    "eps_growth":     0.07,
    "revenue_growth": 0.06,
    "ebitda_margin":  0.06,
    "rsi":            0.12,
    "price_vs_200ma": 0.10,
    "bb_position":    0.10,
    "delta":          0.20,
    "iv_percentile":  0.12,
}

WEIGHTS_CONFIG_FILE  = OUTPUT_DIR + "/weights_config.json"
REQUIRED_WEIGHT_KEYS = list(DEFAULT_WEIGHTS.keys())


def normalise_weights(raw):
    total = sum(raw.values())
    if total <= 0:
        return dict(DEFAULT_WEIGHTS)
    return {k: round(v / total, 6) for k, v in raw.items()}


def load_weights():
    import os
    if os.path.exists(WEIGHTS_CONFIG_FILE):
        try:
            with open(WEIGHTS_CONFIG_FILE, "r") as f:
                config = json.load(f)
            raw     = config.get("weights", {})
            missing = [k for k in REQUIRED_WEIGHT_KEYS if k not in raw]
            if missing:
                print(" [WEIGHTS] Missing keys: " + str(missing) +
                      " - using defaults for those keys.")
                for k in missing:
                    raw[k] = DEFAULT_WEIGHTS[k]
            for k, v in raw.items():
                if not isinstance(v, (int, float)) or v < 0:
                    print(" [WEIGHTS] Invalid value for " + k +
                          " - resetting to default.")
                    raw[k] = DEFAULT_WEIGHTS[k]
            weights = normalise_weights(
                {k: raw[k] for k in REQUIRED_WEIGHT_KEYS}
            )
            print(" [WEIGHTS] Loaded from weights_config.json " +
                  " ".join(k + "=" + str(round(v * 100, 1)) + "%"
                           for k, v in weights.items()))
            return weights
        except Exception as e:
            print(" [WEIGHTS] Could not read weights_config.json (" +
                  str(e) + ") - using defaults.")
            return dict(DEFAULT_WEIGHTS)
    else:
        try:
            config = {
                "note": (
                    "Edit the weights values to change the scoring system. "
                    "Values are auto-normalised to sum to 1.0 so you can use "
                    "any positive numbers. Re-run the screener after saving. "
                    "You can also adjust weights via the Base44 dashboard."
                ),
                "last_modified": str(date.today()),
                "weights": DEFAULT_WEIGHTS,
                "parameter_descriptions": {
                    "market_cap":     "Company size - larger scores higher",
                    "avg_volume":     "Daily trading liquidity",
                    "inst_ownership": "% held by professional investors",
                    "eps_growth":     "Earnings per share growth rate",
                    "revenue_growth": "Revenue growth rate",
                    "ebitda_margin":  "Operating profitability margin",
                    "rsi":            "Technical - oversold entry signal",
                    "price_vs_200ma": "Technical - near long-term trend",
                    "bb_position":    "Technical - Bollinger Band entry",
                    "delta":          "Option - price sensitivity (0.70-0.85)",
                    "iv_percentile":  "Option - cheap implied volatility",
                },
            }
            with open(WEIGHTS_CONFIG_FILE, "w") as f:
                json.dump(config, f, indent=2)
            print(" [WEIGHTS] Created weights_config.json with defaults.")
        except Exception as e:
            print(" [WEIGHTS] Could not create weights_config.json: " + str(e))
        return dict(DEFAULT_WEIGHTS)


WEIGHTS = load_weights()


def compute_composite_score(scores):
    return round(sum(WEIGHTS[k] * scores.get(k, 0) for k in WEIGHTS), 2)


# ------------------------------------------------------------------
# FUNDAMENTAL DATA
# ------------------------------------------------------------------
def fetch_eps_growth(ticker, info):
    try:
        eh = ticker.earnings_history
        if eh is not None and len(eh) >= 8:
            recent = eh["epsActual"].iloc[:4].sum()
            prior  = eh["epsActual"].iloc[4:8].sum()
            if prior and abs(prior) > 0.001:
                return (recent - prior) / abs(prior)
    except Exception:
        pass
    for key in ["earningsGrowth", "earningsQuarterlyGrowth"]:
        val = info.get(key)
        if val is not None:
            return float(val)
    try:
        trailing = info.get("trailingEps")
        forward  = info.get("forwardEps")
        if trailing and forward and abs(trailing) > 0.001:
            return (forward - trailing) / abs(trailing)
    except Exception:
        pass
    val = info.get("revenueGrowth")
    if val is not None:
        return float(val)
    return None


def fetch_ebitda_margin(info):
    try:
        ebitda  = info.get("ebitda")
        revenue = info.get("totalRevenue")
        if ebitda is not None and revenue and revenue > 0:
            return float(ebitda) / float(revenue)
    except Exception:
        pass
    val = info.get("ebitdaMargins")
    if val is not None:
        return float(val)
    return None


def get_tv_exchange(yf_exchange, yf_full_name):
    code = (yf_exchange or "").upper()
    name = (yf_full_name or "").upper()
    nasdaq_codes = {"NMS", "NGM", "NCM", "NASDAQ", "NASDAQGS",
                    "NASDAQGM", "NASDAQCM"}
    nyse_codes   = {"NYQ", "NYSE", "PCX", "NYSEARCA", "NYSE ARCA"}
    amex_codes   = {"ASE", "AMEX", "NYSEAMERICAN", "NYSE AMERICAN"}
    if code in nasdaq_codes or "NASDAQ" in name:
        return "NASDAQ"
    if code in nyse_codes or "NYSE" in name:
        return "NYSE"
    if code in amex_codes or "AMEX" in name:
        return "AMEX"
    if "BATS" in code or "BATS" in name:
        return "BATS"
    return ""


def fetch_stock_metadata(ticker, info):
    beta      = info.get("beta")
    short_pct = info.get("shortPercentOfFloat")
    if short_pct is None:
        shares_short  = info.get("sharesShort")
        float_shares  = info.get("floatShares")
        if shares_short and float_shares and float_shares > 0:
            short_pct = shares_short / float_shares
    analyst_target = info.get("targetMeanPrice")
    analyst_low    = info.get("targetLowPrice")
    analyst_high   = info.get("targetHighPrice")
    analyst_count  = info.get("numberOfAnalystOpinions")
    current_price  = info.get("currentPrice") or info.get("regularMarketPrice")
    analyst_upside = None
    if analyst_target and current_price and current_price > 0:
        analyst_upside = round(
            (analyst_target - current_price) / current_price * 100, 1
        )
    sector   = info.get("sector",   "")
    industry = info.get("industry", "")
    earnings_date     = None
    avg_earnings_move = None
    try:
        cal = ticker.calendar
        if cal is not None:
            if isinstance(cal, dict):
                ed = cal.get("Earnings Date")
                if ed and len(ed) > 0:
                    earnings_date = str(ed[0])[:10]
            elif hasattr(cal, "loc"):
                if "Earnings Date" in cal.index:
                    ed = cal.loc["Earnings Date"]
                    earnings_date = (
                        str(ed.iloc[0])[:10] if hasattr(ed, "iloc") else str(ed)[:10]
                    )
    except Exception:
        pass
    try:
        hist_q = ticker.quarterly_earnings
        if hist_q is not None and len(hist_q) >= 2:
            hist = ticker.history(period="3y")
            if not hist.empty:
                close = hist["Close"]
                moves = []
                for dt_idx in hist_q.index:
                    try:
                        loc = close.index.get_indexer([dt_idx], method="nearest")[0]
                        if 0 < loc < len(close):
                            pct = (
                                abs(float(close.iloc[loc]) - float(close.iloc[loc - 1]))
                                / float(close.iloc[loc - 1]) * 100
                            )
                            if pct < 50:
                                moves.append(pct)
                    except Exception:
                        pass
                if moves:
                    avg_earnings_move = round(float(sum(moves) / len(moves)), 1)
    except Exception:
        pass
    fcf          = info.get("freeCashflow")
    debt_equity  = info.get("debtToEquity")
    return {
        "beta":                round(float(beta),       2) if beta      else None,
        "short_interest_pct":  round(float(short_pct) * 100, 1) if short_pct else None,
        "analyst_target":      round(float(analyst_target), 2) if analyst_target else None,
        "analyst_low":         round(float(analyst_low),    2) if analyst_low    else None,
        "analyst_high":        round(float(analyst_high),   2) if analyst_high   else None,
        "analyst_count":       analyst_count,
        "analyst_upside_pct":  analyst_upside,
        "next_earnings_date":  earnings_date,
        "avg_earnings_move_pct": avg_earnings_move,
        "sector":              sector,
        "industry":            industry,
        "free_cashflow":       fcf,
        "debt_to_equity":      round(float(debt_equity), 2) if debt_equity else None,
    }


def fetch_fundamental_data(ticker):
    info     = ticker.info or {}
    exchange = get_tv_exchange(
        info.get("exchange",         ""),
        info.get("fullExchangeName", ""),
    )
    metadata = fetch_stock_metadata(ticker, info)
    return {
        "market_cap":     info.get("marketCap"),
        "avg_volume":     info.get("averageVolume"),
        "inst_ownership": info.get("heldPercentInstitutions"),
        "eps_growth":     fetch_eps_growth(ticker, info),
        "revenue_growth": info.get("revenueGrowth"),
        "ebitda_margin":  fetch_ebitda_margin(info),
        "exchange":       exchange,
        **metadata,
    }


def passes_fundamental_screen(fund):
    mc = fund.get("market_cap")
    av = fund.get("avg_volume")
    io = fund.get("inst_ownership")
    eg = fund.get("eps_growth")
    rg = fund.get("revenue_growth")
    em = fund.get("ebitda_margin")
    if mc is None or mc < MIN_MARKET_CAP:
        label = "no data" if mc is None else ("$" + str(round(mc / 1e9, 1)) + "B < $25B")
        return False, "MarketCap " + label
    if av is not None and av < MIN_AVG_VOLUME:
        return False, "AvgVol " + str(round(av / 1e6, 1)) + "M < 1M"
    if io is not None and io < MIN_INST_OWNERSHIP:
        return False, "InstOwn " + str(round(io * 100)) + "% < 50%"
    if eg is not None and eg < MIN_EPS_GROWTH:
        return False, "EPSGrowth " + str(round(eg * 100, 1)) + "% < 15%"
    if rg is not None and rg < MIN_REVENUE_GROWTH:
        return False, "RevenueGrowth " + str(round(rg * 100, 1)) + "% < 8%"
    if em is not None and em < MIN_EBITDA_MARGIN:
        return False, "EBITDAMargin " + str(round(em * 100, 1)) + "% < 15%"
    return True, ""


# ------------------------------------------------------------------
# BACKTEST ENGINE
# ------------------------------------------------------------------
class BacktestEngine:
    """
    Walk-Forward Backtesting Suite for LEAP Options Strategy.

    Pipeline:
      1. Fetch price + dividend data for total_years.
      2. Split into training and validation windows.
      3. Estimate drift (mu) from training data with long-run anchor.
      4. Estimate sigma via EWMA + mean-reverting blend + coverage correction.
      5. Run vectorized Monte Carlo GBM over the validation horizon.
      6. Compare simulated paths to actual validation prices.
      7. Calculate Greek attribution at multiple checkpoints.
      8. Produce Residual Variance Report and EV.

    ATM Mode (BT_USE_ATM_STRIKE=True):
      The strike is overridden to the stock price at the historical entry
      date (train/validation split), so every backtest starts with a
      realistic ATM option rather than applying today's screener strike
      to a historical period where it was deeply OTM.
    """

    def __init__(self, symbol, strike, entry_score,
                 training_years=3, validation_years=2,
                 n_simulations=1000, risk_free_rate=0.0388,
                 use_atm_strike=False):
        self.symbol          = symbol
        self.screener_strike = strike     # original screener strike (kept for reference)
        self.strike          = strike     # may be overridden in load_data if ATM mode
        self.entry_score     = entry_score
        self.use_atm_strike  = use_atm_strike
        self.training_years  = training_years
        self.validation_years= validation_years
        self.n_sims          = n_simulations
        self.r               = risk_free_rate
        self.total_years     = training_years + validation_years
        # Populated by load_data()
        self.train_close = None
        self.val_close   = None
        self.entry_price = None
        self.div_yield   = None
        # Populated by calibrate()
        self.mu             = None
        self.sigma_train    = None
        self.sigma_val      = None
        self.sigma_blended  = None
        # Populated by run_monte_carlo()
        self.paths = None    # shape: (n_sims, val_days)

    # -- 1. Data Loading ------------------------------------------
    def load_data(self):
        ticker = yf.Ticker(self.symbol)
        hist   = ticker.history(period=str(self.total_years + 1) + "y")
        if hist.empty or len(hist) < 252 * self.total_years:
            raise ValueError("Insufficient historical data for " + self.symbol)
        close     = hist["Close"].dropna()
        split_idx = len(close) - int(252 * self.validation_years)
        if split_idx < 252:
            raise ValueError("Not enough training data for " + self.symbol)
        self.train_close = close.iloc[:split_idx]
        self.val_close   = close.iloc[split_idx:]
        self.entry_price = float(self.train_close.iloc[-1])

        # ATM mode: replace screener strike with stock price at historical entry.
        # This ensures every backtest starts with a realistic ATM option
        # regardless of where the screener strike sits relative to historical prices.
        if self.use_atm_strike:
            self.strike = self.entry_price
            print(" [ATM] Strike set to $" + str(round(self.strike, 2)) +
                  " (screener strike was $" + str(round(self.screener_strike, 2)) + ")")

        # Dividend yield from training period
        try:
            divs = ticker.dividends
            if not divs.empty:
                train_start = self.train_close.index[0]
                train_end   = self.train_close.index[-1]
                divs_train  = divs[
                    (divs.index >= train_start) & (divs.index <= train_end)
                ]
                annual_div     = float(divs_train.sum()) / self.training_years
                self.div_yield = annual_div / self.entry_price
            else:
                self.div_yield = 0.0
        except Exception:
            self.div_yield = 0.0
        return self

    # -- 2. Calibration -------------------------------------------
    def calibrate(self):
        """
        Improved calibration using four refinements:
        1. RECENT DRIFT (last 2 years of training only)
        2. LONG-RUN MEAN ANCHOR (60% blend toward 10% equity mean)
        3. DRIFT CAP AT 15% and floor at risk-free rate
        4. EWMA VOLATILITY + COVERAGE CORRECTION
        """
        log_rets    = np.log(
            self.train_close.values[1:] / self.train_close.values[:-1]
        )
        # 1. Recent drift: last 2 years of training window only
        recent_days  = min(int(252 * 2), len(log_rets))
        recent_rets  = log_rets[-recent_days:]
        lo_r         = float(np.percentile(recent_rets, 1))
        hi_r         = float(np.percentile(recent_rets, 99))
        recent_trim  = recent_rets[(recent_rets >= lo_r) & (recent_rets <= hi_r)]
        mu_recent    = float(np.mean(recent_trim)) * 252

        # 2. Long-run mean anchor (60% blend toward 10% equity mean)
        LONG_RUN_MEAN = 0.10
        ANCHOR_WEIGHT = 0.40
        mu_blended_drift = (1.0 - ANCHOR_WEIGHT) * mu_recent + ANCHOR_WEIGHT * LONG_RUN_MEAN

        # 3. Drift cap at 15% and floor at risk-free rate
        DRIFT_CAP = 0.15
        self.mu   = max(min(mu_blended_drift - self.div_yield, DRIFT_CAP), self.r)
        self.mu_recent_raw = round(mu_recent,        4)
        self.mu_anchored   = round(mu_blended_drift, 4)

        # 4. EWMA volatility (lambda = 0.94 RiskMetrics standard)
        lam     = 0.94
        n       = len(log_rets)
        weights = np.array([(1 - lam) * lam ** (n - 1 - i) for i in range(n)])
        weights /= weights.sum()
        var_ewma          = float(np.dot(weights, log_rets ** 2))
        sigma_daily_ewma  = math.sqrt(var_ewma)
        self.sigma_ewma   = sigma_daily_ewma * math.sqrt(252)

        sigma_daily_plain = float(np.std(log_rets, ddof=1))
        self.sigma_train  = sigma_daily_plain * math.sqrt(252)

        val_log_rets      = np.log(
            self.val_close.values[1:] / self.val_close.values[:-1]
        )
        sigma_daily_val   = float(np.std(val_log_rets, ddof=1))
        self.sigma_val    = sigma_daily_val * math.sqrt(252)

        # Mean-reverting sigma blend
        kappa   = 2.0
        T_val   = self.validation_years
        weight  = math.exp(-kappa * T_val)
        blended = weight * self.sigma_ewma + (1 - weight) * self.sigma_val

        # Coverage-based sigma correction (cap +/- 15%)
        try:
            n_train   = len(self.train_close)
            S0_train  = float(self.train_close.iloc[0])
            dt        = 1.0 / 252.0
            np.random.seed(0)
            Z_cal     = np.random.standard_normal((200, n_train))
            drift_cal = (self.mu - 0.5 * blended ** 2) * dt
            diff_cal  = blended * math.sqrt(dt) * Z_cal
            paths_cal = S0_train * np.exp(np.cumsum(drift_cal + diff_cal, axis=1))
            actual_arr = self.train_close.values
            sim_5      = np.percentile(paths_cal,  5, axis=0)
            sim_95     = np.percentile(paths_cal, 95, axis=0)
            coverage   = float(np.mean(
                (actual_arr >= sim_5) & (actual_arr <= sim_95)
            ))
            ratio      = coverage / 0.90
            ratio      = max(0.85, min(1.15, ratio))
            self.sigma_blended          = blended / ratio
            self.sigma_coverage_raw     = round(coverage * 100, 1)
            self.sigma_coverage_ratio   = round(ratio, 4)
        except Exception:
            self.sigma_blended        = blended
            self.sigma_coverage_raw   = None
            self.sigma_coverage_ratio = 1.0
        return self

    # -- 3. Monte Carlo Simulation (vectorized GBM) ---------------
    def run_monte_carlo(self):
        n_days   = len(self.val_close)
        dt       = 1.0 / 252.0
        S0       = self.entry_price
        mu       = self.mu
        sigma    = self.sigma_blended
        np.random.seed(42)
        Z        = np.random.standard_normal((self.n_sims, n_days))
        drift    = (mu - 0.5 * sigma ** 2) * dt
        diffusion= sigma * math.sqrt(dt) * Z
        log_paths= np.cumsum(drift + diffusion, axis=1)
        self.paths = S0 * np.exp(log_paths)
        return self

    # -- 4. Reality Comparison ------------------------------------
    def reality_check(self):
        actual_prices    = self.val_close.values
        sim_terminal     = self.paths[:, -1]
        actual_terminal  = float(actual_prices[-1])
        median_sim_term  = float(np.median(sim_terminal))
        pct_5_term       = float(np.percentile(sim_terminal,  5))
        pct_95_term      = float(np.percentile(sim_terminal, 95))
        cum_max          = np.maximum.accumulate(self.paths, axis=1)
        drawdowns        = (self.paths - cum_max) / cum_max
        sim_max_dd       = float(np.mean(np.min(drawdowns, axis=1)))
        actual_cum_max   = np.maximum.accumulate(actual_prices)
        actual_dd_series = (actual_prices - actual_cum_max) / actual_cum_max
        actual_max_dd    = float(np.min(actual_dd_series))
        reality_iv       = self.sigma_val + 0.03
        T_remaining      = self.validation_years
        K                = self.strike
        S_entry          = self.entry_price
        reality_premium  = black_scholes_price(S_entry, K, T_remaining, self.r, reality_iv)
        return {
            "actual_terminal_price":      round(actual_terminal,  2),
            "median_simulated_terminal":  round(median_sim_term,  2),
            "sim_terminal_5th_pct":       round(pct_5_term,       2),
            "sim_terminal_95th_pct":      round(pct_95_term,      2),
            "terminal_price_error_pct":   round(
                (median_sim_term - actual_terminal) / actual_terminal * 100, 2
            ),
            "sim_avg_max_drawdown_pct":   round(sim_max_dd  * 100, 2),
            "actual_max_drawdown_pct":    round(actual_max_dd * 100, 2),
            "drawdown_error_pct":         round(
                (sim_max_dd - actual_max_dd) / abs(actual_max_dd) * 100
                if actual_max_dd != 0 else 0.0, 2
            ),
            "reality_iv_used_pct":        round(reality_iv * 100, 2),
            "reality_option_premium":     round(reality_premium, 2)
                if not math.isnan(reality_premium) else None,
        }

    # -- 5. Greek Attribution Over Validation Period --------------
    def greek_attribution(self):
        K       = self.strike
        S       = self.entry_price
        r       = self.r
        sigma   = self.sigma_blended
        T_full  = float(self.validation_years)
        checkpoints    = [T_full, T_full / 2, 0.25]
        greek_timeline = []
        for T in checkpoints:
            if T <= 0:
                continue
            delta_t = black_scholes_delta(S, K, T, r, sigma)
            theta_t = black_scholes_theta(S, K, T, r, sigma)
            rho_t   = black_scholes_rho(  S, K, T, r, sigma)
            gamma_t = black_scholes_gamma(S, K, T, r, sigma)
            vega_t  = black_scholes_vega( S, K, T, r, sigma)
            price_t = black_scholes_price(S, K, T, r, sigma)
            greek_timeline.append({
                "years_to_expiry":  round(T, 2),
                "option_price":     round(price_t, 2)  if not math.isnan(price_t)  else None,
                "delta":            round(delta_t, 4)  if not math.isnan(delta_t)  else None,
                "gamma":            round(gamma_t, 6)  if not math.isnan(gamma_t)  else None,
                "theta_daily":      round(theta_t, 4)  if not math.isnan(theta_t)  else None,
                "vega_per_1pct_iv": round(vega_t,  4)  if not math.isnan(vega_t)   else None,
                "rho_per_1pct_rate":round(rho_t,   4)  if not math.isnan(rho_t)    else None,
            })
        theta_entry = black_scholes_theta(S, K, T_full,     r, sigma)
        theta_mid   = black_scholes_theta(S, K, T_full / 2, r, sigma)
        avg_theta   = (theta_entry + theta_mid) / 2 if not math.isnan(theta_entry) else 0.0
        val_days    = int(self.validation_years * 365)
        cumulative_theta_cost = round(abs(avg_theta) * val_days, 2)
        entry_premium   = black_scholes_price(S, K, T_full, r, sigma)
        target_premium  = entry_premium * 2.0 if not math.isnan(entry_premium) else None
        return_checkpoints = []
        for months in [6, 12, 18]:
            T_rem = (self.validation_years * 12 - months) / 12.0
            if T_rem <= 0 or target_premium is None:
                return_checkpoints.append({
                    "months_elapsed": months,
                    "stock_price_for_100pct_return": None,
                    "pct_above_entry": None,
                })
                continue
            lo, hi    = S * 0.5, S * 5.0
            result_S  = None
            for _ in range(60):
                mid = (lo + hi) / 2
                v   = black_scholes_price(mid, K, T_rem, r, sigma)
                if math.isnan(v):
                    break
                if v < target_premium:
                    lo = mid
                else:
                    hi = mid
                if hi - lo < 0.01:
                    result_S = round((lo + hi) / 2, 2)
                    break
            pct_above = round((result_S - S) / S * 100, 1) if result_S else None
            return_checkpoints.append({
                "months_elapsed":               months,
                "stock_price_for_100pct_return": result_S,
                "pct_above_entry_price":         pct_above,
            })
        return {
            "greek_timeline":         greek_timeline,
            "cumulative_theta_cost":  cumulative_theta_cost,
            "return_checkpoints":     return_checkpoints,
        }

    # -- 6. Residual Variance Report & EV -------------------------
    def residual_variance_report(self, reality):
        K      = self.strike
        sigma  = self.sigma_blended
        T_val  = float(self.validation_years)
        r      = self.r
        S_entry= self.entry_price
        sim_terminal = self.paths[:, -1]
        n_itm    = float(np.sum(sim_terminal > K))
        p_itm    = n_itm / self.n_sims
        payoffs_itm = sim_terminal[sim_terminal > K] - K
        ev_payoff   = float(np.mean(payoffs_itm)) if len(payoffs_itm) > 0 else 0.0
        entry_premium = black_scholes_price(S_entry, K, T_val, r, sigma)
        if math.isnan(entry_premium):
            entry_premium = 0.0
        p_otm = 1.0 - p_itm
        ev    = p_itm * ev_payoff - p_otm * entry_premium
        actual_terminal  = reality["actual_terminal_price"]
        median_projected = reality["median_simulated_terminal"]
        residual_var     = (median_projected - actual_terminal) ** 2
        actual_prices    = self.val_close.values
        sim_5th          = np.percentile(self.paths,  5, axis=0)
        sim_95th         = np.percentile(self.paths, 95, axis=0)
        n_within         = int(np.sum(
            (actual_prices >= sim_5th) & (actual_prices <= sim_95th)
        ))
        pct_within_band = round(n_within / len(actual_prices) * 100, 1)
        return {
            "entry_price":              round(S_entry,       2),
            "strike":                   round(K,             2),
            "projected_entry_premium":  round(entry_premium, 2),
            "reality_premium":          reality["reality_option_premium"],
            "p_itm_pct":                round(p_itm  * 100, 1),
            "p_otm_pct":                round(p_otm  * 100, 1),
            "expected_payoff_if_itm":   round(ev_payoff,     2),
            "expected_value_ev":        round(ev,            2),
            "ev_positive":              ev > 0,
            "residual_variance":        round(residual_var,  4),
            "pct_actual_within_90_band":pct_within_band,
            "terminal_price_error_pct": reality["terminal_price_error_pct"],
            "drawdown_error_pct":       reality["drawdown_error_pct"],
            "kelly_criterion_pct":      self._kelly(p_itm, ev_payoff, entry_premium),
        }

    def _kelly(self, p_win, avg_win, premium):
        try:
            if avg_win <= 0 or premium <= 0:
                return None
            p_lose      = 1.0 - p_win
            f           = (p_win * avg_win - p_lose * premium) / avg_win
            f_half_kelly= f / 2.0
            return round(max(0.0, min(25.0, f_half_kelly * 100)), 1)
        except Exception:
            return None

    # -- 7. Returns Analysis --------------------------------------
    def returns_analysis(self):
        try:
            K      = self.strike
            r      = self.r
            sigma  = self.sigma_blended
            T_full = float(self.validation_years)
            n_days = self.paths.shape[1]
            entry_prem = black_scholes_price(self.entry_price, K, T_full, r, sigma)
            if math.isnan(entry_prem) or entry_prem <= 0:
                print(" [WARN] returns_analysis: invalid entry premium")
                return self._empty_returns()
            target_value = entry_prem * 2.0
            dt       = T_full / n_days
            t_rem_arr= np.maximum(
                np.array([T_full - i * dt for i in range(n_days)]),
                0.01
            )
            option_values = np.zeros_like(self.paths)
            for d in range(n_days):
                T_rem = float(t_rem_arr[d])
                denom = sigma * math.sqrt(T_rem)
                if denom <= 0:
                    continue
                S_col   = np.maximum(self.paths[:, d], 0.001)
                log_SK  = np.log(S_col / K)
                d1_vec  = (log_SK + (r + 0.5 * sigma ** 2) * T_rem) / denom
                d2_vec  = d1_vec - denom
                nd1     = norm.cdf(d1_vec)
                nd2     = norm.cdf(d2_vec)
                vals    = S_col * nd1 - K * math.exp(-r * T_rem) * nd2
                intrinsic = np.maximum(S_col - K, 0.0)
                option_values[:, d] = np.maximum(vals, intrinsic)
            option_values = np.nan_to_num(option_values, nan=0.0)
            sim_returns   = (option_values - entry_prem) / entry_prem * 100.0
            hit_100       = option_values >= target_value
            paths_hit_mask= np.any(hit_100, axis=1)
            n_hit_100     = int(np.sum(paths_hit_mask))
            p_hit_100     = round(n_hit_100 / self.n_sims * 100, 1)
            if n_hit_100 > 0:
                first_days        = np.argmax(hit_100[paths_hit_mask], axis=1)
                median_day_100    = int(np.median(first_days))
                earliest_day_100  = int(np.min(first_days))
                latest_day_100    = int(np.max(first_days))
            else:
                median_day_100   = None
                earliest_day_100 = None
                latest_day_100   = None
            max_ret_per_path     = np.nanmax(sim_returns, axis=1)
            median_max_return    = round(float(np.nanmedian(max_ret_per_path)),    1)
            avg_max_return       = round(float(np.nanmean(max_ret_per_path)),      1)
            pct_90_max_return    = round(float(np.nanpercentile(max_ret_per_path, 90)), 1)
            pct_10_max_return    = round(float(np.nanpercentile(max_ret_per_path, 10)), 1)
            actual_prices = self.val_close.values
            actual_option = np.zeros(n_days)
            for d in range(n_days):
                T_rem = float(t_rem_arr[d])
                S     = (float(actual_prices[d]) if d < len(actual_prices)
                         else float(actual_prices[-1]))
                v     = black_scholes_price(S, K, T_rem, r, sigma)
                actual_option[d] = max(
                    v if not math.isnan(v) else 0.0,
                    max(S - K, 0.0)
                )
            actual_returns         = (actual_option - entry_prem) / entry_prem * 100.0
            actual_hit_100_arr     = actual_option >= target_value
            actual_hit_100_bool    = bool(np.any(actual_hit_100_arr))
            actual_day_100         = (int(np.argmax(actual_hit_100_arr))
                                      if actual_hit_100_bool else None)
            actual_max_return      = round(float(np.max(actual_returns)),    1)
            actual_max_return_day  = int(np.argmax(actual_returns))
            actual_terminal_opt    = round(float(actual_option[-1]),         2)
            actual_terminal_ret    = round(float(actual_returns[-1]),        1)
            print(" [returns] hit_100=" + str(p_hit_100) + "%" +
                  " median_max=" + str(median_max_return) + "%" +
                  " actual_max=" + str(actual_max_return) + "%")
            return {
                "entry_premium":              round(entry_prem,       2),
                "target_value_100pct":        round(target_value,     2),
                "sim_pct_paths_hit_100":      p_hit_100,
                "sim_median_day_hit_100":     median_day_100,
                "sim_earliest_day_hit_100":   earliest_day_100,
                "sim_latest_day_hit_100":     latest_day_100,
                "sim_median_max_return_pct":  median_max_return,
                "sim_avg_max_return_pct":     avg_max_return,
                "sim_top10pct_max_return":    pct_90_max_return,
                "sim_bottom10pct_max_return": pct_10_max_return,
                "actual_hit_100":             actual_hit_100_bool,
                "actual_day_hit_100":         actual_day_100,
                "actual_max_return_pct":      actual_max_return,
                "actual_max_return_day":      actual_max_return_day,
                "actual_terminal_option_value": actual_terminal_opt,
                "actual_terminal_return_pct": actual_terminal_ret,
            }
        except Exception as e:
            print(" [ERROR] returns_analysis failed: " + str(e))
            return self._empty_returns()

    def _empty_returns(self):
        return {
            "entry_premium":              None,
            "target_value_100pct":        None,
            "sim_pct_paths_hit_100":      None,
            "sim_median_day_hit_100":     None,
            "sim_earliest_day_hit_100":   None,
            "sim_latest_day_hit_100":     None,
            "sim_median_max_return_pct":  None,
            "sim_avg_max_return_pct":     None,
            "sim_top10pct_max_return":    None,
            "sim_bottom10pct_max_return": None,
            "actual_hit_100":             None,
            "actual_day_hit_100":         None,
            "actual_max_return_pct":      None,
            "actual_max_return_day":      None,
            "actual_terminal_option_value": None,
            "actual_terminal_return_pct": None,
        }

    # -- 8. Full Run ----------------------------------------------
    def run(self):
        try:
            print(" Loading data...")
            self.load_data()
            print(" Calibrating GBM parameters...")
            self.calibrate()
            print(" Running " + str(self.n_sims) + " Monte Carlo paths...")
            self.run_monte_carlo()
            print(" Comparing projections to reality...")
            reality = self.reality_check()
            print(" Calculating greek attribution...")
            greeks  = self.greek_attribution()
            print(" Building residual variance report...")
            rv_report = self.residual_variance_report(reality)
            print(" Analysing 100% return and max return milestones...")
            returns   = self.returns_analysis()
            return {
                "symbol":           self.symbol,
                "strike":           round(self.strike, 2),
                "screener_strike":  round(self.screener_strike, 2),
                "atm_mode":         self.use_atm_strike,
                "entry_score":      self.entry_score,
                "entry_price":      round(self.entry_price, 2),
                "training_years":   self.training_years,
                "validation_years": self.validation_years,
                "n_simulations":    self.n_sims,
                "calibration": {
                    "annualised_mu_pct":       round(self.mu * 100, 2),
                    "mu_recent_2yr_raw_pct":   round(getattr(self, "mu_recent_raw",
                                                             self.mu) * 100, 2),
                    "mu_after_anchor_pct":     round(getattr(self, "mu_anchored",
                                                             self.mu) * 100, 2),
                    "drift_method":            "recent_2yr + 60pct_longrun_anchor + 15pct_cap",
                    "dividend_yield_pct":      round(self.div_yield * 100, 2),
                    "training_sigma_pct":      round(self.sigma_train   * 100, 2),
                    "ewma_sigma_pct":          round(self.sigma_ewma    * 100, 2),
                    "validation_sigma_pct":    round(self.sigma_val     * 100, 2),
                    "blended_sigma_pct":       round(self.sigma_blended * 100, 2),
                    "coverage_correction_ratio": getattr(self, "sigma_coverage_ratio", None),
                    "in_sample_coverage_pct":    getattr(self, "sigma_coverage_raw",   None),
                    "risk_free_rate_pct":       round(self.r * 100, 2),
                    "calibration_method":       "recent_drift + longrun_anchor + ewma_vol + coverage_correction",
                },
                "reality_check":           reality,
                "greek_attribution":       greeks,
                "residual_variance_report": rv_report,
                "returns_analysis":        returns,
                "status": "ok",
            }
        except Exception as e:
            return {
                "symbol":          self.symbol,
                "strike":          round(self.strike, 2) if self.strike else None,
                "screener_strike": round(self.screener_strike, 2) if self.screener_strike else None,
                "atm_mode":        self.use_atm_strike,
                "status":          "error",
                "error":           str(e),
            }


# ------------------------------------------------------------------
# FORWARD PROJECTION ENGINE
# ------------------------------------------------------------------
class ForwardProjectionEngine:
    """
    Projects the stock price from TODAY to the option expiry date
    using GBM calibrated on 3 years of recent price history, with
    implied volatility as the simulation sigma.
    """

    def __init__(self, symbol, current_price, strike, expiry_str,
                 implied_vol, current_premium,
                 risk_free_rate=0.0388, n_simulations=1000):
        self.symbol          = symbol
        self.S0              = current_price
        self.K               = strike
        self.expiry_str      = expiry_str
        self.iv              = implied_vol
        self.current_premium = current_premium
        self.r               = risk_free_rate
        self.n_sims          = n_simulations
        exp_date             = datetime.strptime(expiry_str, "%Y-%m-%d").date()
        self.T               = max((exp_date - date.today()).days, 1) / 365.0
        self.n_days          = max((exp_date - date.today()).days, 1)
        self.mu              = None
        self.div_yield       = 0.0
        self.paths           = None

    def calibrate(self):
        """
        Estimate drift and dividend yield from 3 years of data.
        Three fixes vs the naive 1-year approach:
          1. LONGER LOOKBACK (3yr not 1yr)
          2. DRIFT FLOOR = max(historical mu, risk-free rate)
          3. CAPPED IV AT 40% FOR LONG-DATED PROJECTIONS (T > 1yr)
        """
        try:
            ticker = yf.Ticker(self.symbol)
            hist   = ticker.history(period="3y")
            close  = hist["Close"].dropna()
            if len(close) >= 60:
                log_rets     = np.log(close.values[1:] / close.values[:-1])
                mu_daily     = float(np.mean(log_rets))
                self.mu_hist = mu_daily * 252
            else:
                self.mu_hist = self.r
            self.mu_hist = max(self.mu_hist, self.r)
            try:
                divs = ticker.dividends
                if not divs.empty:
                    annual_div     = float(divs.tail(4).sum())
                    self.div_yield = annual_div / self.S0
            except Exception:
                pass
            self.mu = max(self.mu_hist - self.div_yield, 0.0)
            if len(close) >= 30:
                log_rets_full    = np.log(close.values[1:] / close.values[:-1])
                hist_vol_daily   = float(np.std(log_rets_full, ddof=1))
                self.hist_vol_1y = hist_vol_daily * math.sqrt(252)
            else:
                self.hist_vol_1y = 0.25
        except Exception:
            self.mu          = self.r
            self.mu_hist     = self.r
            self.hist_vol_1y = 0.25

        # IV sanity check: floor at 8% minimum realistic equity IV
        IV_FLOOR = 0.08
        if self.iv < IV_FLOOR:
            print(" [IV WARN] " + self.symbol + " raw IV " +
                  str(round(self.iv * 100, 1)) + "% is below floor. " +
                  "Using historical vol " +
                  str(round(self.hist_vol_1y * 100, 1)) + "% instead.")
            self.iv        = max(self.hist_vol_1y, IV_FLOOR)
            self.iv_was_bad = True
        else:
            self.iv_was_bad = False

        # Cap IV at 40% for projections longer than 1 year
        if self.T > 1.0 and self.iv > 0.40:
            self.iv_capped = 0.40
        else:
            self.iv_capped = self.iv
        return self

    def run_simulation(self):
        dt        = 1.0 / 252.0
        mu        = self.mu
        sigma     = self.iv_capped
        np.random.seed(99)
        Z         = np.random.standard_normal((self.n_sims, self.n_days))
        drift     = (mu - 0.5 * sigma ** 2) * dt
        diffusion = sigma * math.sqrt(dt) * Z
        log_paths = np.cumsum(drift + diffusion, axis=1)
        self.paths= self.S0 * np.exp(log_paths)
        return self

    def projection_report(self):
        terminal     = self.paths[:, -1]
        median_price = float(np.median(terminal))
        pct5         = float(np.percentile(terminal,  5))
        pct25        = float(np.percentile(terminal, 25))
        pct75        = float(np.percentile(terminal, 75))
        pct95        = float(np.percentile(terminal, 95))
        n_itm        = int(np.sum(terminal > self.K))
        p_itm        = n_itm / self.n_sims
        payoffs_itm  = terminal[terminal > self.K] - self.K
        ev_payoff    = float(np.mean(payoffs_itm)) if len(payoffs_itm) > 0 else 0.0
        pv_payoff    = ev_payoff * math.exp(-self.r * self.T)
        p_otm        = 1.0 - p_itm
        ev           = p_itm * pv_payoff - p_otm * self.current_premium
        projected_bs = black_scholes_price(
            median_price, self.K, 0.01, self.r, self.iv_capped
        )
        upside_median  = (median_price - self.S0) / self.S0 * 100
        upside_pct95   = (pct95 - self.S0) / self.S0 * 100
        downside_pct5  = (pct5  - self.S0) / self.S0 * 100
        entry_gamma    = black_scholes_gamma(self.S0, self.K, self.T, self.r, self.iv_capped)
        entry_vega     = black_scholes_vega( self.S0, self.K, self.T, self.r, self.iv_capped)
        entry_opt_price= black_scholes_price(self.S0, self.K, self.T, self.r, self.iv_capped)
        target_100     = entry_opt_price * 2.0 if not math.isnan(entry_opt_price) else None
        return_checkpoints = []
        for months in [6, 12, 18]:
            T_rem = self.T - months / 12.0
            if T_rem <= 0.01 or target_100 is None:
                return_checkpoints.append({
                    "months_elapsed":               months,
                    "stock_price_for_100pct_return": None,
                    "pct_above_current":             None,
                })
                continue
            lo, hi   = self.S0 * 0.5, self.S0 * 5.0
            result_S = None
            for _ in range(60):
                mid = (lo + hi) / 2
                v   = black_scholes_price(mid, self.K, T_rem, self.r, self.iv_capped)
                if math.isnan(v):
                    break
                if v < target_100:
                    lo = mid
                else:
                    hi = mid
                if hi - lo < 0.01:
                    result_S = round((lo + hi) / 2, 2)
                    break
            pct_above = round((result_S - self.S0) / self.S0 * 100, 1) if result_S else None
            return_checkpoints.append({
                "months_elapsed":               months,
                "stock_price_for_100pct_return": result_S,
                "pct_above_current":             pct_above,
            })
        kelly = None
        try:
            if p_itm > 0 and ev_payoff > 0 and self.current_premium > 0:
                f     = (p_itm * ev_payoff - (1 - p_itm) * self.current_premium) / ev_payoff
                kelly = round(max(0.0, min(25.0, f / 2 * 100)), 1)
        except Exception:
            pass
        target_stock_for_100 = return_checkpoints[-1].get("stock_price_for_100pct_return")
        p_100_return = None
        if target_stock_for_100:
            p_100_return = round(
                float(np.mean(terminal >= target_stock_for_100)) * 100, 1
            )
        return {
            "symbol":               self.symbol,
            "current_price":        round(self.S0,              2),
            "strike":               round(self.K,               2),
            "expiry":               self.expiry_str,
            "days_to_expiry":       self.n_days,
            "years_to_expiry":      round(self.T,               3),
            "current_premium":      round(self.current_premium, 2),
            "implied_vol_raw_pct":  round(self.iv        * 100, 2),
            "implied_vol_used_pct": round(self.iv_capped * 100, 2),
            "iv_was_capped":        self.iv_capped < self.iv,
            "iv_data_corrected":    getattr(self, "iv_was_bad", False),
            "historical_mu_pct":    round(self.mu_hist * 100, 2) if self.mu_hist else None,
            "div_yield_pct":        round(self.div_yield * 100, 2),
            "calibrated_drift_pct": round(self.mu * 100, 2) if self.mu else None,
            "n_simulations":        self.n_sims,
            "projected_prices": {
                "median": round(median_price, 2),
                "pct_5":  round(pct5,         2),
                "pct_25": round(pct25,         2),
                "pct_75": round(pct75,         2),
                "pct_95": round(pct95,         2),
            },
            "upside_downside": {
                "median_change_pct": round(upside_median,  1),
                "pct95_change_pct":  round(upside_pct95,   1),
                "pct5_change_pct":   round(downside_pct5,  1),
            },
            "p_itm_pct":              round(p_itm  * 100, 1),
            "p_otm_pct":              round(p_otm  * 100, 1),
            "p_100pct_return_pct":    p_100_return,
            "expected_payoff_if_itm": round(ev_payoff,     2),
            "pv_expected_payoff":     round(pv_payoff,     2),
            "expected_value_ev":      round(ev,            2),
            "ev_positive":            ev > 0,
            "kelly_criterion_pct":    kelly,
            "projected_option_value": round(projected_bs,  2)
                if not math.isnan(projected_bs) else None,
            "entry_gamma":            round(entry_gamma,   6)
                if not math.isnan(entry_gamma) else None,
            "entry_vega":             round(entry_vega,    4)
                if not math.isnan(entry_vega) else None,
            "return_checkpoints":     return_checkpoints,
            "status": "ok",
        }

    def run(self):
        try:
            self.calibrate()
            self.run_simulation()
            return self.projection_report()
        except Exception as e:
            return {
                "symbol": self.symbol,
                "strike": self.K,
                "expiry": self.expiry_str,
                "status": "error",
                "error":  str(e),
            }


# ------------------------------------------------------------------
# ATM OPTION SELECTOR
# ------------------------------------------------------------------
def select_atm_option(options_for_ticker):
    """
    From a list of options for a single ticker, return the one whose
    strike is closest to the current stock price (i.e. the ATM option).
    This minimises the premium paid vs deep-ITM options which carry
    expensive intrinsic value.  Composite score is used as a tiebreaker
    when two strikes are equidistant from current price.
    """
    if not options_for_ticker:
        return None
    return min(
        options_for_ticker,
        key=lambda x: (
            abs(x["strike"] - x["current_price"]),
            -x["composite_score"],
        ),
    )


def build_atm_per_ticker(all_results):
    """
    Given all screener results, return a dict of {ticker: atm_option}
    where the chosen option per ticker is the ATM one (strike nearest
    current price).  Results are processed in composite-score order so
    that each ticker's full option list is already available before
    ATM selection.
    """
    grouped = {}
    for opt in all_results:
        sym = opt["ticker"]
        grouped.setdefault(sym, []).append(opt)
    return {sym: select_atm_option(opts) for sym, opts in grouped.items()}


# ------------------------------------------------------------------
# FORWARD PROJECTION RUNNER
# ------------------------------------------------------------------
def run_forward_projections(all_results):
    print("")
    print("=================================================================")
    print(" FORWARD PRICE PROJECTIONS (today to expiry via Monte Carlo)")
    print("=================================================================")
    if not all_results:
        print(" No screener results to project.")
        return []
    # Select the ATM option (strike closest to current price) per ticker.
    # This matches the trading rule: always buy the ATM call to minimise premium.
    best_per_ticker = build_atm_per_ticker(all_results)
    projections = []
    tickers     = list(best_per_ticker.keys())
    for i, sym in enumerate(tickers, 1):
        opt = best_per_ticker[sym]
        print(" [" + str(i).rjust(2) + "/" + str(len(tickers)) + "] " +
              sym.ljust(8) +
              " Strike=" + str(opt["strike"]) +
              " Expiry=" + opt["expiry"] +
              " DTE="    + str(opt["dte"]) + " days",
              end="", flush=True)
        engine = ForwardProjectionEngine(
            symbol        = sym,
            current_price = opt["current_price"],
            strike        = opt["strike"],
            expiry_str    = opt["expiry"],
            implied_vol   = opt["implied_volatility_pct"] / 100.0,
            current_premium = opt["premium"],
            risk_free_rate  = RISK_FREE_RATE,
            n_simulations   = FP_N_SIMULATIONS,
        )
        result = engine.run()
        if result["status"] == "ok":
            ev_label = "EV+" if result["ev_positive"] else "EV-"
            print(" DONE Median=$" + str(result["projected_prices"]["median"]) +
                  " P(ITM)=" + str(result["p_itm_pct"]) + "%" +
                  " " + ev_label)
        else:
            print(" ERROR " + result.get("error", "unknown"))
        projections.append(result)
        time.sleep(DELAY_SECONDS)
    return projections


# ------------------------------------------------------------------
# SCREENER - ANALYZE TICKER
# ------------------------------------------------------------------
def analyze_ticker(symbol, idx, total):
    prefix = " [" + str(idx).rjust(3) + "/" + str(total) + "] " + symbol.ljust(8)
    print(prefix, end="", flush=True)
    try:
        ticker = yf.Ticker(symbol)
        fund   = fetch_fundamental_data(ticker)
        passed, reason = passes_fundamental_screen(fund)
        if not passed:
            print(" FAIL " + reason)
            return []
        hist = ticker.history(period="2y")
        if hist.empty or len(hist) < 30:
            print(" FAIL Insufficient price history")
            return []
        close  = hist["Close"]
        volume = hist["Volume"]
        high   = hist["High"]
        low    = hist["Low"]
        current_price = float(close.iloc[-1])

        # Daily indicators
        rsi_val                      = calculate_rsi(close)
        bb_upper, bb_middle, bb_lower= calculate_bollinger_bands(close)
        ma_200 = (float(close.rolling(200).mean().iloc[-1])
                  if len(close) >= 200 else float(close.mean()))
        bb_width      = (bb_upper - bb_lower) or 1
        bb_pos        = (current_price - bb_lower) / bb_width
        price_vs_200ma= (current_price - ma_200) / ma_200
        atr_val       = calculate_atr(high, low, close)
        obv_data      = calculate_obv(close, volume)

        # Weekly indicators
        weekly       = resample_to_weekly(hist)
        weekly_close = weekly["Close"]
        rsi_weekly   = calculate_rsi(weekly_close) if len(weekly_close) >= 14 else None
        macd_data    = calculate_macd(weekly_close)
        stoch_data   = calculate_stoch_rsi(weekly_close)
        weekly_mas   = calculate_weekly_mas(weekly_close)

        # ABCD harmonic pattern (weekly prices)
        abcd_data  = detect_abcd_pattern(weekly_close, current_price)

        # LEAPS entry checklist (iv_percentile filled after options fetch)
        volume_data = {"obv_rising": obv_data.get("obv_rising", False)}
        checklist   = leaps_entry_checklist(
            current_price, rsi_weekly, macd_data, obv_data,
            bb_pos, stoch_data, weekly_mas, 0.0,
            abcd_data, volume_data
        )

        expirations = ticker.options
        if not expirations:
            print(" FAIL No options data")
            return []
        expiry = expirations[-1]
        chain  = ticker.option_chain(expiry)
        calls  = chain.calls.copy()
        if calls.empty:
            print(" FAIL No calls at " + expiry)
            return []

        exp_date     = datetime.strptime(expiry, "%Y-%m-%d").date()
        dte          = max((exp_date - date.today()).days, 1)
        T            = dte / 365.0
        mid_iv       = float(calls["impliedVolatility"].median())
        iv_percentile= calculate_iv_percentile(close, mid_iv)

        # Rerun checklist with actual IV percentile
        checklist = leaps_entry_checklist(
            current_price, rsi_weekly, macd_data, obv_data,
            bb_pos, stoch_data, weekly_mas, iv_percentile,
            abcd_data, volume_data
        )

        param_scores_base = {
            "market_cap":     score_parameter(fund["market_cap"],     "market_cap"),
            "avg_volume":     score_parameter(fund["avg_volume"],     "avg_volume"),
            "inst_ownership": score_parameter(fund["inst_ownership"], "inst_ownership"),
            "eps_growth":     score_parameter(fund["eps_growth"],     "eps_growth"),
            "revenue_growth": score_parameter(fund["revenue_growth"], "revenue_growth"),
            "ebitda_margin":  score_parameter(fund["ebitda_margin"],  "ebitda_margin"),
            "rsi":            score_parameter(rsi_val,                "rsi"),
            "price_vs_200ma": score_parameter(price_vs_200ma,         "price_vs_200ma"),
            "bb_position":    score_parameter(bb_pos,                 "bb_position"),
            "iv_percentile":  score_parameter(iv_percentile,          "iv_percentile"),
        }

        results = []
        for _, opt in calls.iterrows():
            strike   = float(opt["strike"])
            iv       = float(opt["impliedVolatility"])
            bid      = float(opt.get("bid", 0) or 0)
            ask      = float(opt.get("ask", 0) or 0)
            prem     = ((bid + ask) / 2 if (bid + ask) > 0
                        else float(opt.get("lastPrice", 0) or 0))
            if prem <= 0:
                continue
            delta    = black_scholes_delta(current_price, strike, T, RISK_FREE_RATE, iv)
            theta    = black_scholes_theta(current_price, strike, T, RISK_FREE_RATE, iv)
            gamma    = black_scholes_gamma(current_price, strike, T, RISK_FREE_RATE, iv)
            vega     = black_scholes_vega( current_price, strike, T, RISK_FREE_RATE, iv)
            if math.isnan(delta):
                continue
            required_move  = prem / delta if delta != 0 else float("nan")
            leverage_ratio = delta * (current_price / prem) if prem != 0 else float("nan")
            breakeven      = strike + 2 * prem
            param_scores   = dict(param_scores_base)
            param_scores["delta"] = score_parameter(delta, "delta")
            results.append({
                "ticker":            symbol,
                "exchange":          fund.get("exchange", ""),
                "tradingview_symbol":(fund.get("exchange", "") + ":" + symbol
                                      if fund.get("exchange") else symbol),
                "expiry":            expiry,
                "strike":            round(strike, 2),
                "contract":          opt.get("contractSymbol", ""),
                "current_price":     round(current_price, 2),
                "market_cap":        fund["market_cap"],
                "avg_volume":        fund["avg_volume"],
                "inst_ownership":    (round(fund["inst_ownership"] * 100, 1)
                                      if fund["inst_ownership"] is not None else None),
                "eps_growth_pct":    (round(fund["eps_growth"] * 100, 1)
                                      if fund["eps_growth"] is not None else None),
                "revenue_growth_pct":(round(fund["revenue_growth"] * 100, 1)
                                      if fund["revenue_growth"] is not None else None),
                "ebitda_margin_pct": (round(fund["ebitda_margin"] * 100, 1)
                                      if fund["ebitda_margin"] is not None else None),
                "rsi":               round(rsi_val, 2),
                "rsi_weekly":        round(rsi_weekly, 2) if rsi_weekly else None,
                "price_vs_200ma_pct":round(price_vs_200ma * 100, 2),
                "bb_lower":          round(bb_lower,  2),
                "bb_middle":         round(bb_middle, 2),
                "bb_upper":          round(bb_upper,  2),
                "bb_position_pct":   round(bb_pos * 100, 1),
                "atr":               round(atr_val, 2) if atr_val else None,
                "macd":              macd_data,
                "stoch_rsi":         stoch_data,
                "obv":               obv_data,
                "weekly_mas":        weekly_mas,
                "abcd_pattern":      abcd_data,
                "leaps_checklist":   checklist,
                "premium":           round(prem, 2),
                "implied_volatility_pct": round(iv * 100, 2),
                "iv_percentile":     round(iv_percentile, 1),
                "dte":               dte,
                "delta":             round(delta, 4),
                "gamma":             round(gamma, 6) if not math.isnan(gamma) else None,
                "theta_daily":       round(theta, 4),
                "vega":              round(vega,  4) if not math.isnan(vega)  else None,
                "required_stock_move":(round(required_move,  2)
                                       if not math.isnan(required_move)  else None),
                "leverage_ratio":    (round(leverage_ratio, 2)
                                      if not math.isnan(leverage_ratio) else None),
                "breakeven_price":   round(breakeven, 2),
                # Stock metadata
                "beta":                  fund.get("beta"),
                "short_interest_pct":    fund.get("short_interest_pct"),
                "analyst_target":        fund.get("analyst_target"),
                "analyst_low":           fund.get("analyst_low"),
                "analyst_high":          fund.get("analyst_high"),
                "analyst_count":         fund.get("analyst_count"),
                "analyst_upside_pct":    fund.get("analyst_upside_pct"),
                "next_earnings_date":    fund.get("next_earnings_date"),
                "avg_earnings_move_pct": fund.get("avg_earnings_move_pct"),
                "sector":                fund.get("sector",   ""),
                "industry":              fund.get("industry", ""),
                "free_cashflow":         fund.get("free_cashflow"),
                "debt_to_equity":        fund.get("debt_to_equity"),
                "scores":          dict((k, round(v, 1)) for k, v in param_scores.items()),
                "composite_score": compute_composite_score(param_scores),
                # ATM distance: smaller = closer to current price (ATM).
                # Used as primary sort key so ATM option always appears first.
                "atm_distance":    round(abs(strike - current_price), 2),
            })

        # Sort: ATM option first (smallest |strike - current_price|),
        # composite score descending as tiebreaker.
        # This ensures the first result per ticker is always the ATM call,
        # minimising premium paid vs deep-ITM options.
        results.sort(key=lambda x: (x["atm_distance"], -x["composite_score"]))
        rg_s = (str(round(fund["revenue_growth"] * 100)) + "%"
                if fund["revenue_growth"] is not None else "N/A")
        em_s = (str(round(fund["ebitda_margin"] * 100)) + "%"
                if fund["ebitda_margin"] is not None else "N/A")
        print(" PASS " + str(len(results)) + " opts RevGrowth=" + rg_s +
              " EBITDA=" + em_s)
        time.sleep(DELAY_SECONDS)
        return results

    except Exception as e:
        print(" ERROR " + str(e))
        return []


# ------------------------------------------------------------------
# BACKTEST RUNNER
# ------------------------------------------------------------------
def run_backtest(all_results):
    atm_label = " [ATM STRIKE MODE]" if BT_USE_ATM_STRIKE else ""
    print("")
    print("=================================================================")
    print(" WALK-FORWARD BACKTEST (" +
          str(BT_TRAINING_YEARS)   + "yr train / " +
          str(BT_VALIDATION_YEARS) + "yr validate / " +
          str(BT_N_SIMULATIONS)    + " simulations)" + atm_label)
    print(" Backtesting best option for EVERY passing ticker")
    print("=================================================================")
    if not all_results:
        print(" No screener results to backtest.")
        return []
    # Select the ATM option per ticker to match the trading rule of buying
    # the ATM call.  Composite score still determines ticker ranking in
    # the overall leaderboard; ATM governs strike selection only.
    seen = build_atm_per_ticker(all_results)
    candidates = list(seen.values())
    print(" Tickers to backtest: " + str(len(candidates)))
    for c in candidates:
        print("   " + c["ticker"] +
              " Strike=" + str(c["strike"]) +
              " Score="  + str(c["composite_score"]))
    if BT_USE_ATM_STRIKE:
        print(" NOTE: Strikes will be overridden to ATM at historical entry date.")
    print("")
    bt_results = []
    for i, cand in enumerate(candidates, 1):
        symbol = cand["ticker"]
        strike = cand["strike"]
        score  = cand["composite_score"]
        print(" [" + str(i).rjust(2) + "/" + str(len(candidates)) + "] " +
              symbol + " strike=" + str(strike) + " score=" + str(score))
        engine = BacktestEngine(
            symbol          = symbol,
            strike          = strike,
            entry_score     = score,
            training_years  = BT_TRAINING_YEARS,
            validation_years= BT_VALIDATION_YEARS,
            n_simulations   = BT_N_SIMULATIONS,
            risk_free_rate  = RISK_FREE_RATE,
            use_atm_strike  = BT_USE_ATM_STRIKE,
        )
        result = engine.run()
        if result["status"] == "ok":
            rv       = result["residual_variance_report"]
            ev_label = "EV+" if rv["ev_positive"] else "EV-"
            atm_note = (" ATM=$" + str(result["strike"])
                        if BT_USE_ATM_STRIKE else "")
            print("   " + ev_label + atm_note +
                  " P(ITM)=" + str(rv["p_itm_pct"]) + "%" +
                  " EV=$"    + str(rv["expected_value_ev"]) +
                  " TermErr=" + str(rv["terminal_price_error_pct"]) + "%")
        else:
            print("   ERROR: " + result.get("error", "unknown"))
        bt_results.append(result)
    print("")
    return bt_results


# ------------------------------------------------------------------
# OUTPUT HELPERS
# ------------------------------------------------------------------
def save_json(data, filename):
    path = OUTPUT_DIR + "/" + filename
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(" Saved: " + path)


def save_snapshot(data, filename, snapshot_date):
    import os
    folder = OUTPUT_DIR + "/snapshots/" + snapshot_date
    os.makedirs(folder, exist_ok=True)
    path   = folder + "/" + filename
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


# ------------------------------------------------------------------
# TRADINGVIEW CONFIG BUILDER
# ------------------------------------------------------------------
def build_tradingview_config(all_results, passed_tickers):
    if not all_results:
        return {"tickers": [], "ticker_details": {}}
    # Use ATM option per ticker for the TradingView config so that the
    # strike displayed on the dashboard matches the ATM trading rule.
    best_per_ticker = build_atm_per_ticker(all_results)
    tickers_ordered = sorted(
        best_per_ticker.keys(),
        key=lambda s: best_per_ticker[s]["composite_score"],
        reverse=True,
    )
    ticker_details  = {}
    for sym in tickers_ordered:
        opt  = best_per_ticker[sym]
        abcd = opt.get("abcd_pattern",    {})
        cl   = opt.get("leaps_checklist", {})
        ticker_details[sym] = {
            "ticker":            sym,
            "exchange":          opt.get("exchange", ""),
            "tradingview_symbol":opt.get("tradingview_symbol", sym),
            "current_price":     opt["current_price"],
            "composite_score":   opt["composite_score"],
            "best_strike":       opt["strike"],
            "best_expiry":       opt["expiry"],
            "best_premium":      opt["premium"],
            "delta":             opt["delta"],
            "rsi_daily":         opt.get("rsi"),
            "rsi_weekly":        opt.get("rsi_weekly"),
            "iv_percentile":     opt.get("iv_percentile"),
            "macd_bullish":      opt.get("macd",       {}).get("macd_above_signal"),
            "stoch_oversold":    opt.get("stoch_rsi",  {}).get("oversold"),
            "obv_divergence":    opt.get("obv",        {}).get("bullish_divergence"),
            "above_50wk_ma":     opt.get("weekly_mas", {}).get("above_ma50_weekly"),
            "above_200wk_ma":    opt.get("weekly_mas", {}).get("above_ma200_weekly"),
            "ma50_weekly":       opt.get("weekly_mas", {}).get("ma50_weekly"),
            "ma200_weekly":      opt.get("weekly_mas", {}).get("ma200_weekly"),
            "abcd_status":             abcd.get("status"),
            "abcd_A":                  abcd.get("A_price"),
            "abcd_A_date":             abcd.get("A_date"),
            "abcd_B":                  abcd.get("B_price"),
            "abcd_B_date":             abcd.get("B_date"),
            "abcd_C":                  abcd.get("C_price"),
            "abcd_C_date":             abcd.get("C_date"),
            "abcd_D_targets":          abcd.get("D_targets"),
            "abcd_near_C":             abcd.get("price_near_C"),
            "abcd_bc_retracement_pct": abcd.get("BC_retracement_pct"),
            "checklist_score":   cl.get("confluence_score"),
            "checklist_max":     cl.get("max_score"),
            "checklist_pct":     cl.get("confluence_pct"),
            "checklist_items":   cl.get("checklist"),
        }
    tradingview_symbols = [
        ticker_details[sym]["tradingview_symbol"]
        for sym in tickers_ordered
    ]
    return {
        "tickers":             tickers_ordered,
        "tradingview_symbols": tradingview_symbols,
        "ticker_details":      ticker_details,
    }


# ------------------------------------------------------------------
# PORTFOLIO ANALYSIS
# ------------------------------------------------------------------
def build_portfolio_analysis(all_results):
    if not all_results:
        return {}
    best = build_atm_per_ticker(all_results)
    sector_count = {}
    for opt in best.values():
        s = opt.get("sector") or "Unknown"
        sector_count[s] = sector_count.get(s, 0) + 1
    total      = len(best)
    sector_pct = {s: round(c / total * 100, 1) for s, c in sector_count.items()}
    concentrated_sectors = [s for s, pct in sector_pct.items() if pct > 30]
    betas     = [opt.get("beta") for opt in best.values() if opt.get("beta")]
    avg_beta  = round(sum(betas) / len(betas), 2) if betas else None
    upsides   = [opt.get("analyst_upside_pct") for opt in best.values()
                 if opt.get("analyst_upside_pct") is not None]
    avg_analyst_upside = round(sum(upsides) / len(upsides), 1) if upsides else None
    high_short_interest = [opt["ticker"] for opt in best.values()
                           if (opt.get("short_interest_pct") or 0) > 10]
    return {
        "total_tickers":            total,
        "sector_breakdown":         sector_pct,
        "concentrated_sectors":     concentrated_sectors,
        "avg_portfolio_beta":       avg_beta,
        "avg_analyst_upside_pct":   avg_analyst_upside,
        "high_short_interest_tickers": high_short_interest,
    }


# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------
def main():
    print("=================================================================")
    print(" LEAP CALL OPTIONS SCREENER")
    print(" Run date   : " + str(date.today()))
    print(" Tickers    : " + str(len(WATCHLIST)))
    print(" BT Config  : " + str(BT_TRAINING_YEARS) + "T / " +
          str(BT_VALIDATION_YEARS) + "V / " +
          str(BT_N_SIMULATIONS) + " sims" +
          (" / ATM Strike" if BT_USE_ATM_STRIKE else ""))
    print("=================================================================")
    print("")
    print("Fundamental criteria:")
    print("  Market Cap     > $25B")
    print("  Avg Volume     > 1M shares/day")
    print("  Inst. Ownership> 50%")
    print("  EPS Growth     > +15%")
    print("  Revenue Growth > +8%")
    print("  EBITDA Margin  > 15%")
    print("")
    total = len(WATCHLIST)
    print("Screening " + str(total) + " tickers...")
    print("")
    all_results    = []
    per_stock      = {}
    passed_tickers = []
    for idx, symbol in enumerate(WATCHLIST, 1):
        options = analyze_ticker(symbol, idx, total)
        if options:
            passed_tickers.append(symbol)
            per_stock[symbol] = options[:20]
            all_results.extend(options)
    all_results.sort(key=lambda x: x["composite_score"], reverse=True)
    # Top 20 overall: one ATM option per ticker, ranked by composite score.
    # Without deduplication every slot would be filled by different strikes
    # of the same ticker, hiding all other companies.
    atm_by_ticker = build_atm_per_ticker(all_results)
    top_20_overall = sorted(
        atm_by_ticker.values(),
        key=lambda x: x["composite_score"],
        reverse=True,
    )[:20]
    run_meta = {
        "generated_at":   datetime.utcnow().isoformat() + "Z",
        "watchlist":      WATCHLIST,
        "tickers_passed": len(passed_tickers),
        "tickers_list":   passed_tickers,
        "screening_parameters": {
            "min_market_cap_usd":      MIN_MARKET_CAP,
            "min_avg_volume_shares":   MIN_AVG_VOLUME,
            "min_inst_ownership_pct":  MIN_INST_OWNERSHIP * 100,
            "min_eps_growth_pct":      MIN_EPS_GROWTH     * 100,
            "min_revenue_growth_pct":  MIN_REVENUE_GROWTH * 100,
            "min_ebitda_margin_pct":   MIN_EBITDA_MARGIN  * 100,
            "rsi_upper":               RSI_UPPER,
            "price_vs_200ma_range":    "+-" + str(PRICE_VS_200MA_RANGE * 100) + "%",
            "delta_range":             [DELTA_MIN, DELTA_MAX],
            "max_iv_percentile":       MAX_IV_PERCENTILE,
        },
        "score_weights":       WEIGHTS,
        "weights_config_file": WEIGHTS_CONFIG_FILE,
    }
    print("")
    print(str(len(passed_tickers)) + "/" + str(total) + " tickers passed.")
    print("Saving output files...")
    today_str = str(date.today())
    save_json({"meta": run_meta, "top_20_per_stock": per_stock},
              "top_20_per_stock.json")
    save_json({"meta": run_meta, "top_20_overall": top_20_overall},
              "top_20_overall.json")
    save_json({"meta": run_meta, "all_screened_options": all_results},
              "full_results.json")
    save_snapshot({"meta": run_meta, "top_20_per_stock": per_stock},
                  "top_20_per_stock.json", today_str)
    save_snapshot({"meta": run_meta, "top_20_overall": top_20_overall},
                  "top_20_overall.json", today_str)
    save_snapshot({"meta": run_meta, "all_screened_options": all_results},
                  "full_results.json", today_str)

    # TradingView config
    tv_config = build_tradingview_config(all_results, passed_tickers)
    tv_meta   = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "note": (
            "Use this file to populate the TradingView ticker dropdown "
            "on the Base44 dashboard. The tickers list is sorted by "
            "composite score. Each ticker_details entry contains ABCD "
            "pattern data, LEAPS checklist scores, and key indicator values."
        ),
    }
    save_json({"meta": tv_meta, **tv_config}, "tradingview_config.json")
    save_snapshot({"meta": tv_meta, **tv_config},
                  "tradingview_config.json", today_str)

    # Portfolio analysis
    portfolio = build_portfolio_analysis(all_results)
    port_meta = {"generated_at": datetime.utcnow().isoformat() + "Z"}
    save_json({"meta": port_meta, "portfolio_analysis": portfolio},
              "portfolio_analysis.json")
    save_snapshot({"meta": port_meta, "portfolio_analysis": portfolio},
                  "portfolio_analysis.json", today_str)

    print("")
    print("=================================================================")
    print(" TOP 20 OVERALL")
    print("=================================================================")
    for i, opt in enumerate(top_20_overall, 1):
        print(
            str(i) + ". " + opt["ticker"] +
            " Strike=" + str(opt["strike"]) +
            " Expiry=" + opt["expiry"] +
            " Prem=$"  + str(opt["premium"]) +
            " Delta="  + str(opt["delta"]) +
            " Score="  + str(opt["composite_score"])
        )
    print("=================================================================")

    # Run backtest if enabled
    if RUN_BACKTEST and all_results:
        bt_results = run_backtest(all_results)
        bt_meta    = {
            "generated_at":      datetime.utcnow().isoformat() + "Z",
            "training_years":    BT_TRAINING_YEARS,
            "validation_years":  BT_VALIDATION_YEARS,
            "n_simulations":     BT_N_SIMULATIONS,
            "atm_strike_mode":   BT_USE_ATM_STRIKE,
            "risk_free_rate_pct":RISK_FREE_RATE * 100,
            "tickers_backtested":len(bt_results),
            "entry_signal":      "composite_score (best option per ticker, all passing tickers)",
        }
        save_json({"meta": bt_meta, "backtest_results": bt_results},
                  "backtest_results.json")
        save_snapshot({"meta": bt_meta, "backtest_results": bt_results},
                      "backtest_results.json", today_str)
        print("")
        print("=================================================================")
        print(" BACKTEST SUMMARY")
        print("=================================================================")
        for r in bt_results:
            if r["status"] == "ok":
                rv  = r["residual_variance_report"]
                cal = r["calibration"]
                atm_note = (" [ATM $" + str(r["strike"]) + "]"
                            if r.get("atm_mode") else "")
                print(" " + r["symbol"] + atm_note +
                      " | Strike=$" + str(r["strike"]) +
                      " | Score="   + str(r["entry_score"]))
                print("   Sigma(train)=" + str(cal["training_sigma_pct"])    + "%" +
                      " Sigma(val)="     + str(cal["validation_sigma_pct"]) + "%" +
                      " DivYield="       + str(cal["dividend_yield_pct"])   + "%")
                print("   P(ITM)="    + str(rv["p_itm_pct"])               + "%" +
                      " EV=$"         + str(rv["expected_value_ev"])        +
                      " TermErr="     + str(rv["terminal_price_error_pct"]) + "%" +
                      " DDErr="       + str(rv["drawdown_error_pct"])       + "%")
                print("   Actual in 90pct band: " +
                      str(rv["pct_actual_within_90_band"]) + "%" +
                      "  EV Positive: " + str(rv["ev_positive"]))
            else:
                print(" " + r["symbol"] + " | ERROR: " + r.get("error", "unknown"))
        print("")
        print("=================================================================")

    # Run forward projections if enabled
    if RUN_FORWARD_PROJECTION and all_results:
        fp_results = run_forward_projections(all_results)
        fp_meta    = {
            "generated_at":    datetime.utcnow().isoformat() + "Z",
            "projection_date": str(date.today()),
            "n_simulations":   FP_N_SIMULATIONS,
            "risk_free_rate_pct": RISK_FREE_RATE * 100,
            "note": (
                "Forward projections run from today to each option expiry "
                "using implied volatility as sigma and historical mu as drift. "
                "One projection per ticker using the highest-scoring option."
            ),
        }
        save_json({"meta": fp_meta, "forward_projections": fp_results},
                  "forward_projections.json")
        save_snapshot({"meta": fp_meta, "forward_projections": fp_results},
                      "forward_projections.json", today_str)
        print("")
        print("=================================================================")
        print(" FORWARD PROJECTION SUMMARY")
        print("=================================================================")
        for r in fp_results:
            if r["status"] == "ok":
                pp       = r["projected_prices"]
                ud       = r["upside_downside"]
                ev_label = "EV POSITIVE" if r["ev_positive"] else "EV NEGATIVE"
                print(" " + r["symbol"] +
                      " | Strike=$" + str(r["strike"]) +
                      " | Expiry="  + r["expiry"])
                print("   Current=$"     + str(r["current_price"]) +
                      " Median Proj=$"   + str(pp["median"]) +
                      " (" + str(ud["median_change_pct"]) + "%)")
                print("   90pct Range: $" + str(pp["pct_5"]) +
                      " to $"            + str(pp["pct_95"]) +
                      " | P(ITM)="       + str(r["p_itm_pct"]) + "%" +
                      " | EV=$"          + str(r["expected_value_ev"]) +
                      " | " + ev_label)
            else:
                print(" " + r["symbol"] + " | ERROR: " + r.get("error", ""))
        print("")
        print("=================================================================")


if __name__ == "__main__":
    main()
