# LEAP Call Options Screener
# ==========================
# Screens a manual watchlist of stocks for the best LEAP call options.
#
# Outputs:
#   top_20_per_stock.json  - best 20 options for each ticker
#   top_20_overall.json    - best 20 options across all tickers
#   full_results.json      - every screened option with all metrics
#
# Install: pip install yfinance pandas numpy scipy
# Run:     python leap_screener.py

import json
import math
import time
import warnings
from datetime import datetime, date

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

RISK_FREE_RATE = 0.05
OUTPUT_DIR     = "."
DELAY_SECONDS  = 0.2

# Fundamental thresholds
MIN_MARKET_CAP     = 25e9   # $25 Billion
MIN_AVG_VOLUME     = 1e6   # 1 Million shares/day
MIN_INST_OWNERSHIP = 0.50  # 50%
MIN_EPS_GROWTH     = 0.15  # +15%
MIN_REVENUE_GROWTH = 0.08  # +8%
MIN_EBITDA_MARGIN  = 0.15  # 15% (EBITDA / Revenue)

# Technical thresholds
RSI_UPPER            = 40
PRICE_VS_200MA_RANGE = 0.05
BB_NEAR_LOWER_THRESH = 0.10

# Option thresholds
DELTA_MIN         = 0.70
DELTA_MAX         = 0.85
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
# BLACK-SCHOLES GREEKS
# ------------------------------------------------------------------

def black_scholes_delta(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return float("nan")
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    return float(norm.cdf(d1))


def black_scholes_theta(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return float("nan")
    d1    = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2    = d1 - sigma * math.sqrt(T)
    term1 = -(S * norm.pdf(d1) * sigma) / (2 * math.sqrt(T))
    term2 = r * K * math.exp(-r * T) * norm.cdf(d2)
    return float((term1 - term2) / 365)

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


WEIGHTS = {
    "market_cap":      0.06,
    "avg_volume":      0.05,
    "inst_ownership":  0.06,
    "eps_growth":      0.07,
    "revenue_growth":  0.06,
    "ebitda_margin":   0.06,
    "rsi":             0.12,
    "price_vs_200ma":  0.10,
    "bb_position":     0.10,
    "delta":           0.20,
    "iv_percentile":   0.12,
}


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


def fetch_fundamental_data(ticker):
    info = ticker.info or {}
    return {
        "market_cap":     info.get("marketCap"),
        "avg_volume":     info.get("averageVolume"),
        "inst_ownership": info.get("heldPercentInstitutions"),
        "eps_growth":     fetch_eps_growth(ticker, info),
        "revenue_growth": info.get("revenueGrowth"),
        "ebitda_margin":  fetch_ebitda_margin(info),
    }


def passes_fundamental_screen(fund):
    mc = fund.get("market_cap")
    av = fund.get("avg_volume")
    io = fund.get("inst_ownership")
    eg = fund.get("eps_growth")
    rg = fund.get("revenue_growth")
    em = fund.get("ebitda_margin")

    if mc is None or mc < MIN_MARKET_CAP:
        label = "no data" if mc is None else ("$" + str(round(mc / 1e9, 1)) + "B < $2B")
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
# MAIN ANALYSIS FUNCTION
# ------------------------------------------------------------------

def analyze_ticker(symbol, idx, total):
    prefix = "  [" + str(idx).rjust(3) + "/" + str(total) + "] " + symbol.ljust(8)
    print(prefix, end="", flush=True)

    try:
        ticker = yf.Ticker(symbol)

        fund = fetch_fundamental_data(ticker)
        passed, reason = passes_fundamental_screen(fund)
        if not passed:
            print(" FAIL  " + reason)
            return []

        hist = ticker.history(period="1y")
        if hist.empty or len(hist) < 30:
            print(" FAIL  Insufficient price history")
            return []

        close         = hist["Close"]
        current_price = float(close.iloc[-1])

        rsi_val                       = calculate_rsi(close)
        bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(close)
        ma_200         = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else float(close.mean())
        bb_width       = (bb_upper - bb_lower) or 1
        bb_pos         = (current_price - bb_lower) / bb_width
        price_vs_200ma = (current_price - ma_200) / ma_200

        expirations = ticker.options
        if not expirations:
            print(" FAIL  No options data")
            return []

        expiry = expirations[-1]
        chain  = ticker.option_chain(expiry)
        calls  = chain.calls.copy()

        if calls.empty:
            print(" FAIL  No calls at " + expiry)
            return []

        exp_date = datetime.strptime(expiry, "%Y-%m-%d").date()
        dte      = max((exp_date - date.today()).days, 1)
        T        = dte / 365.0

        mid_iv        = float(calls["impliedVolatility"].median())
        iv_percentile = calculate_iv_percentile(close, mid_iv)

        param_scores_base = {
            "market_cap":     score_parameter(fund["market_cap"],     "market_cap"),
            "avg_volume":     score_parameter(fund["avg_volume"],      "avg_volume"),
            "inst_ownership": score_parameter(fund["inst_ownership"],  "inst_ownership"),
            "eps_growth":     score_parameter(fund["eps_growth"],      "eps_growth"),
            "revenue_growth": score_parameter(fund["revenue_growth"],  "revenue_growth"),
            "ebitda_margin":  score_parameter(fund["ebitda_margin"],   "ebitda_margin"),
            "rsi":            score_parameter(rsi_val,                 "rsi"),
            "price_vs_200ma": score_parameter(price_vs_200ma,          "price_vs_200ma"),
            "bb_position":    score_parameter(bb_pos,                  "bb_position"),
            "iv_percentile":  score_parameter(iv_percentile,           "iv_percentile"),
        }

        results = []
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

            param_scores = dict(param_scores_base)
            param_scores["delta"] = score_parameter(delta, "delta")

            results.append({
                "ticker":   symbol,
                "expiry":   expiry,
                "strike":   round(strike, 2),
                "contract": opt.get("contractSymbol", ""),

                "current_price":      round(current_price, 2),
                "market_cap":         fund["market_cap"],
                "avg_volume":         fund["avg_volume"],
                "inst_ownership":     round(fund["inst_ownership"] * 100, 1)  if fund["inst_ownership"]  is not None else None,
                "eps_growth_pct":     round(fund["eps_growth"] * 100, 1)      if fund["eps_growth"]      is not None else None,
                "revenue_growth_pct": round(fund["revenue_growth"] * 100, 1)  if fund["revenue_growth"]  is not None else None,
                "ebitda_margin_pct":  round(fund["ebitda_margin"] * 100, 1)   if fund["ebitda_margin"]   is not None else None,

                "rsi":                round(rsi_val, 2),
                "price_vs_200ma_pct": round(price_vs_200ma * 100, 2),
                "bb_lower":           round(bb_lower, 2),
                "bb_middle":          round(bb_middle, 2),
                "bb_upper":           round(bb_upper, 2),
                "bb_position_pct":    round(bb_pos * 100, 1),

                "premium":                round(prem, 2),
                "implied_volatility_pct": round(iv * 100, 2),
                "iv_percentile":          round(iv_percentile, 1),
                "dte":                    dte,

                "delta":       round(delta, 4),
                "theta_daily": round(theta, 4),

                "required_stock_move": round(required_move, 2)  if not math.isnan(required_move)  else None,
                "leverage_ratio":      round(leverage_ratio, 2) if not math.isnan(leverage_ratio) else None,
                "breakeven_price":     round(breakeven, 2),

                "scores":          dict((k, round(v, 1)) for k, v in param_scores.items()),
                "composite_score": compute_composite_score(param_scores),
            })

        results.sort(key=lambda x: x["composite_score"], reverse=True)

        rg_s = str(round(fund["revenue_growth"] * 100)) + "%" if fund["revenue_growth"] is not None else "?"
        em_s = str(round(fund["ebitda_margin"] * 100)) + "%"  if fund["ebitda_margin"]  is not None else "?"
        print(" PASS  " + str(len(results)) + " opts  RevGrowth=" + rg_s + "  EBITDA=" + em_s + "  " + expiry)

        time.sleep(DELAY_SECONDS)
        return results

    except Exception as e:
        print(" ERROR  " + str(e))
        return []

# ------------------------------------------------------------------
# OUTPUT HELPERS
# ------------------------------------------------------------------

def save_json(data, filename):
    path = OUTPUT_DIR + "/" + filename
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print("  Saved: " + path)

# ------------------------------------------------------------------
# ENTRY POINT
# ------------------------------------------------------------------

def main():
    print("=================================================================")
    print("  LEAP CALL OPTIONS SCREENER")
    print("  Run date : " + str(date.today()))
    print("  Tickers  : " + str(len(WATCHLIST)))
    print("=================================================================")
    print("")
    print("Fundamental criteria:")
    print("  Market Cap       > $25B")
    print("  Avg Volume       > 1M shares/day")
    print("  Inst. Ownership  > 50%")
    print("  EPS Growth       > +15%")
    print("  Revenue Growth   > +8%")
    print("  EBITDA Margin    > 15%")
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
    top_20_overall = all_results[:20]

    run_meta = {
        "generated_at":   datetime.utcnow().isoformat() + "Z",
        "watchlist":      WATCHLIST,
        "tickers_passed": len(passed_tickers),
        "tickers_list":   passed_tickers,
        "screening_parameters": {
            "min_market_cap_usd":     MIN_MARKET_CAP,
            "min_avg_volume_shares":  MIN_AVG_VOLUME,
            "min_inst_ownership_pct": MIN_INST_OWNERSHIP * 100,
            "min_eps_growth_pct":     MIN_EPS_GROWTH * 100,
            "min_revenue_growth_pct": MIN_REVENUE_GROWTH * 100,
            "min_ebitda_margin_pct":  MIN_EBITDA_MARGIN * 100,
            "rsi_upper":              RSI_UPPER,
            "price_vs_200ma_range":   "+-" + str(PRICE_VS_200MA_RANGE * 100) + "%",
            "delta_range":            [DELTA_MIN, DELTA_MAX],
            "max_iv_percentile":      MAX_IV_PERCENTILE,
        },
        "score_weights": WEIGHTS,
    }

    print("")
    print(str(len(passed_tickers)) + "/" + str(total) + " tickers passed.")
    print("Saving output files...")

    save_json({"meta": run_meta, "top_20_per_stock":     per_stock},      "top_20_per_stock.json")
    save_json({"meta": run_meta, "top_20_overall":       top_20_overall}, "top_20_overall.json")
    save_json({"meta": run_meta, "all_screened_options": all_results},    "full_results.json")

    print("")
    print("=================================================================")
    print("  TOP 20 OVERALL")
    print("=================================================================")
    for i, opt in enumerate(top_20_overall, 1):
        print(
            str(i) + ". " + opt["ticker"] +
            "  Strike=" + str(opt["strike"]) +
            "  Expiry=" + opt["expiry"] +
            "  Prem=$" + str(opt["premium"]) +
            "  Delta=" + str(opt["delta"]) +
            "  Score=" + str(opt["composite_score"])
        )
    print("=================================================================")
    print("Done.")


if __name__ == "__main__":
    main()
