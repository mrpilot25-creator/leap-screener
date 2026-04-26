"""
diagnostic.py
=============
Runs through the first 50 S&P 500 tickers and prints exactly what
Yahoo Finance returns for each of the four fundamental fields,
and which filter is causing the fail.

Run this BEFORE the full screener to understand the data quality issues.

Usage:
    python diagnostic.py
"""

import time
import yfinance as yf

TICKERS_TO_TEST = [
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA", "JPM",
    "LLY", "V", "UNH", "XOM", "MA", "JNJ", "PG", "AVGO", "HD", "MRK",
    "COST", "ABBV", "CVX", "KO", "PEP", "ADBE", "WMT", "BAC", "MCD",
    "CSCO", "CRM", "TMO", "ABT", "ACN", "LIN", "NFLX", "DHR", "TXN",
    "CMCSA", "VZ", "NEE", "PM", "RTX", "ORCL", "QCOM", "HON", "GS",
    "CAT", "SBUX", "BLK", "SPGI", "AXP",
]

MIN_MARKET_CAP     = 2e9
MIN_AVG_VOLUME     = 1e6
MIN_INST_OWNERSHIP = 0.50
MIN_EPS_GROWTH     = 0.15

print(f"{'Ticker':<8} {'MarketCap':>12} {'AvgVol':>10} {'InstOwn':>9} {'EPSGrowth':>10}  {'Result'}")
print("-" * 75)

pass_count = 0
fail_reasons = {"MarketCap": 0, "AvgVol": 0, "InstOwn_None": 0, "InstOwn_Low": 0,
                "EPSGrowth_None": 0, "EPSGrowth_Low": 0, "PASS": 0}

for symbol in TICKERS_TO_TEST:
    try:
        info = yf.Ticker(symbol).info or {}

        mc = info.get("marketCap")
        av = info.get("averageVolume")
        io = info.get("heldPercentInstitutions")
        eg = info.get("earningsGrowth") or info.get("earningsQuarterlyGrowth")

        mc_s = f"${mc/1e9:.0f}B"  if mc else "None"
        av_s = f"{av/1e6:.1f}M"   if av else "None"
        io_s = f"{io*100:.0f}%"   if io else "None"
        eg_s = f"{eg*100:.0f}%"   if eg else "None"

        # Determine result
        fails = []
        if mc is None or mc < MIN_MARKET_CAP:
            fails.append("MarketCap")
            fail_reasons["MarketCap"] += 1
        if av is None or av < MIN_AVG_VOLUME:
            fails.append("AvgVol")
            fail_reasons["AvgVol"] += 1
        if io is None:
            fails.append("InstOwn=None")
            fail_reasons["InstOwn_None"] += 1
        elif io < MIN_INST_OWNERSHIP:
            fails.append(f"InstOwn<50%")
            fail_reasons["InstOwn_Low"] += 1
        if eg is None:
            fails.append("EPSGrowth=None")
            fail_reasons["EPSGrowth_None"] += 1
        elif eg < MIN_EPS_GROWTH:
            fails.append(f"EPSGrowth<15%")
            fail_reasons["EPSGrowth_Low"] += 1

        result = "PASS" if not fails else "FAIL: " + ", ".join(fails)
        if not fails:
            fail_reasons["PASS"] += 1

        print(f"{symbol:<8} {mc_s:>12} {av_s:>10} {io_s:>9} {eg_s:>10}  {result}")
        time.sleep(0.3)

    except Exception as e:
        print(f"{symbol:<8} {'ERROR':>12} {'':>10} {'':>9} {'':>10}  {e}")

print("\n" + "=" * 75)
print("SUMMARY OF FAILURES ACROSS TESTED TICKERS:")
print(f"  PASS                : {fail_reasons['PASS']}")
print(f"  Failed MarketCap    : {fail_reasons['MarketCap']}")
print(f"  Failed AvgVol       : {fail_reasons['AvgVol']}")
print(f"  InstOwn = None      : {fail_reasons['InstOwn_None']}  ← likely main culprit")
print(f"  InstOwn < 50%       : {fail_reasons['InstOwn_Low']}")
print(f"  EPSGrowth = None    : {fail_reasons['EPSGrowth_None']}")
print(f"  EPSGrowth < 15%     : {fail_reasons['EPSGrowth_Low']}")
print("=" * 75)
