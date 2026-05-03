# LEAP Call Options Screener + Walk-Forward Backtesting Suite
# ===========================================================
# Screens a manual watchlist of stocks for the best LEAP call options,
# then runs a 5-year walk-forward backtest on the top candidates.
#
# Outputs:
#   top_20_per_stock.json   - best 20 options for each ticker
#   top_20_overall.json     - best 20 options across all tickers
#   full_results.json       - every screened option with all metrics
#   backtest_results.json     - walk-forward backtest report
#   forward_projections.json  - forward price projections to expiry
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
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
    "META", "TSLA", "JPM", "UNH", "V",
]

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------

RISK_FREE_RATE   = 0.0388  # 2-Year Treasury proxy
OUTPUT_DIR       = "."
DELAY_SECONDS    = 0.2
RUN_BACKTEST          = True    # Set to False to skip the backtest module
RUN_FORWARD_PROJECTION = True  # Set to False to skip forward projections

# Backtest config
BT_TRAINING_YEARS    = 3
BT_VALIDATION_YEARS  = 2
BT_N_SIMULATIONS     = 1000
BT_TOP_N_CANDIDATES  = 999  # Backtest ALL passing tickers (one per ticker, best option)
FP_N_SIMULATIONS     = 1000  # Monte Carlo paths for forward projection

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
    term2 = r * K * math.exp(-r * T) * norm.cdf(d2)
    return float((term1 - term2) / 365)


def black_scholes_rho(S, K, T, r, sigma):
    # Rho: sensitivity of call price to interest rate change
    # Rho = K * T * exp(-r*T) * N(d2)
    _, d2 = bs_d1_d2(S, K, T, r, sigma)
    if math.isnan(d2):
        return float("nan")
    return float(K * T * math.exp(-r * T) * norm.cdf(d2) / 100)

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
      1. Fetch 5 years of daily price + dividend data.
      2. Split into 3-year Training Window and 2-year Validation Window.
      3. Estimate drift (mu) from training data, adjusting for dividend yield.
      4. Estimate sigma from training data; apply mean-reverting IV overlay.
      5. Run vectorized Monte Carlo simulation (GBM) over the validation horizon.
      6. Compare simulated paths to actual validation prices (Reality Check).
      7. Calculate Rho and Theta over the validation period.
      8. Produce a Residual Variance Report.
    """

    def __init__(self, symbol, strike, entry_score,
                 training_years=3, validation_years=2,
                 n_simulations=1000, risk_free_rate=0.0388):
        self.symbol           = symbol
        self.strike           = strike
        self.entry_score      = entry_score
        self.training_years   = training_years
        self.validation_years = validation_years
        self.n_sims           = n_simulations
        self.r                = risk_free_rate
        self.total_years      = training_years + validation_years

        # Populated by load_data()
        self.train_close  = None
        self.val_close    = None
        self.entry_price  = None
        self.div_yield    = None

        # Populated by calibrate()
        self.mu           = None
        self.sigma_train  = None
        self.sigma_val    = None
        self.sigma_blended = None

        # Populated by run_monte_carlo()
        self.paths        = None   # shape: (n_sims, val_days)

    # -- 1. Data Loading ----------------------------------------------

    def load_data(self):
        ticker = yf.Ticker(self.symbol)
        hist   = ticker.history(period=str(self.total_years + 1) + "y")

        if hist.empty or len(hist) < 252 * self.total_years:
            raise ValueError("Insufficient historical data for " + self.symbol)

        close = hist["Close"].dropna()

        # Split point: training_years back from end
        split_idx = len(close) - int(252 * self.validation_years)
        if split_idx < 252:
            raise ValueError("Not enough training data for " + self.symbol)

        self.train_close = close.iloc[:split_idx]
        self.val_close   = close.iloc[split_idx:]
        self.entry_price = float(self.train_close.iloc[-1])

        # Dividend yield from training period
        try:
            divs = ticker.dividends
            if not divs.empty:
                # Filter dividends to training window
                train_start = self.train_close.index[0]
                train_end   = self.train_close.index[-1]
                divs_train  = divs[
                    (divs.index >= train_start) & (divs.index <= train_end)
                ]
                annual_div   = float(divs_train.sum()) / self.training_years
                self.div_yield = annual_div / self.entry_price
            else:
                self.div_yield = 0.0
        except Exception:
            self.div_yield = 0.0

        return self

    # -- 2. Calibration -----------------------------------------------

    def calibrate(self):
        # Log returns from training window
        log_rets = np.log(
            self.train_close.values[1:] / self.train_close.values[:-1]
        )

        # Annualised mean and volatility from training
        mu_daily          = float(np.mean(log_rets))
        sigma_daily_train = float(np.std(log_rets, ddof=1))
        self.sigma_train  = sigma_daily_train * np.sqrt(252)

        # Drift = annualised mean - dividend yield
        # Under risk-neutral measure we use r - q, but for historical
        # projection we use empirical mu - q
        mu_annual = mu_daily * 252
        self.mu   = mu_annual - self.div_yield

        # Actual volatility in validation period (for reality check)
        val_log_rets      = np.log(
            self.val_close.values[1:] / self.val_close.values[:-1]
        )
        sigma_daily_val   = float(np.std(val_log_rets, ddof=1))
        self.sigma_val    = sigma_daily_val * np.sqrt(252)

        # Mean-reverting IV overlay:
        # Blend training sigma toward long-run mean (sigma_val) using
        # an Ornstein-Uhlenbeck inspired decay with kappa = 2.0
        kappa             = 2.0
        T_val             = self.validation_years
        weight            = math.exp(-kappa * T_val)
        self.sigma_blended = (
            weight * self.sigma_train + (1 - weight) * self.sigma_val
        )

        return self

    # -- 3. Monte Carlo Simulation (vectorized GBM) -------------------

    def run_monte_carlo(self):
        n_days   = len(self.val_close)
        dt       = 1.0 / 252.0
        S0       = self.entry_price
        mu       = self.mu
        sigma    = self.sigma_blended

        # Vectorized GBM:
        # Shape of Z: (n_sims, n_days)
        # Each row is one simulation path
        np.random.seed(42)   # reproducibility
        Z = np.random.standard_normal((self.n_sims, n_days))

        # Incremental log returns for each step
        drift     = (mu - 0.5 * sigma ** 2) * dt
        diffusion = sigma * math.sqrt(dt) * Z

        # Cumulative sum gives log price path; exp gives price path
        log_paths   = np.cumsum(drift + diffusion, axis=1)
        self.paths  = S0 * np.exp(log_paths)  # shape: (n_sims, n_days)

        return self

    # -- 4. Reality Comparison ----------------------------------------

    def reality_check(self):
        actual_prices = self.val_close.values   # shape: (n_days,)

        # Terminal prices
        sim_terminal    = self.paths[:, -1]     # shape: (n_sims,)
        actual_terminal = float(actual_prices[-1])
        median_sim_term = float(np.median(sim_terminal))
        pct_5_term      = float(np.percentile(sim_terminal, 5))
        pct_95_term     = float(np.percentile(sim_terminal, 95))

        # Max drawdown per simulation path (vectorized)
        cum_max   = np.maximum.accumulate(self.paths, axis=1)
        drawdowns = (self.paths - cum_max) / cum_max   # shape: (n_sims, n_days)
        sim_max_dd = float(np.mean(np.min(drawdowns, axis=1)))

        # Actual max drawdown
        actual_cum_max = np.maximum.accumulate(actual_prices)
        actual_dd_series = (actual_prices - actual_cum_max) / actual_cum_max
        actual_max_dd  = float(np.min(actual_dd_series))

        # Reality option premium:
        # Use actual HV of validation period + 3% IV-HV spread
        reality_iv = self.sigma_val + 0.03
        T_remaining = self.validation_years
        K = self.strike
        S_entry = self.entry_price

        reality_premium = black_scholes_price(
            S_entry, K, T_remaining, self.r, reality_iv
        )

        return {
            "actual_terminal_price":   round(actual_terminal, 2),
            "median_simulated_terminal": round(median_sim_term, 2),
            "sim_terminal_5th_pct":    round(pct_5_term, 2),
            "sim_terminal_95th_pct":   round(pct_95_term, 2),
            "terminal_price_error_pct": round(
                (median_sim_term - actual_terminal) / actual_terminal * 100, 2
            ),
            "sim_avg_max_drawdown_pct":  round(sim_max_dd * 100, 2),
            "actual_max_drawdown_pct":   round(actual_max_dd * 100, 2),
            "drawdown_error_pct":        round(
                (sim_max_dd - actual_max_dd) / abs(actual_max_dd) * 100
                if actual_max_dd != 0 else 0.0, 2
            ),
            "reality_iv_used_pct":      round(reality_iv * 100, 2),
            "reality_option_premium":   round(reality_premium, 2)
                                        if not math.isnan(reality_premium) else None,
        }

    # -- 5. Greek Attribution Over Validation Period ------------------

    def greek_attribution(self):
        K      = self.strike
        S      = self.entry_price
        r      = self.r
        sigma  = self.sigma_blended
        T_full = float(self.validation_years)

        # Sample greeks at entry, mid-point, and expiry approach
        checkpoints = [T_full, T_full / 2, 0.25]
        greek_timeline = []

        for T in checkpoints:
            if T <= 0:
                continue
            delta_t = black_scholes_delta(S, K, T, r, sigma)
            theta_t = black_scholes_theta(S, K, T, r, sigma)
            rho_t   = black_scholes_rho(S, K, T, r, sigma)
            price_t = black_scholes_price(S, K, T, r, sigma)

            greek_timeline.append({
                "years_to_expiry":    round(T, 2),
                "option_price":       round(price_t, 2) if not math.isnan(price_t) else None,
                "delta":              round(delta_t, 4) if not math.isnan(delta_t) else None,
                "theta_daily":        round(theta_t, 4) if not math.isnan(theta_t) else None,
                "rho_per_1pct_rate":  round(rho_t, 4)  if not math.isnan(rho_t)   else None,
            })

        # Cumulative theta cost over validation period
        # Approximate: average daily theta * validation days
        theta_entry = black_scholes_theta(S, K, T_full, r, sigma)
        theta_mid   = black_scholes_theta(S, K, T_full / 2, r, sigma)
        avg_theta   = (theta_entry + theta_mid) / 2 if not math.isnan(theta_entry) else 0.0
        val_days    = int(self.validation_years * 365)
        cumulative_theta_cost = round(abs(avg_theta) * val_days, 2)

        return {
            "greek_timeline":         greek_timeline,
            "cumulative_theta_cost":  cumulative_theta_cost,
        }

    # -- 6. Residual Variance Report & EV ----------------------------

    def residual_variance_report(self, reality):
        K       = self.strike
        sigma   = self.sigma_blended
        T_val   = float(self.validation_years)
        r       = self.r
        S_entry = self.entry_price

        # Projected terminal price distribution
        sim_terminal = self.paths[:, -1]

        # ITM probability: fraction of paths where terminal > strike
        n_itm     = float(np.sum(sim_terminal > K))
        p_itm     = n_itm / self.n_sims

        # Expected payoff when ITM
        payoffs_itm = sim_terminal[sim_terminal > K] - K
        ev_payoff   = float(np.mean(payoffs_itm)) if len(payoffs_itm) > 0 else 0.0

        # Entry premium (projected)
        entry_premium = black_scholes_price(S_entry, K, T_val, r, sigma)
        if math.isnan(entry_premium):
            entry_premium = 0.0

        # EV = P(ITM) * E[payoff|ITM] - P(OTM) * premium_paid
        p_otm  = 1.0 - p_itm
        ev     = p_itm * ev_payoff - p_otm * entry_premium

        # Residual variance (squared error of median sim vs actual)
        actual_terminal  = reality["actual_terminal_price"]
        median_projected = reality["median_simulated_terminal"]
        residual_var     = (median_projected - actual_terminal) ** 2

        # Confidence band: fraction of actual prices within sim 5th-95th pct
        actual_prices = self.val_close.values
        sim_5th  = np.percentile(self.paths, 5,  axis=0)
        sim_95th = np.percentile(self.paths, 95, axis=0)
        n_within = int(np.sum(
            (actual_prices >= sim_5th) & (actual_prices <= sim_95th)
        ))
        pct_within_band = round(n_within / len(actual_prices) * 100, 1)

        return {
            "entry_price":              round(S_entry, 2),
            "strike":                   round(K, 2),
            "projected_entry_premium":  round(entry_premium, 2),
            "reality_premium":          reality["reality_option_premium"],
            "p_itm_pct":                round(p_itm * 100, 1),
            "p_otm_pct":                round(p_otm * 100, 1),
            "expected_payoff_if_itm":   round(ev_payoff, 2),
            "expected_value_ev":        round(ev, 2),
            "ev_positive":              ev > 0,
            "residual_variance":        round(residual_var, 4),
            "pct_actual_within_90_band": pct_within_band,
            "terminal_price_error_pct": reality["terminal_price_error_pct"],
            "drawdown_error_pct":       reality["drawdown_error_pct"],
        }

    # -- 7. Full Run --------------------------------------------------

    def run(self):
        try:
            print("    Loading data...")
            self.load_data()

            print("    Calibrating GBM parameters...")
            self.calibrate()

            print("    Running " + str(self.n_sims) + " Monte Carlo paths...")
            self.run_monte_carlo()

            print("    Comparing projections to reality...")
            reality = self.reality_check()

            print("    Calculating greek attribution...")
            greeks = self.greek_attribution()

            print("    Building residual variance report...")
            rv_report = self.residual_variance_report(reality)

            return {
                "symbol":            self.symbol,
                "strike":            self.strike,
                "entry_score":       self.entry_score,
                "entry_price":       round(self.entry_price, 2),
                "training_years":    self.training_years,
                "validation_years":  self.validation_years,
                "n_simulations":     self.n_sims,
                "calibration": {
                    "annualised_mu_pct":      round(self.mu * 100, 2),
                    "dividend_yield_pct":     round(self.div_yield * 100, 2),
                    "training_sigma_pct":     round(self.sigma_train * 100, 2),
                    "validation_sigma_pct":   round(self.sigma_val * 100, 2),
                    "blended_sigma_pct":      round(self.sigma_blended * 100, 2),
                    "risk_free_rate_pct":     round(self.r * 100, 2),
                },
                "reality_check":          reality,
                "greek_attribution":      greeks,
                "residual_variance_report": rv_report,
                "status": "ok",
            }

        except Exception as e:
            return {
                "symbol":  self.symbol,
                "strike":  self.strike,
                "status":  "error",
                "error":   str(e),
            }

# ------------------------------------------------------------------
# SCREENER - ANALYZE TICKER
# ------------------------------------------------------------------

# ------------------------------------------------------------------
# FORWARD PROJECTION ENGINE
# ------------------------------------------------------------------

class ForwardProjectionEngine:
    """
    Projects the stock price from TODAY to the option expiry date
    using Geometric Brownian Motion calibrated on 1 year of recent
    price history, with implied volatility as the simulation sigma.

    This is entirely forward-looking and completely separate from
    the walk-forward backtest (which is historical validation).

    For each option that passes the screener, the engine:
      1. Fetches 1yr of price history to estimate historical drift (mu).
      2. Uses the option's implied volatility as sigma (most current
         market estimate of future price movement).
      3. Runs 1,000 vectorized GBM paths from today to expiry.
      4. Reports the projected price distribution at expiry,
         P(ITM), projected option value, and Expected Value (EV).
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

        exp_date   = datetime.strptime(expiry_str, "%Y-%m-%d").date()
        self.T     = max((exp_date - date.today()).days, 1) / 365.0
        self.n_days = max((exp_date - date.today()).days, 1)

        self.mu       = None
        self.div_yield = 0.0
        self.paths    = None

    def calibrate(self):
        """
        Estimate historical drift and dividend yield from 3 years of data.

        Three fixes vs the naive 1-year approach:

        1. LONGER LOOKBACK (3yr not 1yr)
           A 1-year window is too sensitive to whether the recent year was
           a good or bad one. 3 years captures a full market cycle and
           gives a more representative long-run drift.

        2. DRIFT FLOOR = max(historical mu, risk-free rate)
           In log-normal GBM the median price = S0 * exp((mu - sigma^2/2) * T).
           When sigma is large (e.g. 49%) and T is long (e.g. 2.6yr), the
           volatility drag term (sigma^2/2 = 12%/yr) can easily overwhelm a
           low drift, pulling the median projection below the current price
           even for a fundamentally strong stock. Flooring at the risk-free
           rate (3.88%) is theoretically grounded: in efficient markets,
           equities must earn at least the risk-free rate in expectation.

        3. CAPPED IV AT 40% FOR LONG-DATED PROJECTIONS
           Options with very high IV (e.g. 49%) are often driven by short-term
           event risk (earnings, macro). For a 2+ year price path projection,
           this overstates likely vol. Capping at 40% for T > 1yr reduces the
           vol drag to a more realistic 8%/yr and produces price distributions
           that better match long-run analyst expectations.
        """
        try:
            ticker   = yf.Ticker(self.symbol)
            hist     = ticker.history(period="3y")   # 3yr not 1yr
            close    = hist["Close"].dropna()

            if len(close) >= 60:
                log_rets      = np.log(close.values[1:] / close.values[:-1])
                mu_daily      = float(np.mean(log_rets))
                self.mu_hist  = mu_daily * 252
            else:
                self.mu_hist  = self.r

            # Apply drift floor: never project below risk-free rate drift
            self.mu_hist = max(self.mu_hist, self.r)

            # Dividend yield
            try:
                divs       = ticker.dividends
                if not divs.empty:
                    annual_div = float(divs.tail(4).sum())
                    self.div_yield = annual_div / self.S0
            except Exception:
                pass

            # Adjusted drift: historical mu minus dividend yield
            # Final floor at 0: drift must be non-negative
            self.mu = max(self.mu_hist - self.div_yield, 0.0)

        except Exception:
            self.mu = self.r
            self.mu_hist = self.r

        # Cap IV at 40% for projections longer than 1 year to reduce
        # excess volatility drag on long-dated median price estimates
        if self.T > 1.0 and self.iv > 0.40:
            self.iv_capped = 0.40
        else:
            self.iv_capped = self.iv

        return self

    def run_simulation(self):
        """
        Run 1,000 GBM paths from today to expiry.
        Uses capped IV as sigma for long-dated projections.
        Shape of paths: (n_sims, n_days)
        """
        dt     = 1.0 / 252.0
        mu     = self.mu
        sigma  = self.iv_capped   # capped IV, not raw IV

        np.random.seed(99)
        Z = np.random.standard_normal((self.n_sims, self.n_days))

        drift     = (mu - 0.5 * sigma ** 2) * dt
        diffusion = sigma * math.sqrt(dt) * Z

        log_paths  = np.cumsum(drift + diffusion, axis=1)
        self.paths = self.S0 * np.exp(log_paths)
        return self

    def projection_report(self):
        """Build the full forward projection report."""
        terminal = self.paths[:, -1]   # price at expiry for each path

        median_price = float(np.median(terminal))
        pct5         = float(np.percentile(terminal, 5))
        pct25        = float(np.percentile(terminal, 25))
        pct75        = float(np.percentile(terminal, 75))
        pct95        = float(np.percentile(terminal, 95))

        # P(ITM): fraction of paths ending above strike
        n_itm = int(np.sum(terminal > self.K))
        p_itm = n_itm / self.n_sims

        # Expected payoff when ITM (intrinsic value at expiry)
        payoffs_itm = terminal[terminal > self.K] - self.K
        ev_payoff   = float(np.mean(payoffs_itm)) if len(payoffs_itm) > 0 else 0.0

        # Discount payoff back to today (present value)
        pv_payoff   = ev_payoff * math.exp(-self.r * self.T)

        # EV = P(ITM) * E[payoff|ITM] - P(OTM) * premium paid
        p_otm = 1.0 - p_itm
        ev    = p_itm * pv_payoff - p_otm * self.current_premium

        # Projected option value at expiry using median terminal price via BS
        projected_bs = black_scholes_price(
            median_price, self.K, 0.01, self.r, self.iv_capped
        )

        # Upside / downside from current price
        upside_median = (median_price - self.S0) / self.S0 * 100
        upside_pct95  = (pct95 - self.S0) / self.S0 * 100
        downside_pct5 = (pct5 - self.S0) / self.S0 * 100

        return {
            "symbol":               self.symbol,
            "current_price":        round(self.S0, 2),
            "strike":               round(self.K, 2),
            "expiry":               self.expiry_str,
            "days_to_expiry":       self.n_days,
            "years_to_expiry":      round(self.T, 3),
            "current_premium":      round(self.current_premium, 2),
            "implied_vol_raw_pct":  round(self.iv * 100, 2),
            "implied_vol_used_pct": round(self.iv_capped * 100, 2),
            "iv_was_capped":        self.iv_capped < self.iv,
            "historical_mu_pct":    round(self.mu_hist * 100, 2)
                                    if self.mu_hist else None,
            "div_yield_pct":        round(self.div_yield * 100, 2),
            "calibrated_drift_pct": round(self.mu * 100, 2)
                                    if self.mu else None,
            "n_simulations":        self.n_sims,
            "projected_prices": {
                "median":   round(median_price, 2),
                "pct_5":    round(pct5, 2),
                "pct_25":   round(pct25, 2),
                "pct_75":   round(pct75, 2),
                "pct_95":   round(pct95, 2),
            },
            "upside_downside": {
                "median_change_pct":  round(upside_median, 1),
                "pct95_change_pct":   round(upside_pct95, 1),
                "pct5_change_pct":    round(downside_pct5, 1),
            },
            "p_itm_pct":                round(p_itm * 100, 1),
            "p_otm_pct":                round(p_otm * 100, 1),
            "expected_payoff_if_itm":   round(ev_payoff, 2),
            "pv_expected_payoff":       round(pv_payoff, 2),
            "expected_value_ev":        round(ev, 2),
            "ev_positive":              ev > 0,
            "projected_option_value":   round(projected_bs, 2)
                                        if not math.isnan(projected_bs) else None,
            "status": "ok",
        }

    def run(self):
        try:
            self.calibrate()
            self.run_simulation()
            return self.projection_report()
        except Exception as e:
            return {
                "symbol":  self.symbol,
                "strike":  self.K,
                "expiry":  self.expiry_str,
                "status":  "error",
                "error":   str(e),
            }

# ------------------------------------------------------------------
# FORWARD PROJECTION RUNNER
# ------------------------------------------------------------------

def run_forward_projections(all_results):
    """
    For each ticker that passed the screener, take its single
    highest-scoring option and run a forward price projection
    from today to the option's expiry date.
    """
    print("")
    print("=================================================================")
    print("  FORWARD PRICE PROJECTIONS  (today to expiry via Monte Carlo)")
    print("=================================================================")

    if not all_results:
        print("  No screener results to project.")
        return []

    # Pick the best option per ticker (highest composite score)
    best_per_ticker = {}
    for opt in sorted(all_results, key=lambda x: x["composite_score"],
                      reverse=True):
        sym = opt["ticker"]
        if sym not in best_per_ticker:
            best_per_ticker[sym] = opt

    projections = []
    tickers = list(best_per_ticker.keys())

    for i, sym in enumerate(tickers, 1):
        opt = best_per_ticker[sym]
        print("  [" + str(i).rjust(2) + "/" + str(len(tickers)) + "] " +
              sym.ljust(8) + " Strike=" + str(opt["strike"]) +
              "  Expiry=" + opt["expiry"] +
              "  DTE=" + str(opt["dte"]) + " days",
              end="", flush=True)

        engine = ForwardProjectionEngine(
            symbol          = sym,
            current_price   = opt["current_price"],
            strike          = opt["strike"],
            expiry_str      = opt["expiry"],
            implied_vol     = opt["implied_volatility_pct"] / 100.0,
            current_premium = opt["premium"],
            risk_free_rate  = RISK_FREE_RATE,
            n_simulations   = FP_N_SIMULATIONS,
        )

        result = engine.run()

        if result["status"] == "ok":
            ev_label = "EV+" if result["ev_positive"] else "EV-"
            print("  DONE  Median=$" + str(result["projected_prices"]["median"]) +
                  "  P(ITM)=" + str(result["p_itm_pct"]) + "%" +
                  "  " + ev_label)
        else:
            print("  ERROR  " + result.get("error", "unknown"))

        projections.append(result)
        time.sleep(DELAY_SECONDS)

    return projections

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
# BACKTEST RUNNER - Uses screener composite score as entry signal
# ------------------------------------------------------------------

def run_backtest(all_results):
    print("")
    print("=================================================================")
    print("  WALK-FORWARD BACKTEST  (" + str(BT_TRAINING_YEARS) + "yr train / " +
          str(BT_VALIDATION_YEARS) + "yr validate / " +
          str(BT_N_SIMULATIONS) + " simulations)")
    print("  Backtesting best option for EVERY passing ticker")
    print("=================================================================")

    if not all_results:
        print("  No screener results to backtest.")
        return []

    # For each unique ticker that passed the screen, select the single
    # option with the highest composite score. This is the entry signal.
    seen = {}
    for opt in sorted(all_results, key=lambda x: x["composite_score"], reverse=True):
        sym = opt["ticker"]
        if sym not in seen:
            seen[sym] = opt
    # No top_n limit - backtest every ticker that passed the screen

    candidates = list(seen.values())
    print("  Tickers to backtest: " + str(len(candidates)))
    for c in candidates:
        print("    " + c["ticker"] + "  Strike=" + str(c["strike"]) +
              "  Score=" + str(c["composite_score"]))
    print("")

    bt_results = []
    for i, cand in enumerate(candidates, 1):
        symbol = cand["ticker"]
        strike = cand["strike"]
        score  = cand["composite_score"]

        print("  [" + str(i).rjust(2) + "/" + str(len(candidates)) + "] " +
              symbol + "  strike=" + str(strike) + "  score=" + str(score))

        engine = BacktestEngine(
            symbol           = symbol,
            strike           = strike,
            entry_score      = score,
            training_years   = BT_TRAINING_YEARS,
            validation_years = BT_VALIDATION_YEARS,
            n_simulations    = BT_N_SIMULATIONS,
            risk_free_rate   = RISK_FREE_RATE,
        )

        result = engine.run()

        if result["status"] == "ok":
            rv = result["residual_variance_report"]
            ev_label = "EV+" if rv["ev_positive"] else "EV-"
            print("    " + ev_label +
                  "  P(ITM)=" + str(rv["p_itm_pct"]) + "%" +
                  "  EV=$" + str(rv["expected_value_ev"]) +
                  "  TermErr=" + str(rv["terminal_price_error_pct"]) + "%")
        else:
            print("    ERROR: " + result.get("error", "unknown"))

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

    # Run backtest if enabled
    if RUN_BACKTEST and all_results:
        bt_results = run_backtest(all_results)

        bt_meta = {
            "generated_at":      datetime.utcnow().isoformat() + "Z",
            "training_years":    BT_TRAINING_YEARS,
            "validation_years":  BT_VALIDATION_YEARS,
            "n_simulations":     BT_N_SIMULATIONS,
            "risk_free_rate_pct": RISK_FREE_RATE * 100,
            "tickers_backtested":    len(bt_results),
            "entry_signal":          "composite_score (best option per ticker, all passing tickers)",
        }

        save_json(
            {"meta": bt_meta, "backtest_results": bt_results},
            "backtest_results.json"
        )

        print("")
        print("=================================================================")
        print("  BACKTEST SUMMARY")
        print("=================================================================")
        for r in bt_results:
            if r["status"] == "ok":
                rv = r["residual_variance_report"]
                cal = r["calibration"]
                print("  " + r["symbol"] + " | Strike=" + str(r["strike"]) +
                      " | Score=" + str(r["entry_score"]))
                print("    Sigma(train)=" + str(cal["training_sigma_pct"]) + "%" +
                      "  Sigma(val)=" + str(cal["validation_sigma_pct"]) + "%" +
                      "  DivYield=" + str(cal["dividend_yield_pct"]) + "%")
                print("    P(ITM)=" + str(rv["p_itm_pct"]) + "%" +
                      "  EV=$" + str(rv["expected_value_ev"]) +
                      "  TermErr=" + str(rv["terminal_price_error_pct"]) + "%" +
                      "  DDErr=" + str(rv["drawdown_error_pct"]) + "%")
                print("    Actual in 90pct band: " + str(rv["pct_actual_within_90_band"]) + "%" +
                      "  EV Positive: " + str(rv["ev_positive"]))
            else:
                print("  " + r["symbol"] + " | ERROR: " + r.get("error", "unknown"))
            print("")
        print("=================================================================")


    # Run forward projections if enabled
    if RUN_FORWARD_PROJECTION and all_results:
        fp_results = run_forward_projections(all_results)

        fp_meta = {
            "generated_at":      datetime.utcnow().isoformat() + "Z",
            "projection_date":   str(date.today()),
            "n_simulations":     FP_N_SIMULATIONS,
            "risk_free_rate_pct": RISK_FREE_RATE * 100,
            "note": ("Forward projections run from today to each option expiry "
                     "using implied volatility as sigma and historical mu as drift. "
                     "One projection per ticker using the highest-scoring option."),
        }

        save_json(
            {"meta": fp_meta, "forward_projections": fp_results},
            "forward_projections.json"
        )

        print("")
        print("=================================================================")
        print("  FORWARD PROJECTION SUMMARY")
        print("=================================================================")
        for r in fp_results:
            if r["status"] == "ok":
                pp = r["projected_prices"]
                ud = r["upside_downside"]
                ev_label = "EV POSITIVE" if r["ev_positive"] else "EV NEGATIVE"
                print("  " + r["symbol"] +
                      " | Strike=$" + str(r["strike"]) +
                      " | Expiry=" + r["expiry"])
                print("    Current=$" + str(r["current_price"]) +
                      "  Median Proj=$" + str(pp["median"]) +
                      "  (" + str(ud["median_change_pct"]) + "%)")
                print("    90% Range: $" + str(pp["pct_5"]) +
                      " to $" + str(pp["pct_95"]) +
                      "  |  P(ITM)=" + str(r["p_itm_pct"]) + "%" +
                      "  |  EV=$" + str(r["expected_value_ev"]) +
                      "  |  " + ev_label)
            else:
                print("  " + r["symbol"] + " | ERROR: " + r.get("error", ""))
            print("")
        print("=================================================================")


if __name__ == "__main__":
    main()
