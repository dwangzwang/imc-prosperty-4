"""
Round 3 Data Analysis Script
============================
Analyzes HYDROGEL_PACK, VELVETFRUIT_EXTRACT, and VEV options for:
  1. Price trends and mean-reversion characteristics
  2. Cross-product correlations
  3. Implied volatility surface / smile
  4. Option mispricing opportunities
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import norm, pearsonr
from scipy.optimize import brentq
import warnings
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════

DATA_DIR = "data/ROUND3"

def load_prices():
    frames = []
    for day in range(3):
        df = pd.read_csv(f"{DATA_DIR}/prices_round_3_day_{day}.csv", sep=";")
        frames.append(df)
    return pd.concat(frames, ignore_index=True)

def load_trades():
    frames = []
    for day in range(3):
        df = pd.read_csv(f"{DATA_DIR}/trades_round_3_day_{day}.csv", sep=";")
        df["day"] = day
        frames.append(df)
    return pd.concat(frames, ignore_index=True)

prices = load_prices()
trades = load_trades()

# extract mid prices for each product
def get_mid_series(product):
    sub = prices[prices["product"] == product][["day", "timestamp", "mid_price"]].copy()
    sub = sub.sort_values(["day", "timestamp"]).reset_index(drop=True)
    # create a global timestamp for multi-day continuity
    sub["global_ts"] = sub["day"] * 1_000_000 + sub["timestamp"]
    return sub

hydro = get_mid_series("HYDROGEL_PACK")
velvet = get_mid_series("VELVETFRUIT_EXTRACT")

print("=" * 70)
print("ROUND 3 DATA ANALYSIS")
print("=" * 70)

# ══════════════════════════════════════════════════════════════════════════════
# 2. BASIC STATISTICS
# ══════════════════════════════════════════════════════════════════════════════

print("\n>>> BASIC STATISTICS <<<")
for name, series in [("HYDROGEL_PACK", hydro), ("VELVETFRUIT_EXTRACT", velvet)]:
    mid = series["mid_price"]
    print(f"\n  {name}:")
    print(f"    Mean:   {mid.mean():.2f}")
    print(f"    Std:    {mid.std():.2f}")
    print(f"    Min:    {mid.min():.2f}")
    print(f"    Max:    {mid.max():.2f}")
    print(f"    Range:  {mid.max() - mid.min():.2f}")
    
    # per-day summary
    for day in range(3):
        day_data = series[series["day"] == day]["mid_price"]
        print(f"    Day {day}: start={day_data.iloc[0]:.1f}  end={day_data.iloc[-1]:.1f}  "
              f"Δ={day_data.iloc[-1] - day_data.iloc[0]:+.1f}")

# ══════════════════════════════════════════════════════════════════════════════
# 3. RETURNS AND CORRELATION ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

print("\n>>> RETURNS & CORRELATION <<<")

# merge on (day, timestamp) to get aligned pairs
merged = hydro.merge(velvet, on=["day", "timestamp"], suffixes=("_hydro", "_velvet"))

# tick-level returns
merged["ret_hydro"]  = merged["mid_price_hydro"].diff()
merged["ret_velvet"] = merged["mid_price_velvet"].diff()

# drop NaN from diff
clean = merged.dropna(subset=["ret_hydro", "ret_velvet"])

corr_raw, pval_raw = pearsonr(clean["ret_hydro"], clean["ret_velvet"])
print(f"  Tick-level return correlation: {corr_raw:.4f}  (p={pval_raw:.2e})")

# rolling 100-tick return correlation
window = 100
rolling_corr = clean["ret_hydro"].rolling(window).corr(clean["ret_velvet"])
print(f"  Rolling {window}-tick correlation: mean={rolling_corr.mean():.4f}, "
      f"std={rolling_corr.std():.4f}")

# lagged correlation (does one lead the other?)
print("\n  Lagged cross-correlation (Hydro leads Velvet):")
for lag in [-5, -3, -1, 0, 1, 3, 5]:
    if lag >= 0:
        c = clean["ret_hydro"].iloc[lag:].reset_index(drop=True).corr(
            clean["ret_velvet"].iloc[:len(clean)-lag].reset_index(drop=True))
    else:
        c = clean["ret_velvet"].iloc[-lag:].reset_index(drop=True).corr(
            clean["ret_hydro"].iloc[:len(clean)+lag].reset_index(drop=True))
    print(f"    Lag {lag:+d}: {c:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# 4. MEAN REVERSION TESTS
# ══════════════════════════════════════════════════════════════════════════════

print("\n>>> MEAN REVERSION ANALYSIS <<<")

for name, series in [("HYDROGEL_PACK", hydro), ("VELVETFRUIT_EXTRACT", velvet)]:
    mid = series["mid_price"].values
    
    # augmented dickey-fuller style test (simplified)
    # test if Δp_t = α + β * p_{t-1} + ε (β < 0 means mean-reverting)
    dp = np.diff(mid)
    p_lag = mid[:-1]
    
    # remove any NaN
    mask = ~(np.isnan(dp) | np.isnan(p_lag))
    dp, p_lag = dp[mask], p_lag[mask]
    
    # simple OLS: β coefficient
    p_lag_dm = p_lag - p_lag.mean()
    beta = np.sum(dp * p_lag_dm) / np.sum(p_lag_dm ** 2)
    
    # half-life of mean reversion (if β < 0)
    if beta < 0:
        half_life = -np.log(2) / beta
    else:
        half_life = float('inf')
    
    # autocorrelation of returns
    returns = np.diff(mid)
    ac1 = np.corrcoef(returns[:-1], returns[1:])[0, 1]
    ac5 = np.corrcoef(returns[:-5], returns[5:])[0, 1]
    
    print(f"\n  {name}:")
    print(f"    Regression β:     {beta:.6f} ({'mean-reverting' if beta < 0 else 'trending'})")
    print(f"    Half-life:        {half_life:.1f} ticks")
    print(f"    Autocorr(1):      {ac1:.4f} ({'negative=reverting' if ac1 < 0 else 'positive=trending'})")
    print(f"    Autocorr(5):      {ac5:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# 5. VELVETFRUIT LINEAR DRIFT TEST
# ══════════════════════════════════════════════════════════════════════════════

print("\n>>> VELVETFRUIT EXTRACT: LINEAR DRIFT TEST <<<")

for day in range(3):
    day_data = velvet[velvet["day"] == day].copy()
    ts = day_data["timestamp"].values
    mid = day_data["mid_price"].values
    
    # fit linear regression: mid = a + b * timestamp
    A = np.vstack([ts, np.ones(len(ts))]).T
    b_slope, a_intercept = np.linalg.lstsq(A, mid, rcond=None)[0]
    
    residuals = mid - (a_intercept + b_slope * ts)
    
    print(f"  Day {day}: slope = {b_slope*1000:.4f} per 1000 ticks, "
          f"intercept = {a_intercept:.1f}, residual std = {residuals.std():.2f}")

# ══════════════════════════════════════════════════════════════════════════════
# 6. OPTIONS ANALYSIS (Black-Scholes)
# ══════════════════════════════════════════════════════════════════════════════

print("\n>>> VEV OPTIONS ANALYSIS <<<")

# black-scholes for european call
def bs_call_price(S, K, T, sigma, r=0):
    """Black-Scholes European call price."""
    if T <= 0:
        return max(0, S - K)
    if sigma <= 0:
        return max(0, S - K)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def bs_implied_vol(market_price, S, K, T, r=0):
    """Invert Black-Scholes to find implied volatility."""
    if T <= 0:
        return np.nan
    intrinsic = max(0, S - K)
    if market_price <= intrinsic + 0.01:
        return np.nan
    try:
        iv = brentq(lambda sig: bs_call_price(S, K, T, sig, r) - market_price,
                     0.001, 10.0, xtol=1e-6)
        return iv
    except (ValueError, RuntimeError):
        return np.nan

def bs_delta(S, K, T, sigma, r=0):
    """Black-Scholes delta for call."""
    if T <= 0:
        return 1.0 if S > K else 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1)

def bs_vega(S, K, T, sigma, r=0):
    """Black-Scholes vega for call."""
    if T <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return S * np.sqrt(T) * norm.pdf(d1)

# extract VEV strikes
vev_products = [p for p in prices["product"].unique() if p.startswith("VEV_")]
strikes = sorted([int(p.split("_")[1]) for p in vev_products])
print(f"  Available strikes: {strikes}")

# total round length = 3 days * 1,000,000 ticks each
# assuming options expire at end of day 4 (T=0 means just expired)
TOTAL_TICKS = 4 * 1_000_000  # total lifetime
TICKS_PER_YEAR = TOTAL_TICKS  # normalize so T goes from ~1 to 0

# analyze implied vol at various timestamps across days
print("\n  Implied Volatility Surface (sampled snapshots):")
print(f"  {'Day':>3} {'Tick':>8} {'S':>7} {'Strike':>7} {'MktPx':>7} {'IV':>8} {'Delta':>7}")
print("  " + "-" * 60)

iv_records = []

for day in range(3):
    # sample a few timestamps per day
    for ts in [0, 250_000, 500_000, 750_000]:
        # get underlying price at this point
        v_row = prices[(prices["product"] == "VELVETFRUIT_EXTRACT") & 
                       (prices["day"] == day) & 
                       (prices["timestamp"] == ts)]
        if v_row.empty:
            continue
        S = v_row["mid_price"].values[0]
        
        # time to expiry (in "years" of TOTAL_TICKS)
        elapsed = day * 1_000_000 + ts
        T = max(1e-6, (TOTAL_TICKS - elapsed) / TOTAL_TICKS)
        
        for strike in strikes:
            product = f"VEV_{strike}"
            o_row = prices[(prices["product"] == product) & 
                           (prices["day"] == day) & 
                           (prices["timestamp"] == ts)]
            if o_row.empty:
                continue
            mkt_price = o_row["mid_price"].values[0]
            
            iv = bs_implied_vol(mkt_price, S, strike, T)
            if not np.isnan(iv):
                delta = bs_delta(S, strike, T, iv)
                iv_records.append({
                    "day": day, "timestamp": ts, "strike": strike,
                    "S": S, "mkt_price": mkt_price, "iv": iv, "delta": delta, "T": T
                })
                if ts == 0:  # only print day-start for brevity
                    print(f"  {day:>3} {ts:>8} {S:>7.1f} {strike:>7} {mkt_price:>7.1f} "
                          f"{iv:>7.2%} {delta:>7.3f}")

iv_df = pd.DataFrame(iv_records)

if not iv_df.empty:
    print(f"\n  Summary IV statistics:")
    print(f"    Mean IV:     {iv_df['iv'].mean():.2%}")
    print(f"    Median IV:   {iv_df['iv'].median():.2%}")
    print(f"    Std IV:      {iv_df['iv'].std():.2%}")
    print(f"    Min IV:      {iv_df['iv'].min():.2%}")
    print(f"    Max IV:      {iv_df['iv'].max():.2%}")
    
    # volatility smile analysis
    print("\n  Volatility Smile (Day 0, t=0):")
    smile = iv_df[(iv_df["day"] == 0) & (iv_df["timestamp"] == 0)]
    for _, row in smile.iterrows():
        moneyness = row["S"] / row["strike"]
        print(f"    K={row['strike']:>5}  S/K={moneyness:.3f}  IV={row['iv']:.2%}  Δ={row['delta']:.3f}")

# ══════════════════════════════════════════════════════════════════════════════
# 7. PLOTS
# ══════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(3, 2, figsize=(16, 14))
fig.suptitle("Round 3 Analysis Dashboard", fontsize=16, fontweight='bold')

# 7a. Price time series
ax = axes[0, 0]
for day in range(3):
    hd = hydro[hydro["day"] == day]
    ax.plot(hd["timestamp"], hd["mid_price"], alpha=0.7, label=f"Day {day}")
ax.set_title("HYDROGEL_PACK Mid Price")
ax.set_xlabel("Timestamp")
ax.set_ylabel("Mid Price")
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[0, 1]
for day in range(3):
    vd = velvet[velvet["day"] == day]
    ax.plot(vd["timestamp"], vd["mid_price"], alpha=0.7, label=f"Day {day}")
ax.set_title("VELVETFRUIT_EXTRACT Mid Price")
ax.set_xlabel("Timestamp")
ax.set_ylabel("Mid Price")
ax.legend()
ax.grid(True, alpha=0.3)

# 7b. Returns autocorrelation
ax = axes[1, 0]
h_returns = hydro["mid_price"].diff().dropna().values
lags = range(1, 51)
ac_h = [np.corrcoef(h_returns[:-l], h_returns[l:])[0, 1] for l in lags]
ax.bar(lags, ac_h, alpha=0.7, color='steelblue')
ax.axhline(0, color='black', linewidth=0.5)
ax.set_title("HYDROGEL Autocorrelation of Returns")
ax.set_xlabel("Lag (ticks)")
ax.set_ylabel("Autocorrelation")
ax.grid(True, alpha=0.3)

ax = axes[1, 1]
v_returns = velvet["mid_price"].diff().dropna().values
ac_v = [np.corrcoef(v_returns[:-l], v_returns[l:])[0, 1] for l in lags]
ax.bar(lags, ac_v, alpha=0.7, color='darkorange')
ax.axhline(0, color='black', linewidth=0.5)
ax.set_title("VELVETFRUIT Autocorrelation of Returns")
ax.set_xlabel("Lag (ticks)")
ax.set_ylabel("Autocorrelation")
ax.grid(True, alpha=0.3)

# 7c. Cross-correlation scatter
ax = axes[2, 0]
ax.scatter(clean["ret_hydro"].values[::50], clean["ret_velvet"].values[::50],
           alpha=0.1, s=1, color='purple')
ax.set_title(f"Return Scatter (corr={corr_raw:.3f})")
ax.set_xlabel("HYDROGEL return")
ax.set_ylabel("VELVETFRUIT return")
ax.grid(True, alpha=0.3)

# 7d. Volatility smile
ax = axes[2, 1]
if not iv_df.empty:
    for day in range(3):
        smile_d = iv_df[(iv_df["day"] == day) & (iv_df["timestamp"] == 0)]
        if not smile_d.empty:
            ax.plot(smile_d["strike"], smile_d["iv"] * 100, 'o-', label=f"Day {day}", alpha=0.8)
    ax.set_title("VEV Implied Volatility Smile (t=0)")
    ax.set_xlabel("Strike Price")
    ax.set_ylabel("Implied Volatility (%)")
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("round3_analysis.png", dpi=150, bbox_inches='tight')
plt.show()
print("\n  Plot saved to round3_analysis.png")

# ══════════════════════════════════════════════════════════════════════════════
# 8. STRATEGY RECOMMENDATIONS
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("STRATEGY RECOMMENDATIONS")
print("=" * 70)

# hydrogel assessment
h_ac1 = ac_h[0]
print(f"\n  HYDROGEL_PACK:")
if h_ac1 < -0.05:
    print(f"    ✓ Strong mean reversion detected (AC1={h_ac1:.4f})")
    print(f"    → STRATEGY: Market-making with inventory-skewed MM (like OsmiumStrategy)")
elif h_ac1 > 0.05:
    print(f"    ✓ Momentum/trending detected (AC1={h_ac1:.4f})")
    print(f"    → STRATEGY: Trend-following with momentum signals")
else:
    print(f"    ~ Weak signal (AC1={h_ac1:.4f})")
    print(f"    → STRATEGY: Pure market-making with tight spreads")

# velvetfruit assessment
v_ac1 = ac_v[0]
print(f"\n  VELVETFRUIT_EXTRACT:")
# check linear drift
day0_v = velvet[velvet["day"] == 0]["mid_price"]
drift_per_1k = (day0_v.iloc[-1] - day0_v.iloc[0]) / (len(day0_v) * 100 / 1000)
print(f"    Drift rate: {drift_per_1k:.4f} per 1000 ticks")
if abs(drift_per_1k) > 0.005:
    print(f"    ✓ Significant linear drift detected")
    print(f"    → STRATEGY: Buy-and-hold (like IntarianPepperRoot) if upward")
else:
    print(f"    ~ No strong drift")
if v_ac1 < -0.05:
    print(f"    ✓ Mean reversion detected (AC1={v_ac1:.4f})")
    print(f"    → STRATEGY: MM with mean-reversion + drift overlay")

# options assessment
print(f"\n  VEV OPTIONS:")
if not iv_df.empty:
    print(f"    Mean IV: {iv_df['iv'].mean():.2%}")
    iv_std = iv_df.groupby("strike")["iv"].std()
    print(f"    IV stability per strike (std): {iv_std.mean():.4f}")
    print(f"    → STRATEGY: Black-Scholes fair value calculation")
    print(f"    → Buy underpriced options (market price < BS price)")
    print(f"    → Sell overpriced options (market price > BS price)")
    print(f"    → Optional: Delta-hedge with VELVETFRUIT_EXTRACT positions")

# correlation assessment
print(f"\n  CROSS-PRODUCT:")
print(f"    Hydrogel ↔ Velvetfruit correlation: {corr_raw:.4f}")
if abs(corr_raw) > 0.1:
    print(f"    ✓ Significant correlation — potential pairs trading opportunity")
    print(f"    → STRATEGY: Use one as a lead indicator for the other")
else:
    print(f"    ~ Weak correlation — trade independently")

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
