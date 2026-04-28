import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.optimize import brentq
import os

# ── Black-Scholes helpers ─────────────────────────────────────────────────────

def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def bs_call(S, K, T, sigma):
    if T <= 1e-8 or sigma <= 0:
        return max(0.0, S - K)
    d1 = (math.log(S / K) + 0.5 * sigma**2 * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * norm_cdf(d1) - K * norm_cdf(d2)

def implied_vol(market_price, S, K, T, tol=1e-6):
    if T <= 1e-6:
        return None
    intrinsic = max(0.0, S - K)
    if market_price <= intrinsic + 0.05: # Buffer for noisy mid
        return None
    try:
        iv = brentq(
            lambda sigma: bs_call(S, K, T, sigma) - market_price,
            1e-4, 4.0, xtol=tol
        )
        return iv
    except:
        return None

# ── CSV Loading Logic ─────────────────────────────────────────────────────────

def load_round_data(path):
    """Loads a single prices CSV and groups by timestamp."""
    print(f"Loading {path}...")
    df = pd.read_csv(path, sep=';')
    
    # We need bid_price_1, ask_price_1, etc.
    final_data = {}
    
    for ts, group in df.groupby('timestamp'):
        products = {}
        for _, row in group.iterrows():
            prod = row['product']
            products[prod] = {
                'bid': row['bid_price_1'],
                'bid_vol': row['bid_volume_1'],
                'ask': row['ask_price_1'],
                'ask_vol': row['ask_volume_1'],
                'mid': row['mid_price']
            }
        final_data[ts] = products
        
    return final_data

# ── Parameters — match your algo exactly ─────────────────────────────────────

STRIKES     = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
TOTAL_LIFE  = 8

def get_T(timestamp, day):
    # Prosperity days are 0, 1, 2. We are calibrating Round 3.
    tte_days = (TOTAL_LIFE - day) - timestamp / 1_000_000
    return max(1e-8, tte_days / TOTAL_LIFE)

def get_S(products):
    vf = products.get("VELVETFRUIT_EXTRACT")
    if not vf or pd.isna(vf['bid']) or pd.isna(vf['ask']):
        return None
    bb, ba = vf['bid'], vf['ask']
    bsz, asz = vf['bid_vol'], abs(vf['ask_vol'])
    # Weighted mid for better precision
    return (bb * asz + ba * bsz) / (bsz + asz)

# ── Collect (m, iv) points ───────────────────────────────────────────────────

def collect_smile_points(days_data):
    m_vals  = []
    iv_vals = []
    skipped = 0

    for day, ticks in days_data.items():
        print(f"Processing Day {day}...")
        for ts, products in ticks.items():
            T = get_T(ts, day)
            S = get_S(products)

            if S is None or T < 1e-6:
                continue

            for K in STRIKES:
                key = f"VEV_{K}"
                opt = products.get(key)
                if not opt or pd.isna(opt['bid']) or pd.isna(opt['ask']):
                    continue

                mid = (opt['bid'] + opt['ask']) / 2.0
                iv = implied_vol(mid, S, K, T)

                if iv is None or iv < 0.001 or iv > 1.5:
                    skipped += 1
                    continue

                m = math.log(K / S) / math.sqrt(T)

                # Filter extreme moneyness
                if abs(m) > 2.5:
                    skipped += 1
                    continue

                m_vals.append(m)
                iv_vals.append(iv)

    print(f"Collected {len(m_vals)} points, skipped {skipped}")
    return np.array(m_vals), np.array(iv_vals)

# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    days_to_load = [0, 1, 2]
    data_by_day = {}
    
    # Base path for data
    base_data_path = "data/ROUND3/"
    
    for d in days_to_load:
        p = os.path.join(base_data_path, f"prices_round_3_day_{d}.csv")
        if os.path.exists(p):
            data_by_day[d] = load_round_data(p)
        else:
            print(f"Warning: Could not find {p}")

    if not data_by_day:
        print(f"No data loaded from {base_data_path}. Check paths.")
        exit()

    # Per-day check
    print("\n── Per-day coefficients ──")
    all_coeffs = []
    for day, ticks in data_by_day.items():
        m, iv = collect_smile_points({day: ticks})
        if len(m) > 100:
            coeffs = np.polyfit(m, iv, deg=2)
            all_coeffs.append(coeffs)
            print(f"  Day {day}: a={coeffs[0]:.6f} b={coeffs[1]:.6f} c={coeffs[2]:.6f}")
        else:
            print(f"  Day {day}: Not enough points")

    # Final Fit
    m_all, iv_all = collect_smile_points(data_by_day)
    if len(m_all) > 0:
        final_coeffs = np.polyfit(m_all, iv_all, deg=2)
        
        print("\n── FINAL FIT (ALL DAYS) ──")
        print(f"Coefficients [a, b, c]: {list(np.round(final_coeffs, 6))}")
        
        # Plot
        m_grid = np.linspace(m_all.min(), m_all.max(), 200)
        iv_fit = np.polyval(final_coeffs, m_grid)
        
        plt.figure(figsize=(10, 5))
        plt.scatter(m_all, iv_all, alpha=0.1, s=5, label="Historical IVs")
        plt.plot(m_grid, iv_fit, color="red", lw=2, label="Polynomial Fit")
        plt.xlabel("Standardized Moneyness m = ln(K/S)/sqrt(T)")
        plt.ylabel("Implied Volatility")
        plt.title("VEV Volatility Smile Calibration (Round 3)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig("volatility_smile_calibration.png", dpi=150)
        print("Plot saved to volatility_smile_calibration.png")
        plt.show()
    else:
        print("No points collected for final fit.")
