import os
import pandas as pd
import numpy as np
import math
from scipy.optimize import brentq
import matplotlib.pyplot as plt

# ── Black-Scholes ─────────────────────────────────────────────────────────

def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def bs_call(S, K, T, sigma):
    # NOTE: r=0 assumed (no discounting). Fine for competition use.
    if T <= 1e-8 or sigma <= 0:
        return max(0.0, S - K)
    d1 = (math.log(S / K) + 0.5 * sigma**2 * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * norm_cdf(d1) - K * norm_cdf(d2)

def implied_vol(market_price, S, K, T, tol=1e-6):
    if T <= 1e-8:
        return None
    intrinsic = max(0.0, S - K)
    if market_price <= intrinsic + 0.01:
        return None
    try:
        iv = brentq(
            lambda sigma: bs_call(S, K, T, sigma) - market_price,
            1e-4, 10.0, xtol=tol, maxiter=100  # FIX: was already 10.0, kept consistent
        )
        return iv
    except:
        return None

# ── Params ───────────────────────────────────────────────────────────────

STRIKES = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]

# FIX: Separate TOTAL_LIFE per round. Verify these match your actual expiry schedule.
# If both rounds genuinely share the same 7-day life, set both to 7.
TOTAL_LIFE = 8

# FIX: Maximum allowed bid-ask spread as a fraction of mid price.
# Options with wider spreads are considered stale/unreliable and are skipped.
MAX_SPREAD_FRAC = 0.15

def collect_points_from_file(path, current_day, round_name):
    m_vals = []
    iv_vals = []

    print(f"Loading {path}...")
    df = pd.read_csv(path, sep=';')


    for ts, group in df.groupby('timestamp'):
        tte_days = (TOTAL_LIFE - current_day) - ts / 1_000_000.0

        vf_row = group[group['product'] == 'VELVETFRUIT_EXTRACT']
        if vf_row.empty:
            continue

        vf = vf_row.iloc[0]
        if pd.isna(vf['bid_price_1']) or pd.isna(vf['ask_price_1']):
            continue

        bb, ba = vf['bid_price_1'], vf['ask_price_1']
        bsz, asz = vf['bid_volume_1'], abs(vf['ask_volume_1'])
        if bsz + asz == 0:
            continue

        S = (bb * asz + ba * bsz) / (bsz + asz)

        # FIX: Use round-specific TOTAL_LIFE for TTE calculation
        tte_days = (TOTAL_LIFE - current_day) - ts / 1_000_000.0
        T = max(1e-8, tte_days / 365.0)

        for K in STRIKES:
            opt_row = group[group['product'] == f'VEV_{K}']
            if opt_row.empty:
                continue

            opt = opt_row.iloc[0]
            if pd.isna(opt['bid_price_1']) or pd.isna(opt['ask_price_1']):
                continue

            bid_o, ask_o = opt['bid_price_1'], opt['ask_price_1']
            mid = (bid_o + ask_o) / 2.0

            # FIX: Skip stale/wide-spread quotes before computing IV
            if mid > 0 and (ask_o - bid_o) / mid > MAX_SPREAD_FRAC:
                continue

            actual_iv = implied_vol(mid, S, K, T)

            # FIX: Apply consistent IV bounds (same as Script 2 now uses)
            if actual_iv is not None and 0.001 < actual_iv < 2.0:
                m = math.log(K / S) / math.sqrt(T)

                if abs(m) < 3.0:
                    m_vals.append(m)
                    iv_vals.append(actual_iv)

    return m_vals, iv_vals

def main():
    files_to_load = [
        ("data/ROUND3/prices_round_3_day_0.csv", 0, 'ROUND3'),
        ("data/ROUND3/prices_round_3_day_1.csv", 1, 'ROUND3'),
        ("data/ROUND3/prices_round_3_day_2.csv", 2, 'ROUND3'),
        ("data/ROUND4/prices_round_4_day_1.csv", 1, 'ROUND4'),
        ("data/ROUND4/prices_round_4_day_2.csv", 2, 'ROUND4'),
        ("data/ROUND4/prices_round_4_day_3.csv", 3, 'ROUND4'),
    ]

    all_m = []
    all_iv = []

    for path, current_day, round_name in files_to_load:
        if not os.path.exists(path):
            print(f"Skipping {path}, file not found.")
            continue

        m_vals, iv_vals = collect_points_from_file(path, current_day, round_name)
        all_m.extend(m_vals)
        all_iv.extend(iv_vals)
        print(f"  -> Collected {len(m_vals)} valid points from this file.")

    if not all_m:
        print("No data points collected.")
        return

    print(f"\nTotal points collected: {len(all_m)}")
    print("Fitting quadratic polynomial (iv = a*m^2 + b*m + c)...")

    all_m_arr = np.array(all_m)
    all_iv_arr = np.array(all_iv)

    # FIX: Weight ATM points more heavily. OTM quotes are noisy and should
    # influence the fit less. Gaussian weights centred at m=0 (ATM).
    weights = np.exp(-0.5 * all_m_arr ** 2)

    coeffs = np.polyfit(all_m_arr, all_iv_arr, deg=2, w=weights)
    a, b, c = coeffs

    m_range = np.linspace(-3, 3, 300)
    fitted_curve = np.polyval(coeffs, m_range)

    plt.figure(figsize=(10, 6))
    plt.scatter(all_m_arr, all_iv_arr, alpha=0.08, s=5, label='Observed IVs')
    plt.plot(m_range, fitted_curve, color='red', linewidth=2, label='Fitted smile')

    # Mark where each strike lands approximately
    if all_m_arr.size > 0:
        plt.xlabel('Moneyness  m = log(K/S) / sqrt(T)')
        plt.ylabel('Implied Vol')
        plt.title('IV Smile: Observed vs Fitted')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('smile_scatter.png', dpi=150, bbox_inches='tight')
        print("Saved smile_scatter.png")

    # Diagnostics
    fitted = np.polyval(coeffs, all_m_arr)
    residuals = all_iv_arr - fitted
    rmse = np.sqrt(np.mean(residuals ** 2))

    print("\n" + "=" * 50)
    print("SMILE CALIBRATION RESULTS")
    print("=" * 50)
    print(f"a = {a:.6f}")
    print(f"b = {b:.6f}")
    print(f"c = {c:.6f}")
    print(f"Fit RMSE = {rmse:.6f} (in IV units)")
    print("=" * 50)
    print("\nCopy and paste this into your algorithm:")
    print(f"SMILE_COEFFS = [{a:.6f}, {b:.6f}, {c:.6f}]")

if __name__ == '__main__':
    main()