import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import brentq

# ── Black-Scholes ──────────────────────────────────────────────────────────────

def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def bs_call(S, K, T, sigma):
    # r=0 assumed throughout (no discounting).
    if T <= 1e-8 or sigma <= 0:
        return max(0.0, S - K)
    d1 = (math.log(S / K) + 0.5 * sigma**2 * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * norm_cdf(d1) - K * norm_cdf(d2)

def bs_delta(S, K, T, sigma):
    if T <= 1e-8 or sigma <= 0:
        return 1.0 if S > K else 0.0
    d1 = (math.log(S / K) + 0.5 * sigma**2 * T) / (sigma * math.sqrt(T))
    return norm_cdf(d1)

def implied_vol(market_price, S, K, T, tol=1e-6):
    if T <= 1e-8:
        return None
    intrinsic = max(0.0, S - K)
    if market_price <= intrinsic + 0.01:
        return None
    try:
        return brentq(
            lambda sigma: bs_call(S, K, T, sigma) - market_price,
            1e-4, 10.0, xtol=tol, maxiter=100
        )
    except Exception:
        return None

# ── Configuration ──────────────────────────────────────────────────────────────

STRIKES      = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
TOTAL_LIFE   = 8        # TTE at start of day N = (8 - N) days
MAX_SPREAD_FRAC = 0.15  # skip quotes whose spread > 15% of mid

# All files to process: (path, day_number, round_label)
FILES = [
    ("data/ROUND3/prices_round_3_day_0.csv", 0, "ROUND3"),
    ("data/ROUND3/prices_round_3_day_1.csv", 1, "ROUND3"),
    ("data/ROUND3/prices_round_3_day_2.csv", 2, "ROUND3"),
    ("data/ROUND4/prices_round_4_day_1.csv", 1, "ROUND4"),
    ("data/ROUND4/prices_round_4_day_2.csv", 2, "ROUND4"),
    ("data/ROUND4/prices_round_4_day_3.csv", 3, "ROUND4"),
]

UNDERLYING = 'VELVETFRUIT_EXTRACT'   # change if your CSV uses a different name

# ── Helpers ────────────────────────────────────────────────────────────────────

def weighted_mid(row):
    """Size-weighted mid price from best bid/ask."""
    bb, ba = row['bid_price_1'], row['ask_price_1']
    bsz = row['bid_volume_1']
    asz = abs(row['ask_volume_1'])
    if bsz + asz == 0:
        return (bb + ba) / 2.0
    return (bb * asz + ba * bsz) / (bsz + asz)

def tte(current_day, ts):
    """Time to expiry in years."""
    tte_days = (TOTAL_LIFE - current_day) - ts / 1_000_000.0
    return max(1e-8, tte_days / 365.0)

# ── Per-Strike IV table: lookup + linear interpolation ────────────────────────

def make_smile_iv_fn(strike_iv_table):
    """
    Returns a smile_iv(K, S, T) function that looks up IV from the table.
    For K values not in the table, linearly interpolates between neighbours.
    S and T are accepted for API compatibility but not used — each strike has
    one fixed IV derived from historical data.
    """
    strikes_sorted = sorted(strike_iv_table.keys())

    def smile_iv(K, S=None, T=None):
        if K in strike_iv_table:
            return max(0.005, strike_iv_table[K])
        lower = [k for k in strikes_sorted if k < K]
        upper = [k for k in strikes_sorted if k > K]
        if not lower or not upper:
            return None   # outside calibrated range — don't extrapolate
        k_lo, k_hi = lower[-1], upper[0]
        t = (K - k_lo) / (k_hi - k_lo)
        iv = strike_iv_table[k_lo] * (1 - t) + strike_iv_table[k_hi] * t
        return max(0.005, iv)

    return smile_iv

# ── Pass 1: Calibration — collect IV observations per strike ───────────────────

def calibrate(files):
    """
    Iterate over all files and collect implied vol observations per strike.
    Returns a dict {strike: median_iv} and prints a summary table.
    """
    print("=" * 60)
    print("PASS 1 — CALIBRATING PER-STRIKE IV TABLE")
    print("=" * 60)

    iv_by_strike = {K: [] for K in STRIKES}

    for path, current_day, round_label in files:
        if not os.path.exists(path):
            print(f"  Skipping {path} (not found)")
            continue

        print(f"  Calibrating from {path} (day {current_day})...")
        df = pd.read_csv(path, sep=';')

        for ts, group in df.groupby('timestamp'):
            # Underlying
            vf_row = group[group['product'] == UNDERLYING]
            if vf_row.empty:
                continue
            vf = vf_row.iloc[0]
            if pd.isna(vf['bid_price_1']) or pd.isna(vf['ask_price_1']):
                continue
            bsz = vf['bid_volume_1']
            asz = abs(vf['ask_volume_1'])
            if bsz + asz == 0:
                continue
            S = weighted_mid(vf)
            T = tte(current_day, ts)

            for K in STRIKES:
                opt_row = group[group['product'] == f'VEV_{K}']
                if opt_row.empty:
                    continue
                opt = opt_row.iloc[0]
                if pd.isna(opt['bid_price_1']) or pd.isna(opt['ask_price_1']):
                    continue

                bid_o, ask_o = opt['bid_price_1'], opt['ask_price_1']
                mid = (bid_o + ask_o) / 2.0

                # Skip stale/wide quotes
                if mid > 0 and (ask_o - bid_o) / mid > MAX_SPREAD_FRAC:
                    continue

                iv = implied_vol(mid, S, K, T)
                if iv is not None and 0.001 < iv < 2.0:
                    iv_by_strike[K].append(iv)

    # Build table from medians
    print()
    print(f"  {'Strike':>8}  {'N pts':>6}  {'Median IV':>10}  {'Std IV':>8}  {'Min IV':>8}  {'Max IV':>8}")
    print(f"  {'-'*8}  {'-'*6}  {'-'*10}  {'-'*8}  {'-'*8}  {'-'*8}")

    strike_iv_table = {}
    for K in STRIKES:
        ivs = iv_by_strike[K]
        if len(ivs) >= 10:
            med = float(np.median(ivs))
            strike_iv_table[K] = med
            print(f"  {K:>8}  {len(ivs):>6}  {med:>10.4f}  "
                  f"{np.std(ivs):>8.4f}  {np.min(ivs):>8.4f}  {np.max(ivs):>8.4f}")
        else:
            print(f"  {K:>8}  {len(ivs):>6}  {'SKIPPED — insufficient data':>10}")

    print()
    print("  Copy into your trading algorithm:")
    print(f"  STRIKE_IV_TABLE = {strike_iv_table}")
    print()

    return strike_iv_table

# ── Pass 2: Analysis — compute fair values and deviations ─────────────────────

def process_file(path, current_day, round_label, smile_iv):
    """
    Process one CSV file and return a list of per-observation dicts.
    """
    df = pd.read_csv(path, sep=';')
    results = []

    for ts, group in df.groupby('timestamp'):
        vf_row = group[group['product'] == UNDERLYING]
        if vf_row.empty:
            continue
        vf = vf_row.iloc[0]
        if pd.isna(vf['bid_price_1']) or pd.isna(vf['ask_price_1']):
            continue
        bsz = vf['bid_volume_1']
        asz = abs(vf['ask_volume_1'])
        if bsz + asz == 0:
            continue

        S = weighted_mid(vf)
        T = tte(current_day, ts)

        for K in STRIKES:
            opt_row = group[group['product'] == f'VEV_{K}']
            if opt_row.empty:
                continue
            opt = opt_row.iloc[0]
            if pd.isna(opt['bid_price_1']) or pd.isna(opt['ask_price_1']):
                continue

            bid_o, ask_o = opt['bid_price_1'], opt['ask_price_1']
            mid = (bid_o + ask_o) / 2.0

            if mid > 0 and (ask_o - bid_o) / mid > MAX_SPREAD_FRAC:
                continue

            actual_iv = implied_vol(mid, S, K, T)
            fitted_iv = smile_iv(K, S, T)

            if actual_iv is None or fitted_iv is None:
                continue
            if not (0.001 < actual_iv < 2.0):
                continue

            fair_price  = bs_call(S, K, T, fitted_iv)
            delta       = bs_delta(S, K, T, fitted_iv)

            results.append({
                'dataset':      round_label,
                'day':          current_day,
                'timestamp':    ts,
                'strike':       K,
                'S':            S,
                'T':            T,
                'actual_iv':    actual_iv,
                'fitted_iv':    fitted_iv,
                'iv_diff':      actual_iv - fitted_iv,
                'market_price': mid,
                'fair_price':   fair_price,
                'price_dev':    mid - fair_price,
                'bid':          bid_o,
                'ask':          ask_o,
                'delta':        delta,
            })

    return results

# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_all(df_res):
    colors = plt.cm.tab10(np.linspace(0, 1, len(STRIKES)))
    strike_color = {K: colors[i] for i, K in enumerate(STRIKES)}

    for dataset in ['ROUND3', 'ROUND4']:
        ds = df_res[df_res['dataset'] == dataset]
        if ds.empty:
            continue

        for day in sorted(ds['day'].unique()):
            dd = ds[ds['day'] == day]
            if dd.empty:
                continue

            tag = f"{dataset.lower()}_day{day}"
            print(f"  Plotting {dataset} Day {day}...")

            # ── Plot 1: IV Difference (actual - fitted) ────────────────────
            fig, ax = plt.subplots(figsize=(15, 6))
            for K in STRIKES:
                sd = dd[dd['strike'] == K]
                if not sd.empty:
                    ax.plot(sd['timestamp'], sd['iv_diff'],
                            label=f'K={K}', color=strike_color[K],
                            alpha=0.8, linewidth=1)
            ax.axhline(0, color='black', linestyle='--', linewidth=1)
            ax.set_title(f"IV Difference (Actual − Fitted) — {dataset} Day {day}\n"
                         f"(per-strike table fit; zero = perfectly fitted)")
            ax.set_xlabel("Timestamp")
            ax.set_ylabel("IV Difference")
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            fig.savefig(f"iv_diff_{tag}.png", dpi=150, bbox_inches='tight')
            plt.close(fig)

            # ── Plot 2: Market Price vs Fair Price ─────────────────────────
            fig, ax = plt.subplots(figsize=(15, 7))
            for K in STRIKES:
                sd = dd[dd['strike'] == K]
                if not sd.empty:
                    ax.plot(sd['timestamp'], sd['market_price'],
                            label=f'Market K={K}', color=strike_color[K],
                            alpha=0.9, linewidth=1.5)
                    ax.plot(sd['timestamp'], sd['fair_price'],
                            color=strike_color[K], alpha=0.6,
                            linewidth=1.5, linestyle='--')
            # Dummy line for legend entry
            ax.plot([], [], color='grey', linewidth=1.5, label='Solid = Market')
            ax.plot([], [], color='grey', linewidth=1.5, linestyle='--', label='Dashed = Fair')
            ax.set_title(f"Market Price (solid) vs Fair Price (dashed) — {dataset} Day {day}")
            ax.set_xlabel("Timestamp")
            ax.set_ylabel("Price")
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            fig.savefig(f"price_comparison_{tag}.png", dpi=150, bbox_inches='tight')
            plt.close(fig)

            # ── Plot 3: Price Deviation (market − fair), normalised to 0 ───
            active = [K for K in STRIKES if not dd[dd['strike'] == K].empty]
            n = len(active)
            if n == 0:
                continue

            fig, axes = plt.subplots(n, 1, figsize=(15, 3 * n), sharex=True)
            if n == 1:
                axes = [axes]

            fig.suptitle(
                f"Market Price Deviation from Fair Value — {dataset} Day {day}\n"
                f"(positive = market expensive / sell signal, "
                f"negative = market cheap / buy signal)",
                fontsize=12
            )

            for ax, K in zip(axes, active):
                sd = dd[dd['strike'] == K].copy()
                dev = sd['price_dev']

                ax.fill_between(sd['timestamp'], dev, 0,
                                where=(dev >= 0), alpha=0.25, color='red',
                                label='Above fair (sell)')
                ax.fill_between(sd['timestamp'], dev, 0,
                                where=(dev < 0), alpha=0.25, color='green',
                                label='Below fair (buy)')
                ax.plot(sd['timestamp'], dev,
                        color=strike_color[K], linewidth=1, alpha=0.9)
                ax.axhline(0, color='black', linestyle='--', linewidth=0.8)

                # Annotate with mean deviation so you can see if bias remains
                mean_dev = dev.mean()
                ax.axhline(mean_dev, color='orange', linestyle=':', linewidth=1,
                           label=f'Mean={mean_dev:+.2f}')

                yabs = max(dev.abs().max(), 0.01)
                ax.set_ylim(-yabs * 1.4, yabs * 1.4)
                ax.set_ylabel(f'K={K}', fontsize=9)
                ax.grid(True, alpha=0.3)
                ax.legend(loc='upper right', fontsize=7)

            axes[-1].set_xlabel("Timestamp (0 to 1,000,000)")
            plt.tight_layout()
            fig.savefig(f"price_deviation_{tag}.png", dpi=150, bbox_inches='tight')
            plt.close(fig)

            # ── Plot 4: Per-Strike IV Scatter (calibrated table overlaid) ──
            # Shows where the table value sits relative to the intraday IV cloud.
            # Useful for spotting intraday IV drift.
            fig, ax = plt.subplots(figsize=(12, 6))
            for K in STRIKES:
                sd = dd[dd['strike'] == K]
                if sd.empty:
                    continue
                ax.scatter(sd['timestamp'], sd['actual_iv'],
                           color=strike_color[K], alpha=0.2, s=4,
                           label=f'K={K}')
                ax.axhline(sd['fitted_iv'].iloc[0],
                           color=strike_color[K], linewidth=1.2,
                           linestyle='--', alpha=0.8)
            ax.set_title(f"Actual IV over Time (dots) vs Table IV (dashed line) "
                         f"— {dataset} Day {day}")
            ax.set_xlabel("Timestamp")
            ax.set_ylabel("Implied Vol")
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            fig.savefig(f"iv_scatter_{tag}.png", dpi=150, bbox_inches='tight')
            plt.close(fig)

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    # ── Pass 1: build per-strike IV table from all available data ──────────
    strike_iv_table = calibrate(FILES)

    if not strike_iv_table:
        print("No strikes calibrated — check file paths and UNDERLYING name.")
        return

    smile_iv = make_smile_iv_fn(strike_iv_table)

    # ── Pass 2: compute fair values and collect results ─────────────────────
    print("=" * 60)
    print("PASS 2 — COMPUTING FAIR VALUES AND DEVIATIONS")
    print("=" * 60)

    all_data = []
    for path, current_day, round_label in FILES:
        if not os.path.exists(path):
            print(f"  Skipping {path} (not found)")
            continue
        print(f"  Processing {path} (day {current_day})...")
        rows = process_file(path, current_day, round_label, smile_iv)
        all_data.extend(rows)
        print(f"    -> {len(rows)} valid observations")

    if not all_data:
        print("No data processed.")
        return

    df_res = pd.DataFrame(all_data).sort_values(['dataset', 'day', 'timestamp'])

    # ── Summary stats per strike across all data ───────────────────────────
    print()
    print("=" * 60)
    print("DEVIATION SUMMARY (mean price_dev per strike, all days)")
    print("If these are close to zero your calibration is good.")
    print("=" * 60)
    summary = (df_res.groupby('strike')['price_dev']
               .agg(['mean', 'std', 'count'])
               .rename(columns={'mean': 'Mean Dev', 'std': 'Std Dev', 'count': 'N'}))
    print(summary.to_string())
    print()

    # ── Pass 3: generate plots ──────────────────────────────────────────────
    print("=" * 60)
    print("PASS 3 — GENERATING PLOTS")
    print("=" * 60)
    plot_all(df_res)

    print()
    print("Done. Output files:")
    print("  iv_diff_*           — IV difference (actual minus table IV) over time")
    print("  price_comparison_*  — market price vs BS fair price over time")
    print("  price_deviation_*   — price deviation normalised to 0 (main signal plot)")
    print("  iv_scatter_*        — actual IV dots vs calibrated table value per strike")

if __name__ == '__main__':
    main()