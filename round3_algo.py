from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List, Dict
import math

# ═══════════════════════════════════════════════════════════════
# BASE CLASS
# ═══════════════════════════════════════════════════════════════

class Strategy:
    def __init__(self, product: str, position_limit: int):
        self.product = product
        self.position_limit = position_limit

    def get_orders(self, state: TradingState) -> List[Order]:
        raise NotImplementedError

    def order(self, price: int, qty: int) -> Order:
        return Order(self.product, price, qty)


# ═══════════════════════════════════════════════════════════════
# MARKET MAKER STRATEGY (used for Hydrogel and Velvetfruit)
# ═══════════════════════════════════════════════════════════════
#
# How it works:
#   1. "Wall mid" estimates fair value by finding the first bid/ask
#      levels with real volume (>=WALL_VOL), filtering out thin spoofs.
#   2. If true_value is set, the strategy leans toward it when the
#      market drifts more than arb_thresh away (mean-reversion).
#   3. Inventory skew shifts our fair value against our position
#      to prevent getting stuck at position limits.
#   4. Taker: sweep any mispriced resting orders.
#   5. Maker: rest passive limit orders to earn spread.

class MarketMakerStrategy(Strategy):
    WALL_VOL     = 15       # ignore levels thinner than this
    SKEW_DIVISOR = 20       # how fast inventory skew kicks in
    MIN_EDGE     = 1        # minimum profit demanded per trade

    def __init__(self, product: str, position_limit: int,
                 true_value: float = None, arb_thresh: float = 3):
        super().__init__(product, position_limit)
        self.true_value = true_value
        self.arb_thresh = arb_thresh

    def _get_wall_mid(self, depth: OrderDepth) -> float:
        valid_bid = max(depth.buy_orders.keys())
        for bid in sorted(depth.buy_orders.keys(), reverse=True):
            if depth.buy_orders[bid] >= self.WALL_VOL:
                valid_bid = bid
                break

        valid_ask = min(depth.sell_orders.keys())
        for ask in sorted(depth.sell_orders.keys()):
            if abs(depth.sell_orders[ask]) >= self.WALL_VOL:
                valid_ask = ask
                break

        return (valid_bid + valid_ask) / 2

    def get_orders(self, state: TradingState) -> List[Order]:
        depth = state.order_depths[self.product]
        position = state.position.get(self.product, 0)

        if not depth.buy_orders or not depth.sell_orders:
            return []

        fair = self._get_wall_mid(depth)

        # optional mean-reversion anchor
        if self.true_value is not None:
            diff = fair - self.true_value
            if diff > self.arb_thresh:
                fair -= (diff - self.arb_thresh)
            elif diff < -self.arb_thresh:
                fair -= (diff + self.arb_thresh)

        # inventory skew
        skew = position / self.SKEW_DIVISOR
        buy_fv = math.floor(fair - skew)
        sell_fv = math.ceil(fair - skew)

        buy_cap = self.position_limit - position
        sell_cap = self.position_limit + position
        orders: List[Order] = []

        # taker
        for ask in sorted(depth.sell_orders.keys()):
            if ask <= buy_fv - self.MIN_EDGE and buy_cap > 0:
                qty = min(buy_cap, abs(depth.sell_orders[ask]))
                orders.append(self.order(ask, qty))
                buy_cap -= qty

        for bid in sorted(depth.buy_orders.keys(), reverse=True):
            if bid >= sell_fv + self.MIN_EDGE and sell_cap > 0:
                qty = min(sell_cap, depth.buy_orders[bid])
                orders.append(self.order(bid, -qty))
                sell_cap -= qty

        # maker
        best_bid = max(depth.buy_orders.keys())
        best_ask = min(depth.sell_orders.keys())

        our_bid = min(best_bid + 1, buy_fv - self.MIN_EDGE)
        our_ask = max(best_ask - 1, sell_fv + self.MIN_EDGE)

        if our_bid >= our_ask:
            our_bid = our_ask - 1

        if buy_cap > 0:
            orders.append(self.order(our_bid, buy_cap))
        if sell_cap > 0:
            orders.append(self.order(our_ask, -sell_cap))

        return orders


# ═══════════════════════════════════════════════════════════════
# BLACK-SCHOLES HELPERS (no scipy — uses math.erf)
# ═══════════════════════════════════════════════════════════════
#
# VEV options have a 7-day expiry starting from Round 1.
# Historical data: day 0 (tutorial, TTE=8d), day 1 (R1, TTE=7d),
# day 2 (R2, TTE=6d). Round 3 submission: TTE=5d.
#
# We normalize time so T=1.0 at TTE=8 days (start of historical data)
# and T=0.0 at expiry. This gives σ ≈ 3.55% consistently.
#
# T = TTE_in_days / 8
# For Round 3: T = (5 - timestamp/1_000_000) / 8

def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)

def bs_call(S: float, K: float, T: float, sigma: float) -> float:
    """Black-Scholes European call price (r=0)."""
    if T <= 1e-8 or sigma <= 0:
        return max(0.0, S - K)
    d1 = (math.log(S / K) + 0.5 * sigma**2 * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * norm_cdf(d1) - K * norm_cdf(d2)

def bs_delta(S: float, K: float, T: float, sigma: float) -> float:
    """Black-Scholes call delta (r=0)."""
    if T <= 1e-8:
        return 1.0 if S > K else 0.0
    d1 = (math.log(S / K) + 0.5 * sigma**2 * T) / (sigma * math.sqrt(T))
    return norm_cdf(d1)

def bs_vega(S: float, K: float, T: float, sigma: float) -> float:
    """Sensitivity of option price to volatility changes."""
    if T <= 1e-8:
        return 0.0
    d1 = (math.log(S / K) + 0.5 * sigma**2 * T) / (sigma * math.sqrt(T))
    return S * math.sqrt(T) * norm_pdf(d1)

def implied_vol(market_price: float, S: float, K: float, T: float) -> float:
    """Newton's method to find the σ that makes BS price = market price.
    Returns None if it can't converge (deep OTM, bad data, etc)."""
    if T <= 1e-8:
        return None
    intrinsic = max(0.0, S - K)
    if market_price <= intrinsic + 0.01:
        return None
    sigma = 0.04  # initial guess
    for _ in range(50):
        price = bs_call(S, K, T, sigma)
        vega = bs_vega(S, K, T, sigma)
        if vega < 1e-10:
            return None
        sigma -= (price - market_price) / vega
        if sigma <= 0.001:
            return None
        if abs(price - market_price) < 0.001:
            return sigma
    return None


# ═══════════════════════════════════════════════════════════════
# VEV OPTIONS STRATEGY
# ═══════════════════════════════════════════════════════════════
#
# How it works:
#   1. Read velvetfruit mid price = underlying S
#   2. Compute T = (TTE_START - timestamp/1M) / 8
#      where TTE_START = 8 - CURRENT_DAY
#   3. Estimate σ from the market: compute IV from all near-ATM
#      strikes and use median. Falls back to DEFAULT_SIGMA if
#      not enough data.
#   4. BS(S, K, T, σ) → theoretical fair value for this option
#   5. If market price deviates from fair by more than MIN_EDGE:
#      take it. Also rest passive orders at fair ± edge.
#
# CURRENT_DAY mapping:
#   0 = tutorial/historical day 0 (TTE=8)
#   1 = Round 1 / historical day 1 (TTE=7)
#   2 = Round 2 / historical day 2 (TTE=6)
#   3 = Round 3 (TTE=5) ← set this for submission

# shared state across all VEV instances for live IV estimation
class VEVShared:
    """All VEV strategies share this to pool IV estimates."""
    DEFAULT_SIGMA = 0.0355      # fallback IV from our data analysis
    ema_sigma     = 0.0355      # live EMA of estimated IV
    last_update   = -1          # last timestamp we recalibrated

class VEVStrategy(Strategy):
    CURRENT_DAY  = 3            # *** SET THIS FOR EACH ROUND ***
    TOTAL_LIFE   = 8            # total option lifetime in days (TTE at day 0)
    MIN_EDGE     = 1.5          # minimum mispricing to trade (wider = safer)
    IV_STRIKES   = [5000, 5100, 5200, 5300, 5400, 5500]  # strikes for IV estimation

    def __init__(self, strike: int, position_limit: int = 300):
        super().__init__(f"VEV_{strike}", position_limit)
        self.strike = strike

    def _get_T(self, timestamp: int) -> float:
        """Time to expiry, normalized so T=1.0 at TTE=8d, T=0.0 at expiry."""
        tte_days = (self.TOTAL_LIFE - self.CURRENT_DAY) - timestamp / 1_000_000
        return max(1e-8, tte_days / self.TOTAL_LIFE)

    def _estimate_iv(self, state: TradingState, S: float, T: float):
        """Pool IV estimates from all near-ATM strikes, update EMA."""
        # only recalibrate once per timestamp (first VEV instance triggers)
        if VEVShared.last_update == state.timestamp:
            return
        VEVShared.last_update = state.timestamp

        ivs = []
        for k in self.IV_STRIKES:
            prod = f"VEV_{k}"
            depth = state.order_depths.get(prod)
            if not depth or not depth.buy_orders or not depth.sell_orders:
                continue
            mid = (max(depth.buy_orders) + min(depth.sell_orders)) / 2
            if mid <= 0.5:  # worthless option, skip
                continue
            iv = implied_vol(mid, S, k, T)
            if iv is not None and 0.005 < iv < 0.20:
                ivs.append(iv)

        if ivs:
            # use median of valid IVs
            ivs.sort()
            median_iv = ivs[len(ivs) // 2]
            # EMA: 90% old, 10% new — smooth out noise
            VEVShared.ema_sigma = 0.9 * VEVShared.ema_sigma + 0.1 * median_iv

    def get_orders(self, state: TradingState) -> List[Order]:
        # get underlying price
        vf_depth = state.order_depths.get("VELVETFRUIT_EXTRACT")
        if not vf_depth or not vf_depth.buy_orders or not vf_depth.sell_orders:
            return []
        S = (max(vf_depth.buy_orders) + min(vf_depth.sell_orders)) / 2

        # get this option's order book
        depth = state.order_depths.get(self.product)
        if not depth or not depth.buy_orders or not depth.sell_orders:
            return []

        position = state.position.get(self.product, 0)
        T = self._get_T(state.timestamp)

        # estimate live IV from the market (shared across all VEV instances)
        self._estimate_iv(state, S, T)
        sigma = VEVShared.ema_sigma

        # compute BS fair value for this option
        fair = bs_call(S, self.strike, T, sigma)

        buy_cap = self.position_limit - position
        sell_cap = self.position_limit + position
        orders: List[Order] = []

        # taker: buy cheap asks
        for ask in sorted(depth.sell_orders.keys()):
            if ask < fair - self.MIN_EDGE and buy_cap > 0:
                qty = min(buy_cap, abs(depth.sell_orders[ask]))
                orders.append(self.order(ask, qty))
                buy_cap -= qty

        # taker: sell expensive bids
        for bid in sorted(depth.buy_orders.keys(), reverse=True):
            if bid > fair + self.MIN_EDGE and sell_cap > 0:
                qty = min(sell_cap, depth.buy_orders[bid])
                orders.append(self.order(bid, -qty))
                sell_cap -= qty

        # maker: rest passive orders around fair value
        best_bid = max(depth.buy_orders.keys())
        best_ask = min(depth.sell_orders.keys())

        our_bid = min(best_bid + 1, math.floor(fair - self.MIN_EDGE))
        our_ask = max(best_ask - 1, math.ceil(fair + self.MIN_EDGE))

        if our_bid >= our_ask:
            our_bid = our_ask - 1

        if our_bid >= 1 and buy_cap > 0:
            orders.append(self.order(our_bid, buy_cap))
        if sell_cap > 0:
            orders.append(self.order(our_ask, -sell_cap))

        return orders


# ═══════════════════════════════════════════════════════════════
# STRATEGY REGISTRY
# ═══════════════════════════════════════════════════════════════
#
# Position limits (from wiki):
#   HYDROGEL_PACK: 200
#   VELVETFRUIT_EXTRACT: 200
#   VEV_XXXX: 300 each

STRATEGIES: Dict[str, Strategy] = {
    # Hydrogel: mean-reverts around 10,000 (same as Osmium)
    "HYDROGEL_PACK": MarketMakerStrategy(
        "HYDROGEL_PACK", position_limit=200,
        true_value=10_000, arb_thresh=3
    ),
    # Velvetfruit: pure MM, no true value (drift is unreliable)
    "VELVETFRUIT_EXTRACT": MarketMakerStrategy(
        "VELVETFRUIT_EXTRACT", position_limit=200,
        true_value=None
    )
    # # VEV options: BS arb on all tradeable strikes
    # "VEV_5000": VEVStrategy(strike=5000, position_limit=300),
    # "VEV_5100": VEVStrategy(strike=5100, position_limit=300),
    # "VEV_5200": VEVStrategy(strike=5200, position_limit=300),
    # "VEV_5300": VEVStrategy(strike=5300, position_limit=300),
    # "VEV_5400": VEVStrategy(strike=5400, position_limit=300),
    # "VEV_5500": VEVStrategy(strike=5500, position_limit=300),
}


# ═══════════════════════════════════════════════════════════════
# TRADER (entry point)
# ═══════════════════════════════════════════════════════════════

class Trader:
    def run(self, state: TradingState):
        result = {}
        for product, strategy in STRATEGIES.items():
            if product not in state.order_depths:
                continue
            result[product] = strategy.get_orders(state)
        return result, 0, ""
