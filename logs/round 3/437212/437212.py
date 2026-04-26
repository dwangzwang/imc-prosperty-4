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
# This is the exact same strategy we used for Osmium in Rounds 1-2.
# It works for any product that mean-reverts around a stable value.
#
# How it works:
#   1. "Wall mid" estimates fair value by finding the first bid/ask
#      levels with real volume (>=WALL_VOL), filtering out thin spoofs.
#   2. If true_value is set, the strategy LEANS toward it when the
#      market drifts too far away (mean-reversion signal).
#      - Price at 10015, true_value=10000, arb_thresh=3 →
#        we cap our fair value at 10003, making us eager to sell.
#   3. Inventory skew pushes our prices down when we're long (want
#      to sell) and up when we're short (want to buy), preventing
#      us from getting stuck at position limits.
#   4. Taker: sweep any mispriced orders sitting on the book.
#   5. Maker: rest passive limit orders to earn spread.
#
# Hydrogel: true_value=10000 (strong mean-reversion to 10k)
# Velvetfruit: true_value=None (drift is unreliable, just follow market)

class MarketMakerStrategy(Strategy):
    WALL_VOL     = 15       # ignore levels thinner than this
    SKEW_DIVISOR = 40       # how fast inventory skew kicks in
    MIN_EDGE     = 1        # minimum profit demanded per trade

    def __init__(self, product: str, position_limit: int,
                 true_value: float = None, arb_thresh: float = 3):
        super().__init__(product, position_limit)
        self.true_value = true_value
        self.arb_thresh = arb_thresh

    def _get_wall_mid(self, depth: OrderDepth) -> float:
        # find first bid level with real volume
        valid_bid = max(depth.buy_orders.keys())
        for bid in sorted(depth.buy_orders.keys(), reverse=True):
            if depth.buy_orders[bid] >= self.WALL_VOL:
                valid_bid = bid
                break

        # find first ask level with real volume
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

        # optional mean-reversion anchor toward true_value
        if self.true_value is not None:
            diff = fair - self.true_value
            if diff > self.arb_thresh:
                fair -= (diff - self.arb_thresh)
            elif diff < -self.arb_thresh:
                fair -= (diff + self.arb_thresh)

        # inventory skew: shift fair value against our position
        skew = position / self.SKEW_DIVISOR
        buy_fv = math.floor(fair - skew)
        sell_fv = math.ceil(fair - skew)

        buy_cap = self.position_limit - position
        sell_cap = self.position_limit + position
        orders: List[Order] = []

        # taker: sweep mispriced orders
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

        # maker: rest passive limit orders
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
# BLACK-SCHOLES HELPERS (no scipy needed — uses math.erf)
# ═══════════════════════════════════════════════════════════════
#
# Black-Scholes prices a European call option:
#   Call = S * N(d1) - K * N(d2)      (r=0 in this game)
#
#   d1 = [ln(S/K) + σ²/2 * T] / (σ√T)
#   d2 = d1 - σ√T
#
# Where:
#   S     = current underlying price (velvetfruit mid)
#   K     = strike price (the XXXX in VEV_XXXX)
#   T     = time until expiry (1.0 = start of round, 0.0 = expiry)
#   σ     = volatility of the underlying
#   N(x)  = cumulative standard normal distribution
#
# We measured σ ≈ 3.55% from the Round 3 data. The idea is:
# if BS says VEV_5200 should be worth 100, but it's trading at 95,
# we buy it (it's cheap). If it's trading at 106, we sell (expensive).

def norm_cdf(x: float) -> float:
    """Standard normal CDF using math.erf."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def bs_call(S: float, K: float, T: float, sigma: float) -> float:
    """Black-Scholes European call price (r=0)."""
    if T <= 1e-8 or sigma <= 0:
        return max(0.0, S - K)
    d1 = (math.log(S / K) + 0.5 * sigma**2 * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * norm_cdf(d1) - K * norm_cdf(d2)

def bs_delta(S: float, K: float, T: float, sigma: float) -> float:
    """Black-Scholes call delta (r=0). How much the option moves per $1 of underlying."""
    if T <= 1e-8:
        return 1.0 if S > K else 0.0
    d1 = (math.log(S / K) + 0.5 * sigma**2 * T) / (sigma * math.sqrt(T))
    return norm_cdf(d1)


# ═══════════════════════════════════════════════════════════════
# VEV OPTIONS STRATEGY
# ═══════════════════════════════════════════════════════════════
#
# Trades a single VEV_XXXX call option using Black-Scholes pricing.
#
# How it works:
#   1. Read velvetfruit mid price from the order book (= underlying S)
#   2. Compute time-to-expiry T from the current timestamp and day
#   3. BS(S, K, T, σ) → theoretical fair value of the option
#   4. Compare fair value to market prices:
#      - Market ask < fair - edge → option is CHEAP, buy it
#      - Market bid > fair + edge → option is EXPENSIVE, sell it
#   5. Rest passive orders at fair ± edge to earn spread
#
# IMPORTANT: You must set CURRENT_DAY before each daily submission!
# The strategy needs to know how much time is left until expiry.
#
# The σ was measured at 3.55% from our backtesting data analysis.
# If PnL is poor, try adjusting σ up or down by 0.5% increments.

class VEVStrategy(Strategy):
    SIGMA       = 0.0355    # implied volatility from our data analysis
    TOTAL_DAYS  = 4         # assumed total round length (days 0-3)
    CURRENT_DAY = 0         # *** SET THIS BEFORE EACH SUBMISSION ***
    MIN_EDGE    = 1.0       # minimum mispricing before we trade

    def __init__(self, strike: int, position_limit: int = 200):
        super().__init__(f"VEV_{strike}", position_limit)
        self.strike = strike

    def get_orders(self, state: TradingState) -> List[Order]:
        # read the underlying price from velvetfruit's order book
        vf_depth = state.order_depths.get("VELVETFRUIT_EXTRACT")
        if not vf_depth or not vf_depth.buy_orders or not vf_depth.sell_orders:
            return []
        S = (max(vf_depth.buy_orders) + min(vf_depth.sell_orders)) / 2

        # read this option's order book
        depth = state.order_depths.get(self.product)
        if not depth or not depth.buy_orders or not depth.sell_orders:
            return []

        position = state.position.get(self.product, 0)

        # time to expiry: T goes from ~1.0 (start of round) to ~0.0 (expiry)
        # each day has 1,000,000 ticks (timestamps 0 to 999,900)
        elapsed_days = self.CURRENT_DAY + state.timestamp / 1_000_000
        T = max(1e-8, (self.TOTAL_DAYS - elapsed_days) / self.TOTAL_DAYS)

        # compute the theoretical fair value of this option
        fair = bs_call(S, self.strike, T, self.SIGMA)

        buy_cap = self.position_limit - position
        sell_cap = self.position_limit + position
        orders: List[Order] = []

        # taker: buy underpriced asks
        for ask in sorted(depth.sell_orders.keys()):
            if ask < fair - self.MIN_EDGE and buy_cap > 0:
                qty = min(buy_cap, abs(depth.sell_orders[ask]))
                orders.append(self.order(ask, qty))
                buy_cap -= qty

        # taker: sell overpriced bids
        for bid in sorted(depth.buy_orders.keys(), reverse=True):
            if bid > fair + self.MIN_EDGE and sell_cap > 0:
                qty = min(sell_cap, depth.buy_orders[bid])
                orders.append(self.order(bid, -qty))
                sell_cap -= qty

        # maker: rest passive limit orders at fair ± edge
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
# Position limits are estimates — verify against the competition
# rules PDF before submitting! Common values in past rounds:
# underlyings ~300-400, options ~200
#
# We skip VEV_4000/4500 (deep ITM, behave like the underlying,
# little option edge) and VEV_6000/6500 (deep OTM, essentially
# worthless at ~0.5, illiquid and no edge).

STRATEGIES: Dict[str, Strategy] = {
    # Hydrogel: mean-reverts around 10,000
    "HYDROGEL_PACK": MarketMakerStrategy(
        "HYDROGEL_PACK", position_limit=300,
        true_value=10_000, arb_thresh=3
    ),
    # Velvetfruit: MM only, no true value anchor (drift is unreliable)
    "VELVETFRUIT_EXTRACT": MarketMakerStrategy(
        "VELVETFRUIT_EXTRACT", position_limit=300,
        true_value=None
    ),
    # VEV options: BS arbitrage on the liquid near-ATM strikes
    "VEV_5000": VEVStrategy(strike=5000, position_limit=200),
    "VEV_5100": VEVStrategy(strike=5100, position_limit=200),
    "VEV_5200": VEVStrategy(strike=5200, position_limit=200),
    "VEV_5300": VEVStrategy(strike=5300, position_limit=200),
    "VEV_5400": VEVStrategy(strike=5400, position_limit=200),
    "VEV_5500": VEVStrategy(strike=5500, position_limit=200),
}


# ═══════════════════════════════════════════════════════════════
# TRADER (entry point for the simulation)
# ═══════════════════════════════════════════════════════════════

class Trader:
    def run(self, state: TradingState):
        result = {}
        for product, strategy in STRATEGIES.items():
            if product not in state.order_depths:
                continue
            result[product] = strategy.get_orders(state)
        return result, 0, ""