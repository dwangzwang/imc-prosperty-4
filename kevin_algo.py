from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List, Dict
import math

# ══════════════════════════════════════════════════════════════════════════════
# BASE CLASS
# All product strategies inherit from this. Implement `get_orders()`.
# ══════════════════════════════════════════════════════════════════════════════

class Strategy:
    """Base class for a single-product strategy."""

    def __init__(self, product: str, position_limit: int):
        self.product = product
        self.position_limit = position_limit

    def get_orders(self, state: TradingState) -> List[Order]:
        """Override this in each product subclass."""
        raise NotImplementedError(f"{self.__class__.__name__} must implement get_orders()")

    def order(self, price: int, qty: int) -> Order:
        """Helper: positive qty = buy, negative qty = sell."""
        return Order(self.product, price, qty)

# ══════════════════════════════════════════════════════════════════════════════
# INTARIAN_PEPPER_ROOT (Hybrid: Buy & Hold + Market Making)
# Uses a "Hold Target" to stash assets and a "MM Zone" to scalp.
# ══════════════════════════════════════════════════════════════════════════════

class IntarianPepperRootStrategy(Strategy):
    INIT_OVERPAY = 7
    BUY_EDGE     = 5
    SELL_EDGE    = 6

    def __init__(self):
        super().__init__("INTARIAN_PEPPER_ROOT", position_limit=80)
        self.base_value = None

    def get_orders(self, state: TradingState) -> List[Order]:
        depth     = state.order_depths[self.product]
        position  = state.position.get(self.product, 0)
        timestamp = state.timestamp

        if not depth.buy_orders or not depth.sell_orders:
            return []

        # Anchor fair value on first tick's mid (linear drift is known-correct)
        if self.base_value is None:
            self.base_value = (max(depth.buy_orders) + min(depth.sell_orders)) / 2 - (timestamp / 1000)

        buy_fv  = math.floor(self.base_value + timestamp / 1000)
        sell_fv = math.ceil(self.base_value  + timestamp / 1000)

        # Overpay decays to 0 at end of day. Desperation adds urgency when empty,
        # but is NOT throttled by time — being empty late is the worst case, not the best.
        base_overpay = min(self.INIT_OVERPAY, math.ceil((100_000 - timestamp) / 1_000))
        desperation  = int(8 * (self.position_limit - position) / self.position_limit)
        overpay_amt  = base_overpay + desperation

        max_buy  = buy_fv  + max(overpay_amt, self.BUY_EDGE)
        min_sell = sell_fv + self.SELL_EDGE

        buy_cap  = self.position_limit - position
        sell_cap = self.position_limit + position
        orders: List[Order] = []

        # 1. Aggressive taking
        for ask in sorted(depth.sell_orders):
            if ask > max_buy or buy_cap <= 0:
                break
            qty = min(-depth.sell_orders[ask], buy_cap)
            orders.append(self.order(ask, qty))
            buy_cap -= qty

        for bid in sorted(depth.buy_orders, reverse=True):
            if bid < min_sell or sell_cap <= 0:
                break
            qty = min(depth.buy_orders[bid], sell_cap)
            orders.append(self.order(bid, -qty))
            sell_cap -= qty

        # 2. Passive resting — bid at fair value or penny the book, whichever is lower.
        # Never rest a passive bid above fair value (old algo's discipline).
        best_bid = max(depth.buy_orders)
        best_ask = min(depth.sell_orders)

        our_bid = min(best_bid + 1, buy_fv - 1)
        our_ask = max(best_ask - 1, min_sell)

        if our_bid >= our_ask:
            our_bid = our_ask - 1

        if buy_cap > 0:
            orders.append(self.order(our_bid, buy_cap))
        if sell_cap > 0:
            orders.append(self.order(our_ask, -sell_cap))

        return orders
        
# ══════════════════════════════════════════════════════════════════════════════
# OSMIUM STRATEGY (applied to TOMATOES and ASH_COATED_OSMIUM)
# Fair value is unknown and drifts. We estimate it using WallMid.
# ══════════════════════════════════════════════════════════════════════════════

class OsmiumStrategy(Strategy):
    # ── Tune these ──────────────────────────────────────────────────────────
    MAX_PASSIVE_QTY   = 75     # How much liquidity to provide per tick. Higher = more profit potential, but more risk if swept.
    SKEW_DIVISOR      = 30     # Shift our prices 1 tick for every SKEW_DIVISOR units of inventory.
    VOLATILITY_THRESH = 50     # If recent trade volume > 20, widen spreads to protect from toxic flow.
    
    TRUE_VALUE        = 10_000 # Absolute fair value for Osmium
    ARB_THRESH        = 10      # Max allowed deviation from true value before hard arbitration kicks in
    # ────────────────────────────────────────────────────────────────────────

    def __init__(self, product: str, position_limit: int):
        super().__init__(product, position_limit)
        self.ema_mid = None

    def _deep_weighted_mid_2x(self, depth: OrderDepth) -> int:
        """Calculates mid-price weighted by the top 3 levels of volume to prevent 1-lot baiting."""
        top_bids = sorted(depth.buy_orders.items(), reverse=True)[:3]
        top_asks = sorted(depth.sell_orders.items())[:3]
        
        if not top_bids or not top_asks: 
            return 0
            
        total_bid_vol = sum(abs(v) for p, v in top_bids)
        total_ask_vol = sum(abs(v) for p, v in top_asks)
        
        if total_bid_vol == 0 or total_ask_vol == 0:
            return 0
        
        bid_weight = sum(p * abs(v) for p, v in top_bids)
        ask_weight = sum(p * abs(v) for p, v in top_asks)
        
        # Deep Weighted Mid formula
        w_mid_2x = ((bid_weight / total_bid_vol) * total_ask_vol + (ask_weight / total_ask_vol) * total_bid_vol) * 2 / (total_bid_vol + total_ask_vol)
        return int(w_mid_2x)

    def get_orders(self, state: TradingState) -> List[Order]:
        depth = state.order_depths[self.product]
        position = state.position.get(self.product, 0)
        
        if not depth.buy_orders or not depth.sell_orders:
            return []

        # 1. Pre-calculate sorted keys for efficiency
        sorted_bids = sorted(depth.buy_orders.keys(), reverse=True)
        sorted_asks = sorted(depth.sell_orders.keys())
        best_bid = sorted_bids[0]
        best_ask = sorted_asks[0]

        # 2. Calculate Deep Mid
        mid_2x = self._deep_weighted_mid_2x(depth)
        if mid_2x == 0: return []

        # 3. Local Mean Reversion Fade (The edge generator against short-term noise)
        current_mid = (best_bid + best_ask) / 2
        if self.ema_mid is None:
            self.ema_mid = current_mid
        else:
            self.ema_mid = 0.9 * self.ema_mid + 0.1 * current_mid
        
        diff_from_ema = current_mid - self.ema_mid
        mid_2x -= int(diff_from_ema * 2)

        # 3b. Absolute True Value Arbitrage (Gravity to 10k)
        # If the market has strayed fundamentally away from 10k entirely, 
        # we don't just want to fade local ticks—we want to enforce a hard macro directional bias.
        diff_from_true = current_mid - self.TRUE_VALUE
        
        # If it wanders beyond our allowed stray bound...
        if abs(diff_from_true) > self.ARB_THRESH:
            # Calculate exactly how many ticks 'out of bounds' it is.
            over_drift = diff_from_true - (self.ARB_THRESH if diff_from_true > 0 else -self.ARB_THRESH)
            # Mathematically 'yoink' our pricing back down/up by the exact overflow amount.
            # E.g. if price is 10010 (drift +10), and thresh is 3. We are 7 ticks too high.
            # We subtract 14 half-ticks to forcefully cap our subjective valuation at exactly 10003.
            mid_2x -= int(over_drift * 2)

        # 4. Dynamic Inventory Skewing (The "Lean")
        inventory_skew = position // self.SKEW_DIVISOR
        
        bid_limit = ((mid_2x - 1) // 2) - inventory_skew
        ask_limit = (mid_2x // 2) + 1 - inventory_skew
        
        # 5. Volatility Protection
        market_trades = state.market_trades.get(self.product, [])
        total_volume = sum(t.quantity for t in market_trades)
        
        if total_volume > self.VOLATILITY_THRESH:
            bid_limit -= 1
            ask_limit += 1

        orders: List[Order] = []
        buy_cap = self.position_limit - position
        sell_cap = self.position_limit + position

        # 6. Step 1: Taking (Aggressive)
        for price in sorted_asks:
            if price <= bid_limit and buy_cap > 0:
                qty = min(buy_cap, abs(depth.sell_orders[price]))
                orders.append(self.order(price, qty))
                buy_cap -= qty
                position += qty 

        for price in sorted_bids:
            if price >= ask_limit and sell_cap > 0:
                qty = min(sell_cap, depth.buy_orders[price])
                orders.append(self.order(price, -qty))
                sell_cap -= qty
                position -= qty

        # 7. Step 2: Posting (Passive)
        # Original smart-pennying logic maximizes edge
        our_bid = min(bid_limit, best_bid + 1)
        our_ask = max(ask_limit, best_ask - 1)

        if our_bid >= our_ask:
            our_bid = our_ask - 1

        if buy_cap > 0:
            orders.append(self.order(our_bid, min(self.MAX_PASSIVE_QTY, buy_cap)))
        if sell_cap > 0:
            orders.append(self.order(our_ask, -min(self.MAX_PASSIVE_QTY, sell_cap)))

        return orders


class OsmiumStrategy2(Strategy):
    """
    A drastically simplified 'lean' strategy for Osmium.
    Relies entirely on: 
    1. Wall-Mid Pricing (filtering out sub-15 order sizes to find the true depth anchors).
    2. Linear Inventory Skew (pushing price strictly up/down based on bag limits).
    3. Mandatory Edge execution (never trading for zero theoretical edge).
    4. Structural Mean-Reversion (build long/short bags if price strays from 10k).
    """
    # ── Tune these ──────────────────────────────────────────────────────────
    SKEW_DIVISOR = 40   # Higher = tolerate inventory longer before skewing limits.
    MIN_EDGE     = 1    # The absolute bare minimum tick-profit demanded on every single trade.
    WALL_VOL     = 15    # Ignore any price level that has less than this volume (filters spoofers).
    
    TRUE_VALUE   = 10_000 # The permanent anchor point for structural mean-reversion.
    ARB_THRESH   = 3      # How many ticks the market can wander before we hard-lean into it.
    # ────────────────────────────────────────────────────────────────────────

    def __init__(self, product: str, position_limit: int):
        super().__init__(product, position_limit)

    def _get_wall_mid(self, depth: OrderDepth) -> float:
        """Finds the first bid/ask levels that have substantial volume."""
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

        # 1. Wall Value (Local Anchor)
        wall_mid = self._get_wall_mid(depth)
        
        # 1b. Structural Mean-Reversion Arbitrage (Macro Anchor)
        # Osmium reliably mean-reverts to 10k. If the wall mid wanders significantly 
        # away from 10,000, we forcefully drag our baseline calculation back towards it.
        # This acts as a mathematical magnet, seamlessly triggering directional Taker orders.
        diff_from_true = wall_mid - self.TRUE_VALUE
        
        if diff_from_true > self.ARB_THRESH:
            # Overvalued: Pull fair value down so we refuse to buy, and eagerly sweep bids (short).
            wall_mid -= (diff_from_true - self.ARB_THRESH)
        elif diff_from_true < -self.ARB_THRESH:
            # Undervalued: Pull fair value up so we refuse to short, and eagerly sweep asks (long).
            wall_mid -= (diff_from_true + self.ARB_THRESH)
        
        # 2. Linear Skew (Tilt heavily as we approach position limits to prevent bagging out)
        inventory_skew = position / self.SKEW_DIVISOR
        
        buy_fv = math.floor(wall_mid - inventory_skew)
        sell_fv = math.ceil(wall_mid - inventory_skew)

        buy_cap = self.position_limit - position
        sell_cap = self.position_limit + position
        orders: List[Order] = []
        
        # 3. Market Taking (Aggressive Arbitration)
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

        # 4. Market Making (Passive Liquidity)
        best_bid = max(depth.buy_orders.keys())
        best_ask = min(depth.sell_orders.keys())
        
        # Aggressively penny, but STRICTLY capped by our demanded minimum edge
        our_bid = min(best_bid + 1, buy_fv - self.MIN_EDGE)
        our_ask = max(best_ask - 1, sell_fv + self.MIN_EDGE)
        
        # Ensure spread mathematically doesn't invert
        if our_bid >= our_ask:
            our_bid = our_ask - 1
            
        if buy_cap > 0:
            orders.append(self.order(our_bid, buy_cap))
        if sell_cap > 0:
            orders.append(self.order(our_ask, -sell_cap))

        return orders

# ══════════════════════════════════════════════════════════════════════════════
# REGISTRY — add / remove products here
# ══════════════════════════════════════════════════════════════════════════════

STRATEGIES: Dict[str, Strategy] = {
    "INTARIAN_PEPPER_ROOT": IntarianPepperRootStrategy(),
    # "ASH_COATED_OSMIUM": OsmiumStrategy("ASH_COATED_OSMIUM", position_limit = 80)
    "ASH_COATED_OSMIUM": OsmiumStrategy2("ASH_COATED_OSMIUM", position_limit = 80)
}


# ══════════════════════════════════════════════════════════════════════════════
# TRADER — dispatches each product to its strategy
# ══════════════════════════════════════════════════════════════════════════════

class Trader:

    def bid(self):
        return 4_000

    def run(self, state: TradingState):
        result = {}

        for product, strategy in STRATEGIES.items():
            if product not in state.order_depths:
                continue
            
            result[product] = strategy.get_orders(state)

        return result, 0, ""