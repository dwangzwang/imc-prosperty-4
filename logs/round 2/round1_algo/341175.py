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
    INIT_OVERPAY    = 7
    BUY_EDGE        = 5
    SELL_EDGE       = 6
    CRASH_THRESHOLD = 18
    CRASH_RECOVER   =  8

    def __init__(self):
        super().__init__("INTARIAN_PEPPER_ROOT", position_limit=80)

    def get_orders(self, state: TradingState) -> tuple[List[Order], str]:
        depth     = state.order_depths[self.product]
        position  = state.position.get(self.product, 0)
        timestamp = state.timestamp
        orders: List[Order] = []

        if not depth.buy_orders or not depth.sell_orders:
            return orders, state.traderData

        mid      = (max(depth.buy_orders) + min(depth.sell_orders)) / 2
        best_bid = max(depth.buy_orders)
        best_ask = min(depth.sell_orders)

        # ── Load persisted state ─────────────────────────────────────────────
        try:
            saved = json.loads(state.traderData)
        except Exception:
            saved = {}

        # Fix 1: use `in` check instead of `or` to handle base_value == 0
        base_value = saved["base_value"] if "base_value" in saved else (mid - timestamp / 1000)
        in_crash   = saved.get("in_crash", False)
        crash_fv   = saved.get("crash_fv", None)

        # ── Fair value ───────────────────────────────────────────────────────
        fv = base_value + timestamp / 1000

        # ── Crash detection and recovery ─────────────────────────────────────
        # Fix 3: freeze crash_fv at the moment of crash entry so that the
        # recovery condition is always measured against the original crash level,
        # not the recalibrated floor (which would make recovery always true)
        if not in_crash and mid < fv - self.CRASH_THRESHOLD:
            in_crash = True
            crash_fv = fv                             # freeze reference

        if in_crash:
            if mid < fv:                              # still falling — track floor down
                base_value = mid - timestamp / 1000
                fv = mid
            if mid >= crash_fv - self.CRASH_RECOVER: # recovered vs frozen crash level
                in_crash = False
                crash_fv = None

        # ── Save state ───────────────────────────────────────────────────────
        new_trader_data = json.dumps({
            "base_value": base_value,
            "in_crash":   in_crash,
            "crash_fv":   crash_fv,
        })

        # ── CRASH MODE: hit every bid to dump, no resting orders ─────────────
        # Fix 4: removed resting sell at best_ask - 1 (adverse selection risk)
        if in_crash:
            sell_cap = self.position_limit + position
            for bid in sorted(depth.buy_orders, reverse=True):
                if sell_cap <= 0:
                    break
                qty = min(depth.buy_orders[bid], sell_cap)
                orders.append(self.order(bid, -qty))
                sell_cap -= qty
            return orders, new_trader_data

        # ── NORMAL MODE ───────────────────────────────────────────────────────
        buy_fv  = math.floor(fv)
        sell_fv = math.ceil(fv)

        base_overpay = min(self.INIT_OVERPAY, math.ceil((100_000 - timestamp) / 1_000))
        desperation  = int(8 * (self.position_limit - position) / self.position_limit)
        overpay_amt  = base_overpay + desperation

        max_buy  = buy_fv  + max(overpay_amt, self.BUY_EDGE)
        min_sell = sell_fv + self.SELL_EDGE

        buy_cap  = self.position_limit - position
        sell_cap = self.position_limit + position

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

        # 2. Passive resting
        our_bid = min(best_bid + 1, buy_fv - 1)
        our_ask = max(best_ask - 1, min_sell)

        if our_bid >= our_ask:
            our_bid = our_ask - 1

        if buy_cap > 0:
            orders.append(self.order(our_bid, buy_cap))
        if sell_cap > 0:
            orders.append(self.order(our_ask, -sell_cap))

        return orders, new_trader_data
        
# ══════════════════════════════════════════════════════════════════════════════
# OSMIUM STRATEGY (applied to TOMATOES and ASH_COATED_OSMIUM)
# Fair value is unknown and drifts. We estimate it using WallMid.
# ══════════════════════════════════════════════════════════════════════════════

class OsmiumStrategy(Strategy):
    # ── Tune these ──────────────────────────────────────────────────────────
    MAX_PASSIVE_QTY   = 75     # How much liquidity to provide per tick. Higher = more profit potential, but more risk if swept.
    SKEW_DIVISOR      = 30     # Shift our prices 1 tick for every SKEW_DIVISOR units of inventory.
    VOLATILITY_THRESH = 50     # If recent trade volume > 20, widen spreads to protect from toxic flow.
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

        # 3. Mean Reversion Fade (The edge generator)
        # In a constantly fluctuating but bound market, if price spikes suddenly, 
        # it is usually a false breakout. We use an EMA to track true base, 
        # and aggressively fade any new spikes back to base.
        current_mid = (best_bid + best_ask) / 2
        if self.ema_mid is None:
            self.ema_mid = current_mid
        else:
            self.ema_mid = 0.9 * self.ema_mid + 0.1 * current_mid
        
        # If current_mid is violently higher than ema_mid, we pull mid_2x DOWN to sell the rip
        diff_from_ema = current_mid - self.ema_mid
        mid_2x -= int(diff_from_ema * 2)

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


# ══════════════════════════════════════════════════════════════════════════════
# REGISTRY — add / remove products here
# ══════════════════════════════════════════════════════════════════════════════

STRATEGIES: Dict[str, Strategy] = {
    "INTARIAN_PEPPER_ROOT": IntarianPepperRootStrategy(),
    "ASH_COATED_OSMIUM": OsmiumStrategy("ASH_COATED_OSMIUM", position_limit = 80)
}


# ══════════════════════════════════════════════════════════════════════════════
# TRADER — dispatches each product to its strategy
# ══════════════════════════════════════════════════════════════════════════════

class Trader:

    def run(self, state: TradingState):
        result = {}

        for product, strategy in STRATEGIES.items():
            if product not in state.order_depths:
                continue
            
            result[product] = strategy.get_orders(state)

        return result, 0, ""