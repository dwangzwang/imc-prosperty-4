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
    # ── Tune these ──────────────────────────────────────────────────────────
    BASE_VALUE      = 12_000 
    
    # Accumulation
    INIT_OVERPAY        = 7       # Will buy up to fair_value + 7 to hit MAX position (80)
    
    # Opportunistic Sell Parameters (for occasionally selling what we hold)
    # SELL_EDGE > BUY_EDGE MUST HOLD
    # buy edge should be AT LEAST 4-5 to buy back fast enough
    BUY_EDGE = 4 # how much we're willing to overpay to buy back
    # sell edge should be AT MOST 7 or else it never really happens
    SELL_EDGE = 5 # how much edge we're willing to sell for
    # ────────────────────────────────────────────────────────────────────────

    def __init__(self):
        super().__init__("INTARIAN_PEPPER_ROOT", position_limit=80)

    def get_orders(self, state: TradingState) -> List[Order]:
        depth = state.order_depths[self.product]
        position = state.position.get(self.product, 0)
        timestamp = state.timestamp

        # we can naively buy anything as long as it'll turn a profit by the end
        # by doing so we forgo the chance to wait a short bit of time to buy a much cheaper one
        # we balance these by taking the min of an initial overpay and the actual overpay limit?
        overpay_amt = min(self.INIT_OVERPAY, math.ceil((100_000 - timestamp)/1_000))

        orders: List[Order] = []
        buy_fv = math.floor(self.BASE_VALUE + (timestamp / 1000))
        sell_fv = math.ceil(self.BASE_VALUE + (timestamp / 1000))
        
        buy_cap  = self.position_limit - position
        sell_cap = self.position_limit + position

        # 1. Aggressive Accumulation (Buy & Hold to 80)
        # can buy up to 80 within first 1000 timesteps, so stop this after 1k, and make markets after
        # overfitting to our data...?
        if timestamp < 1000:    
            for ask in sorted(depth.sell_orders):
                if ask > buy_fv + overpay_amt:
                    break
                qty = min(-depth.sell_orders[ask], self.position_limit - position)
                if qty > 0:
                    orders.append(self.order(ask, qty))
                    buy_cap -= qty

        # 2. market making attempt

        # place sells
        for bid in sorted(depth.buy_orders, reverse=True):
            # if people aren't offering high enough, skip
            if bid < sell_fv + self.SELL_EDGE:
                break
            
            qty = min(depth.buy_orders[bid], sell_cap)
            if qty > 0:
                orders.append(self.order(bid, -qty))
                sell_cap -= qty

        # place buys
        for ask in sorted(depth.sell_orders):
            if ask > buy_fv + self.BUY_EDGE:
                break

            qty = min(-depth.sell_orders[ask], buy_cap)
            if qty > 0:
                orders.append(self.order(ask, qty))
                buy_cap -= qty

        # does this do anything?
        # 3. Passive Market Making
        # Provide passive bids to continue accumulating, and passive asks at our minimum sell edge.
        best_bid = max(depth.buy_orders, default=int(buy_fv) - 1)
        best_ask = min(depth.sell_orders, default=int(sell_fv) + 1)
        
        our_bid = min(best_bid + 1, buy_fv - 1)
        our_ask = max(best_ask - 1, sell_fv + self.SELL_EDGE)

        if buy_cap > 0:
            orders.append(self.order(our_bid, min(buy_cap, 20)))
        if sell_cap > 0:
            orders.append(self.order(our_ask, -min(sell_cap, 20)))

        return orders
        
# ══════════════════════════════════════════════════════════════════════════════
# OSMIUM STRATEGY (applied to TOMATOES and ASH_COATED_OSMIUM)
# Fair value is unknown and drifts. We estimate it using WallMid.
# ══════════════════════════════════════════════════════════════════════════════

class OsmiumStrategy(Strategy):
    # ── Tune these ──────────────────────────────────────────────────────────
    MAX_PASSIVE_QTY   = 40     # How much liquidity to provide per tick. Higher = more profit potential, but more risk if swept.
    SKEW_DIVISOR      = 15     # Shift our prices 1 tick for every 15 units of inventory.
    VOLATILITY_THRESH = 20     # If recent trade volume > 20, widen spreads to protect from toxic flow.
    PANIC_THRESHOLD   = 75     # Emergency dump if inventory reaches this level.
    # ────────────────────────────────────────────────────────────────────────

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

        # 3. Dynamic Inventory Skewing (The "Lean")
        # Example: If long (+30), we shift limits down by 2. This makes us sell cheaper and buy lower.
        inventory_skew = position // self.SKEW_DIVISOR
        
        bid_limit = ((mid_2x - 1) // 2) - inventory_skew
        ask_limit = (mid_2x // 2) + 1 - inventory_skew

        # 4. Volatility Protection
        # If market volume is high, the random walk is taking huge steps. Widen our spread.
        market_trades = state.market_trades.get(self.product, [])
        total_volume = sum(t.quantity for t in market_trades)
        
        if total_volume > self.VOLATILITY_THRESH:
            bid_limit -= 1
            ask_limit += 1

        orders: List[Order] = []
        buy_cap = self.position_limit - position
        sell_cap = self.position_limit + position

        # 5. Step 1: Taking (Aggressive)
        for price in sorted_asks:
            if price <= bid_limit and buy_cap > 0:
                qty = min(buy_cap, abs(depth.sell_orders[price]))
                orders.append(self.order(price, qty))
                buy_cap -= qty
                position += qty # Update local position state

        for price in sorted_bids:
            if price >= ask_limit and sell_cap > 0:
                qty = min(sell_cap, depth.buy_orders[price])
                orders.append(self.order(price, -qty))
                sell_cap -= qty
                position -= qty

        # 6. Step 2: Posting (Passive)
        # Attempt to penny the market, but respect our calculated limits
        our_bid = min(bid_limit, best_bid + 1)
        our_ask = max(ask_limit, best_ask - 1)

        # Safety: Ensure a minimum 1-tick spread to prevent order rejection
        if our_bid >= our_ask:
            our_bid = our_ask - 1

        if buy_cap > 0:
            orders.append(self.order(our_bid, min(self.MAX_PASSIVE_QTY, buy_cap)))
        if sell_cap > 0:
            orders.append(self.order(our_ask, -min(self.MAX_PASSIVE_QTY, sell_cap)))

        # 7. Step 3: Emergency Failsafe
        # If the market trends violently and our skewing wasn't enough, dump before we hit the hard limit.
        if abs(position) >= self.PANIC_THRESHOLD:
            # Target the opposite best price to ensure an immediate fill
            flatten_price = best_bid if position > 0 else best_ask
            orders.append(self.order(flatten_price, -position))

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