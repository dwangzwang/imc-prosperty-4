from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List, Dict
import string

# ══════════════════════════════════════════════════════════════════════════════
# BASE CLASS
# All product strategies inherit from this. Implement `get_orders()`.
# ══════════════════════════════════════════════════════════════════════════════

class Strategy:
    """Base class for a single-product strategy."""

    def __init__(self, product: str, position_limit: int):
        self.product = product
        self.position_limit = position_limit

    def get_orders(self, depth: OrderDepth, position: int, timestamp: int) -> List[Order]:
        """Override this in each product subclass."""
        raise NotImplementedError(f"{self.__class__.__name__} must implement get_orders()")

    def order(self, price: int, qty: int) -> Order:
        """Helper: positive qty = buy, negative qty = sell."""
        return Order(self.product, price, qty)


# ══════════════════════════════════════════════════════════════════════════════
# EMERALDS (Rainforest Resin)
# True fair value = 10,000. Stable, so we take edge and post tight passives.
# ══════════════════════════════════════════════════════════════════════════════

class EmeraldsStrategy(Strategy):
    # ── Tune these ──────────────────────────────────────────────────────────
    FAIR_VALUE      = 10_000
    SKEW_THRESHOLD  = 60     # flatten inventory when |position| exceeds this
    PASSIVE_SPREAD  = 1      # fallback spread from fair value if book is thin
    MAX_PASSIVE_QTY = 25     # max qty per passive resting quote
    # ────────────────────────────────────────────────────────────────────────

    def __init__(self):
        super().__init__("EMERALDS", position_limit=80)

    def get_orders(self, depth: OrderDepth, position: int, timestamp: int) -> List[Order]:
        orders: List[Order] = []
        buy_cap  =  self.position_limit - position
        sell_cap =  self.position_limit + position

        # Step 1: Take any resting orders that give positive edge
        for ask in sorted(depth.sell_orders):
            if ask >= self.FAIR_VALUE:
                break
            qty = min(-depth.sell_orders[ask], buy_cap)
            if qty > 0:
                print(f"[EMERALDS] TAKE BUY  {qty}x @ {ask}")
                orders.append(self.order(ask, qty))
                buy_cap -= qty

        for bid in sorted(depth.buy_orders, reverse=True):
            if bid <= self.FAIR_VALUE:
                break
            qty = min(depth.buy_orders[bid], sell_cap)
            if qty > 0:
                print(f"[EMERALDS] TAKE SELL {qty}x @ {bid}")
                orders.append(self.order(bid, -qty))
                sell_cap -= qty

        # Step 2: Post passive quotes just inside the existing best bid/ask
        best_bid = max(depth.buy_orders,  default=self.FAIR_VALUE - self.PASSIVE_SPREAD)
        best_ask = min(depth.sell_orders, default=self.FAIR_VALUE + self.PASSIVE_SPREAD)

        our_bid = min(best_bid + 1, self.FAIR_VALUE - 1)
        our_ask = max(best_ask - 1, self.FAIR_VALUE + 1)

        if (qty := min(self.MAX_PASSIVE_QTY, buy_cap)) > 0:
            print(f"[EMERALDS] PASSIVE BID {qty}x @ {our_bid}")
            orders.append(self.order(our_bid, qty))

        if (qty := min(self.MAX_PASSIVE_QTY, sell_cap)) > 0:
            print(f"[EMERALDS] PASSIVE ASK {qty}x @ {our_ask}")
            orders.append(self.order(our_ask, -qty))

        # Step 3: Flatten if inventory is too skewed
        if abs(position) > self.SKEW_THRESHOLD:
            flatten = -position
            print(f"[EMERALDS] FLATTEN {flatten}x @ {self.FAIR_VALUE} (was {position})")
            orders.append(self.order(self.FAIR_VALUE, flatten))

        return orders


# ══════════════════════════════════════════════════════════════════════════════
# INTARIAN_PEPPER_ROOT (Rainforest Resin with linear growth)
# True fair value = 12,000 + (timestamp / 1000).
# ══════════════════════════════════════════════════════════════════════════════

class IntarianPepperRootStrategy(Strategy):
    # ── Tune these ──────────────────────────────────────────────────────────
    SKEW_THRESHOLD  = 60     # flatten inventory when |position| exceeds this
    PASSIVE_SPREAD  = 1      # fallback spread from fair value if book is thin
    MAX_PASSIVE_QTY = 25     # max qty per passive resting quote
    EXTRA_EDGE = 2
    # ────────────────────────────────────────────────────────────────────────

    def __init__(self):
        super().__init__("INTARIAN_PEPPER_ROOT", position_limit=80)

    def get_orders(self, depth: OrderDepth, position: int, timestamp: int) -> List[Order]:
        # Linear growth: 1000 units per 1M timesteps
        # starts at 12,000 on day 0
        fair_value = 12_000 + (timestamp / 1000)

        orders: List[Order] = []
        buy_cap  =  self.position_limit - position
        sell_cap =  self.position_limit + position

        # Step 1: Take any resting orders that give positive edge
        for ask in sorted(depth.sell_orders):
            if ask >= fair_value - self.EXTRA_EDGE:
                break
            qty = min(-depth.sell_orders[ask], buy_cap)
            if qty > 0:
                print(f"[INTARIAN_PEPPER_ROOT] TAKE BUY  {qty}x @ {ask}")
                orders.append(self.order(ask, qty))
                buy_cap -= qty

        for bid in sorted(depth.buy_orders, reverse=True):
            if bid <= fair_value + self.EXTRA_EDGE:
                break
            qty = min(depth.buy_orders[bid], sell_cap)
            if qty > 0:
                print(f"[INTARIAN_PEPPER_ROOT] TAKE SELL {qty}x @ {bid}")
                orders.append(self.order(bid, -qty))
                sell_cap -= qty

        # Step 2: Post passive quotes just inside the existing best bid/ask
        best_bid = max(depth.buy_orders,  default=int(fair_value) - self.PASSIVE_SPREAD)
        best_ask = min(depth.sell_orders, default=int(fair_value) + self.PASSIVE_SPREAD)

        our_bid = min(best_bid + 1, int(fair_value) - 1)
        our_ask = max(best_ask - 1, int(fair_value) + 1)

        if (qty := min(self.MAX_PASSIVE_QTY, buy_cap)) > 0:
            print(f"[INTARIAN_PEPPER_ROOT] PASSIVE BID {qty}x @ {our_bid}")
            orders.append(self.order(our_bid, qty))

        if (qty := min(self.MAX_PASSIVE_QTY, sell_cap)) > 0:
            print(f"[INTARIAN_PEPPER_ROOT] PASSIVE ASK {qty}x @ {our_ask}")
            orders.append(self.order(our_ask, -qty))

        # Step 3: Flatten if inventory is too skewed
        if abs(position) > self.SKEW_THRESHOLD:
            flatten = -position
            print(f"[INTARIAN_PEPPER_ROOT] FLATTEN {flatten}x @ {int(fair_value)} (was {position})")
            orders.append(self.order(int(fair_value), flatten))

        return orders

class IntarianPepperRootStrategy2(Strategy):
    # ── Tune these ──────────────────────────────────────────────────────────
    BASE_VALUE = 12_000      # Starting value at timestamp 0
    BUY_EDGE   = -8           # Increase this to only buy when "extra" cheap
    # ────────────────────────────────────────────────────────────────────────

    def __init__(self):
        super().__init__("INTARIAN_PEPPER_ROOT", position_limit=80)

    def get_orders(self, depth: OrderDepth, position: int, timestamp: int) -> List[Order]:
        # Fair value growth: 1,000 units per 1,000,000 timesteps
        fair_value = self.BASE_VALUE + (timestamp / 1000)
        
        orders: List[Order] = []
        buy_cap = self.position_limit - position

        # Sort asks from cheapest to most expensive
        for ask in sorted(depth.sell_orders):
            # SAFETY CHECK: Only buy if the price is below our projected fair value
            if ask > (fair_value - self.BUY_EDGE):
                break
            
            qty = min(-depth.sell_orders[ask], buy_cap)
            if qty > 0:
                print(f"[INTARIAN_PEPPER_ROOT] TAKE BUY {qty}x @ {ask} (Fair: {fair_value:.2f})")
                orders.append(self.order(ask, qty))
                buy_cap -= qty
            
            # Stop if we hit our position limit
            if buy_cap <= 0:
                break

        return orders

# ══════════════════════════════════════════════════════════════════════════════
# INTARIAN_PEPPER_ROOT (Hybrid: Buy & Hold + Market Making)
# Uses a "Hold Target" to stash assets and a "MM Zone" to scalp.
# ══════════════════════════════════════════════════════════════════════════════

class IntarianPepperRootStrategy3(Strategy):
    # ── Tune these ──────────────────────────────────────────────────────────
    BASE_VALUE      = 12_000 
    HOLD_PORTION    = 65      # Amount to buy and hold long-term
    
    # MM & Taking Parameters
    EXTRA_EDGE      = 0       # Conservative edge for scalping (MM)
    AGGRESSIVE_EDGE = -7      # Willing to pay 8 OVER fair to accumulate HOLD_PORTION
    PASSIVE_SPREAD  = 1      
    
    # Liquidation
    END_TIMESTAMP    = 100_000 # Simulation ends at 100k
    # ────────────────────────────────────────────────────────────────────────

    def __init__(self):
        super().__init__("INTARIAN_PEPPER_ROOT", position_limit=80)

    def get_orders(self, depth: OrderDepth, position: int, timestamp: int) -> List[Order]:
        orders: List[Order] = []
        fair_value = self.BASE_VALUE + (timestamp / 1000)
        
        # 1. Final Liquidation Logic
        if timestamp >= self.END_TIMESTAMP:
            if position > 0:
                best_bid = max(depth.buy_orders, default=int(fair_value))
                orders.append(self.order(best_bid, -position))
            return orders

        buy_cap  = self.position_limit - position
        sell_cap = self.position_limit + position

        # 2. Aggressive Accumulation (Buy & Hold)
        # If we are below our HOLD_PORTION, buy anything up to (Fair + 8)
        if position < self.HOLD_PORTION:
            for ask in sorted(depth.sell_orders):
                if ask > (fair_value - self.AGGRESSIVE_EDGE): # ask > fair + 8
                    break
                qty = min(-depth.sell_orders[ask], self.HOLD_PORTION - position)
                if qty > 0:
                    print(f"[INTARIAN_PEPPER_ROOT] ACCUMULATING: {qty}x @ {ask}")
                    orders.append(self.order(ask, qty))
                    buy_cap -= qty
                    position += qty

        # 3. Conservative Taking (Scalping)
        # Use the EXTRA_EDGE (2) to take profitable orders for the MM portion
        for ask in sorted(depth.sell_orders):
            if ask >= fair_value - self.EXTRA_EDGE:
                break
            qty = min(-depth.sell_orders[ask], buy_cap)
            if qty > 0:
                orders.append(self.order(ask, qty))
                buy_cap -= qty
                position += qty

        for bid in sorted(depth.buy_orders, reverse=True):
            if bid <= fair_value + self.EXTRA_EDGE:
                break
            qty = min(depth.buy_orders[bid], sell_cap)
            if qty > 0:
                orders.append(self.order(bid, -qty))
                sell_cap -= qty
                position -= qty

        # 4. Passive Quotes (Market Making)
        # Place quotes relative to fair value, biased by how far we are from HOLD_PORTION
        best_bid = max(depth.buy_orders, default=int(fair_value) - 1)
        best_ask = min(depth.sell_orders, default=int(fair_value) + 1)
        
        our_bid = min(best_bid + 1, int(fair_value) - 1)
        our_ask = max(best_ask - 1, int(fair_value) + 1)

        if buy_cap > 0:
            orders.append(self.order(our_bid, min(buy_cap, 20)))
        if sell_cap > 0:
            orders.append(self.order(our_ask, -min(sell_cap, 20)))

        return orders
        
# ══════════════════════════════════════════════════════════════════════════════
# KELP STRATEGY (applied to TOMATOES and ASH_COATED_OSMIUM)
# Fair value is unknown and drifts. We estimate it using WallMid.
# ══════════════════════════════════════════════════════════════════════════════

class KelpStrategy(Strategy):
    # ── Tune these ──────────────────────────────────────────────────────────
    SKEW_THRESHOLD  = 40     # flatten inventory when |position| exceeds this
    PASSIVE_SPREAD  = 2      # fallback half-spread when book is empty
    MAX_PASSIVE_QTY = 20     # max qty per passive resting quote
    # ────────────────────────────────────────────────────────────────────────

    def __init__(self, product: str, position_limit: int):
        super().__init__(product, position_limit)

    def _wall_mid_2x(self, depth: OrderDepth) -> int:
        """Return 2 * WallMid as an integer."""
        best_bid = max(depth.buy_orders,  default=0)
        best_ask = min(depth.sell_orders, default=0)
        return best_bid + best_ask 

    def get_orders(self, depth: OrderDepth, position: int, timestamp: int) -> List[Order]:
        if not depth.buy_orders or not depth.sell_orders:
            return []

        mid_2x  = self._wall_mid_2x(depth)
        orders: List[Order] = []
        buy_cap  = self.position_limit - position
        sell_cap = self.position_limit + position

        bid_limit     = (mid_2x - 1) // 2
        ask_limit     = mid_2x // 2 + 1
        flatten_price = (mid_2x + 1) // 2

        # Step 1: Take any resting orders that give positive edge vs WallMid
        for ask in sorted(depth.sell_orders):
            if ask > bid_limit:
                break
            qty = min(-depth.sell_orders[ask], buy_cap)
            if qty > 0:
                print(f"[{self.product}] TAKE BUY  {qty}x @ {ask}")
                orders.append(self.order(ask, qty))
                buy_cap -= qty

        for bid in sorted(depth.buy_orders, reverse=True):
            if bid < ask_limit:
                break
            qty = min(depth.buy_orders[bid], sell_cap)
            if qty > 0:
                print(f"[{self.product}] TAKE SELL {qty}x @ {bid}")
                orders.append(self.order(bid, -qty))
                sell_cap -= qty

        # Step 2: Post passive quotes
        best_bid = max(depth.buy_orders)
        best_ask = min(depth.sell_orders)

        fallback_bid = flatten_price - self.PASSIVE_SPREAD
        fallback_ask = flatten_price + self.PASSIVE_SPREAD

        our_bid = min(best_bid + 1, bid_limit)
        our_ask = max(best_ask - 1, ask_limit)

        our_bid = max(our_bid, fallback_bid)
        our_ask = min(our_ask, fallback_ask)

        if our_bid < our_ask:
            if (qty := min(self.MAX_PASSIVE_QTY, buy_cap)) > 0:
                print(f"[{self.product}] PASSIVE BID {qty}x @ {our_bid}")
                orders.append(self.order(our_bid, qty))

            if (qty := min(self.MAX_PASSIVE_QTY, sell_cap)) > 0:
                print(f"[{self.product}] PASSIVE ASK {qty}x @ {our_ask}")
                orders.append(self.order(our_ask, -qty))

        # Step 3: Flatten
        if abs(position) > self.SKEW_THRESHOLD:
            flatten = -position
            print(f"[{self.product}] FLATTEN {flatten}x @ {flatten_price} (was {position})")
            orders.append(self.order(flatten_price, flatten))

        return orders


# ══════════════════════════════════════════════════════════════════════════════
# REGISTRY — add / remove products here
# ══════════════════════════════════════════════════════════════════════════════

STRATEGIES: Dict[str, Strategy] = {
    "EMERALDS": EmeraldsStrategy(),
    "TOMATOES": KelpStrategy("TOMATOES", position_limit=80),
    "INTARIAN_PEPPER_ROOT": IntarianPepperRootStrategy3(),
    "ASH_COATED_OSMIUM": KelpStrategy("ASH_COATED_OSMIUM", position_limit=80)
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
                
            depth     = state.order_depths[product]
            position  = state.position.get(product, 0)
            timestamp = state.timestamp
            
            result[product] = strategy.get_orders(depth, position, timestamp)

        return result, 0, ""