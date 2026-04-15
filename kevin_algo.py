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
    BUY_EDGE        = 7       # Will buy up to fair_value + 7 to hit MAX position (80)
    
    # Opportunistic Sell Parameters (for occasionally selling what we hold)
    EXPECTED_WAIT_TICKS = 2_000  # Expected ticks before we can buy it back cheaper
    PROFIT_MARGIN       = 2       # Minimum profit over opportunity cost to justify selling
    
    # Liquidation
    END_TIMESTAMP    = 100_000 # Simulation ends at 100k
    # ────────────────────────────────────────────────────────────────────────

    def __init__(self):
        super().__init__("INTARIAN_PEPPER_ROOT", position_limit=80)

    def get_orders(self, state: TradingState) -> List[Order]:
        depth = state.order_depths[self.product]
        position = state.position.get(self.product, 0)
        timestamp = state.timestamp

        orders: List[Order] = []
        fair_value = self.BASE_VALUE + (timestamp / 1000)
        
        buy_cap  = self.position_limit - position
        sell_cap = self.position_limit + position

        # 1. Aggressive Accumulation (Buy & Hold to 80)
        if position < self.position_limit:
            for ask in sorted(depth.sell_orders):
                if ask > fair_value + self.BUY_EDGE:
                    break
                qty = min(-depth.sell_orders[ask], self.position_limit - position)
                if qty > 0:
                    orders.append(self.order(ask, qty))
                    buy_cap -= qty
                    position += qty

        # 2. Opportunistic Selling (Special Market Making when edge is huge)
        opportunity_cost = self.EXPECTED_WAIT_TICKS / 1000
        large_edge = self.PROFIT_MARGIN + opportunity_cost

        for bid in sorted(depth.buy_orders, reverse=True):
            if bid < fair_value + large_edge:
                break
                
            qty = min(depth.buy_orders[bid], sell_cap)
            if qty > 0:
                orders.append(self.order(bid, -qty))
                sell_cap -= qty
                position -= qty

        # 3. Passive Market Making
        # Provide passive bids to continue accumulating, and passive asks at our minimum sell edge.
        best_bid = max(depth.buy_orders, default=int(fair_value) - 1)
        best_ask = min(depth.sell_orders, default=int(fair_value) + 1)
        
        our_bid = min(best_bid + 1, int(fair_value) - 1)
        our_ask = max(best_ask - 1, int(fair_value + large_edge))

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
    SKEW_THRESHOLD    = 40
    PASSIVE_SPREAD    = 2
    MAX_PASSIVE_QTY   = 20
    NETFLOW_THRESHOLD = 5

    def _weighted_mid_2x(self, depth: OrderDepth) -> int:
        best_bid = max(depth.buy_orders.keys(), default=0)
        best_ask = min(depth.sell_orders.keys(), default=0)
        if best_bid == 0 or best_ask == 0: return 0
        
        v_bid = abs(depth.buy_orders[best_bid])
        v_ask = abs(depth.sell_orders[best_ask])
        return int((best_bid * v_ask + best_ask * v_bid) * 2 / (v_bid + v_ask))

    def get_orders(self, state: TradingState) -> List[Order]:
        depth = state.order_depths[self.product]
        position = state.position.get(self.product, 0)
        
        if not depth.buy_orders or not depth.sell_orders:
            return []

        # Calculate Market Flow (Adverse Selection)
        market_trades = state.market_trades.get(self.product, [])
        net_flow = 0 
        for t in market_trades:
            # If buyer is empty string, a seller hit the bid (aggressive selling)
            if t.buyer == "": net_flow -= t.quantity
            else: net_flow += t.quantity

        mid_2x = self._weighted_mid_2x(depth)
        if mid_2x == 0: return []

        buy_cap = self.position_limit - position
        sell_cap = self.position_limit + position

        bid_limit = (mid_2x - 1) // 2
        ask_limit = (mid_2x // 2) + 1
        flatten_price = (mid_2x + 1) // 2

        # TOXIC FLOW PROTECTION: 
        # If market is being slammed with sells, lower our bid to avoid catching the knife.
        if net_flow < -self.NETFLOW_THRESHOLD:
            bid_limit -= 1
        elif net_flow > self.NETFLOW_THRESHOLD:
            ask_limit += 1

        orders: List[Order] = []

        # Step 1: Taking
        for price, vol in sorted(depth.sell_orders.items()):
            if price <= bid_limit and buy_cap > 0:
                qty = min(buy_cap, abs(vol))
                orders.append(self.order(price, qty))
                buy_cap -= qty

        for price, vol in sorted(depth.buy_orders.items(), reverse=True):
            if price >= ask_limit and sell_cap > 0:
                qty = min(sell_cap, vol)
                orders.append(self.order(price, -qty))
                sell_cap -= qty

        # Step 2: Passive (Competitive Pennying)
        best_bid = max(depth.buy_orders.keys())
        best_ask = min(depth.sell_orders.keys())
        our_bid = min(bid_limit, best_bid + 1)
        our_ask = max(ask_limit, best_ask - 1)

        if our_bid >= our_ask: our_bid = our_ask - 1

        if buy_cap > 0:
            orders.append(self.order(our_bid, min(self.MAX_PASSIVE_QTY, buy_cap)))
        if sell_cap > 0:
            orders.append(self.order(our_ask, -min(self.MAX_PASSIVE_QTY, sell_cap)))

        # Step 3: Flatten
        if abs(position) > self.SKEW_THRESHOLD:
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