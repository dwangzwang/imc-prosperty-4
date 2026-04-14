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
# EMERALDS  (Rainforest Resin)
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
            print(f"[EMERALDS] FLATTEN {flatten}x @ {self.FAIR_VALUE}  (was {position})")
            orders.append(self.order(self.FAIR_VALUE, flatten))

        return orders

# ══════════════════════════════════════════════════════════════════════════════
# INTARAN_PEPPER_ROOT  (Rainforest Resin with linear growth)
# True fair value = 10,000 + (day+2) * 1,000 + timestamp/1000.
# ══════════════════════════════════════════════════════════════════════════════

class IntaranPepperRootStrategy(Strategy):
    # ── Tune these ──────────────────────────────────────────────────────────
    SKEW_THRESHOLD  = 60     # flatten inventory when |position| exceeds this
    PASSIVE_SPREAD  = 1      # fallback spread from fair value if book is thin
    MAX_PASSIVE_QTY = 25     # max qty per passive resting quote
    # ────────────────────────────────────────────────────────────────────────

    def __init__(self):
        super().__init__("INTARAN_PEPPER_ROOT", position_limit=80)

    def get_orders(self, depth: OrderDepth, position: int, timestamp:int) -> List[Order]:
        self.FAIR_VALUE = 13_000 + timestamp/1000

        orders: List[Order] = []
        buy_cap  =  self.position_limit - position
        sell_cap =  self.position_limit + position

        # Step 1: Take any resting orders that give positive edge
        for ask in sorted(depth.sell_orders):
            if ask >= self.FAIR_VALUE:
                break
            qty = min(-depth.sell_orders[ask], buy_cap)
            if qty > 0:
                print(f"[INTARAN_PEPPER_ROOT] TAKE BUY  {qty}x @ {ask}")
                orders.append(self.order(ask, qty))
                buy_cap -= qty

        for bid in sorted(depth.buy_orders, reverse=True):
            if bid <= self.FAIR_VALUE:
                break
            qty = min(depth.buy_orders[bid], sell_cap)
            if qty > 0:
                print(f"[INTARAN_PEPPER_ROOT] TAKE SELL {qty}x @ {bid}")
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
            print(f"[EMERALDS] FLATTEN {flatten}x @ {self.FAIR_VALUE}  (was {position})")
            orders.append(self.order(self.FAIR_VALUE, flatten))

        return orders


# ══════════════════════════════════════════════════════════════════════════════
# KELP STRATEGY  (applied to TOMATOES)
# Fair value is unknown and drifts. We estimate it each timestep using the
# WallMid = (best_bid + best_ask) / 2 and quote around that.
# ══════════════════════════════════════════════════════════════════════════════

class KelpStrategy(Strategy):
    # ── Tune these ──────────────────────────────────────────────────────────
    SKEW_THRESHOLD  = 40     # flatten inventory when |position| exceeds this
    PASSIVE_SPREAD  = 2      # fallback half-spread when book is empty
    MAX_PASSIVE_QTY = 20     # max qty per passive resting quote
    # ────────────────────────────────────────────────────────────────────────

    def __init__(self, product: str, position_limit: int, timestamp: int):
        super().__init__(product, position_limit)

    def _wall_mid_2x(self, depth: OrderDepth) -> int:
        """Return 2 * WallMid as an integer to avoid floating-point errors.
        WallMid = (best_bid + best_ask) / 2.
        """
        best_bid = max(depth.buy_orders,  default=0)
        best_ask = min(depth.sell_orders, default=0)
        return best_bid + best_ask  # = 2 * wall_mid

    def get_orders(self, depth: OrderDepth, position: int) -> List[Order]:
        # Guard: need both sides of the book to compute WallMid
        if not depth.buy_orders or not depth.sell_orders:
            return []

        mid_2x  = self._wall_mid_2x(depth)          # 2 * wall_mid (integer)
        orders: List[Order] = []
        buy_cap  = self.position_limit - position
        sell_cap = self.position_limit + position

        # Edge bounds (all integer arithmetic, no float needed)
        #   bid_limit: largest integer strictly below wall_mid  → guaranteed +edge to buy
        #   ask_limit: smallest integer strictly above wall_mid → guaranteed +edge to sell
        #   flatten_price: nearest integer to wall_mid
        bid_limit     = (mid_2x - 1) // 2      # floor(wall_mid - epsilon)
        ask_limit     = mid_2x // 2 + 1        # floor(wall_mid) + 1
        flatten_price = (mid_2x + 1) // 2      # round(wall_mid)

        # Step 1: Take any resting orders that give positive edge vs WallMid
        for ask in sorted(depth.sell_orders):
            if ask > bid_limit:
                break
            qty = min(-depth.sell_orders[ask], buy_cap)
            if qty > 0:
                print(f"[{self.product}] TAKE BUY  {qty}x @ {ask}  (mid_2x={mid_2x})")
                orders.append(self.order(ask, qty))
                buy_cap -= qty

        for bid in sorted(depth.buy_orders, reverse=True):
            if bid < ask_limit:
                break
            qty = min(depth.buy_orders[bid], sell_cap)
            if qty > 0:
                print(f"[{self.product}] TAKE SELL {qty}x @ {bid}  (mid_2x={mid_2x})")
                orders.append(self.order(bid, -qty))
                sell_cap -= qty

        # Step 2: Post passive quotes just inside the existing best bid/ask
        best_bid = max(depth.buy_orders)
        best_ask = min(depth.sell_orders)

        fallback_bid = flatten_price - self.PASSIVE_SPREAD
        fallback_ask = flatten_price + self.PASSIVE_SPREAD

        our_bid = min(best_bid + 1, bid_limit)   # overbid, but never at or above WallMid
        our_ask = max(best_ask - 1, ask_limit)   # undercut, but never at or below WallMid

        # Clamp to fallback if book is so one-sided the formula would invert
        our_bid = max(our_bid, fallback_bid)
        our_ask = min(our_ask, fallback_ask)

        if our_bid < our_ask:  # sanity check — quotes must not cross
            if (qty := min(self.MAX_PASSIVE_QTY, buy_cap)) > 0:
                print(f"[{self.product}] PASSIVE BID {qty}x @ {our_bid}")
                orders.append(self.order(our_bid, qty))

            if (qty := min(self.MAX_PASSIVE_QTY, sell_cap)) > 0:
                print(f"[{self.product}] PASSIVE ASK {qty}x @ {our_ask}")
                orders.append(self.order(our_ask, -qty))

        # Step 3: Flatten if inventory is too skewed
        if abs(position) > self.SKEW_THRESHOLD:
            flatten = -position
            print(f"[{self.product}] FLATTEN {flatten}x @ {flatten_price}  (was {position})")
            orders.append(self.order(flatten_price, flatten))

        return orders


# ══════════════════════════════════════════════════════════════════════════════
# REGISTRY — add / remove products here
# ══════════════════════════════════════════════════════════════════════════════

STRATEGIES: Dict[str, Strategy] = {
    "EMERALDS": EmeraldsStrategy(),
    "TOMATOES": KelpStrategy("TOMATOES", position_limit=80),
    "INTARAN_PEPPER_ROOT": IntaranPepperRootStrategy(),
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
            depth    = state.order_depths[product]
            position = state.position.get(product, 0)
            timestamp = state.timestamp
            result[product] = strategy.get_orders(depth, position, timestamp)

        return result, 0, ""
