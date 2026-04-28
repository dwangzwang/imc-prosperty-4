from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List, Dict, Tuple
import math
import json


# BASE CLASS
class Strategy:

    def __init__(self, product: str, position_limit: int):
        self.product = product
        self.position_limit = position_limit

    def get_orders(self, state: TradingState, saved: dict) -> Tuple[List[Order], dict]:
        raise NotImplementedError

    def order(self, price: int, qty: int) -> Order:
        return Order(self.product, price, qty)


class HydrogelStrategy(Strategy):
    WALL_VOL    = 5      # min size to count as a wall
    EMA_ALPHA   = 0.18    # fast EMA for fair value
    MEAN_ALPHA  = 0.0001  # very slow EMA anchored at 10,000
    SKEW_DIV    = 30      # inventory skew divisor
    EDGE        = 10       # maker half-spread
    MM_SIZE     = 30      # base quote size per side

    MR_THRESH   = 30       # ticks from mean to trigger MR taker
    MR_SIZE     = 20      # units per MR order

    LADDER_LEVELS = [
        (20, 15),   # (ticks below/above mean, size)
        (35, 20),
        (55, 25),
        (80, 30),
    ]

    def __init__(self):
        super().__init__("HYDROGEL_PACK", position_limit=200)

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

    def get_orders(self, state: TradingState, saved: dict) -> Tuple[List[Order], dict]:
        depth    = state.order_depths[self.product]
        position = state.position.get(self.product, 0)

        if not depth.buy_orders or not depth.sell_orders:
            return [], saved

        best_bid = max(depth.buy_orders)
        best_ask = min(depth.sell_orders)
        wall_mid = self._get_wall_mid(depth)

        # Slow mean anchored at 10,000
        mean = saved.get("mean", 10_000.0)
        mean = self.MEAN_ALPHA * wall_mid + (1 - self.MEAN_ALPHA) * mean
        saved["mean"] = mean

        # Fast EMA fair value
        fair = saved.get("fair", wall_mid)
        fair = self.EMA_ALPHA * wall_mid + (1 - self.EMA_ALPHA) * fair
        saved["fair"] = fair

        # Inventory skew
        skew        = -(position / self.SKEW_DIV)
        skewed_fair = fair + skew

        our_bid = math.floor(skewed_fair) - self.EDGE
        our_ask = math.ceil(skewed_fair)  + self.EDGE

        our_bid = min(our_bid, best_ask - 1)
        our_ask = max(our_ask, best_bid + 1)
        if our_bid >= our_ask:
            our_bid = our_ask - 1

        buy_cap  = self.position_limit - position
        sell_cap = self.position_limit + position

        buy_size  = min(self.MM_SIZE, int(self.MM_SIZE * (buy_cap  / self.position_limit)))
        sell_size = min(self.MM_SIZE, int(self.MM_SIZE * (sell_cap / self.position_limit)))

        mr_bias = wall_mid - mean

        orders: List[Order] = []

        # Suppress maker side that fights the MR signal
        if buy_size > 0 and buy_cap > 0 and mr_bias < self.MR_THRESH:
            orders.append(self.order(our_bid, buy_size))
        if sell_size > 0 and sell_cap > 0 and mr_bias > -self.MR_THRESH:
            orders.append(self.order(our_ask, -sell_size))

    

        # MR taker
        deviation = wall_mid - mean
        if deviation < -self.MR_THRESH and buy_cap > 0:
            orders.append(self.order(best_ask, min(self.MR_SIZE, buy_cap)))
        elif deviation > self.MR_THRESH and sell_cap > 0:
            orders.append(self.order(best_bid, -min(self.MR_SIZE, sell_cap)))

        return orders, saved

class VelvetfruitStrategy(Strategy):
    WALL_VOL   = 10
    EMA_ALPHA  = 0.12
    MEAN_ALPHA = 0.0001  # anchored at first tick price, not 10k
    SKEW_DIV   = 50
    EDGE       = 1
    MM_SIZE    = 30
    MR_THRESH  = 30
    MR_SIZE    = 20

    def __init__(self):
        super().__init__("VELVETFRUIT_EXTRACT", position_limit=200)

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

    def get_orders(self, state: TradingState, saved: dict) -> Tuple[List[Order], dict]:
        depth    = state.order_depths[self.product]
        position = state.position.get(self.product, 0)

        if not depth.buy_orders or not depth.sell_orders:
            return [], saved

        best_bid = max(depth.buy_orders)
        best_ask = min(depth.sell_orders)
        wall_mid = self._get_wall_mid(depth)

        # Initialize mean at first observed price, not 10k
        # VEV trades around 5,200 so hardcoding 10k would be wrong
        mean = saved.get("mean", wall_mid)
        mean = self.MEAN_ALPHA * wall_mid + (1 - self.MEAN_ALPHA) * mean
        saved["mean"] = mean

        fair = saved.get("fair", wall_mid)
        fair = self.EMA_ALPHA * wall_mid + (1 - self.EMA_ALPHA) * fair
        saved["fair"] = fair

        skew        = -(position / self.SKEW_DIV)
        skewed_fair = fair + skew
        deviation   = wall_mid - mean

        buy_cap  = self.position_limit - position
        sell_cap = self.position_limit + position

        our_bid = min(math.floor(skewed_fair) - self.EDGE, best_ask - 1)
        our_ask = max(math.ceil(skewed_fair)  + self.EDGE, best_bid + 1)
        if our_bid >= our_ask:
            our_bid = our_ask - 1

        buy_size  = min(self.MM_SIZE, int(self.MM_SIZE * (buy_cap  / self.position_limit)))
        sell_size = min(self.MM_SIZE, int(self.MM_SIZE * (sell_cap / self.position_limit)))

        orders: List[Order] = []

        # Maker — suppress side fighting MR signal
        if buy_size > 0 and buy_cap > 0 and deviation < self.MR_THRESH:
            orders.append(self.order(our_bid, buy_size))
        if sell_size > 0 and sell_cap > 0 and deviation > -self.MR_THRESH:
            orders.append(self.order(our_ask, -sell_size))

        # MR taker — only on strong deviations from the rolling mean
        if deviation < -self.MR_THRESH and buy_cap > 0:
            orders.append(self.order(best_ask, min(self.MR_SIZE, buy_cap)))
        elif deviation > self.MR_THRESH and sell_cap > 0:
            orders.append(self.order(best_bid, -min(self.MR_SIZE, sell_cap)))

        return orders, saved
    
# registry
STRATEGIES: Dict[str, Strategy] = {
    "HYDROGEL_PACK": HydrogelStrategy(),
    "VELVETFRUIT_EXTRACT": VelvetfruitStrategy(),
}


class Trader:
    def run(self, state: TradingState):
        try:
            saved = json.loads(state.traderData)
        except:
            saved = {}

        result = {}

        for product, strategy in STRATEGIES.items():
            if product not in state.order_depths:
                continue
            product_saved = saved.get(product, {})
            orders, product_saved = strategy.get_orders(state, product_saved)
            result[product] = orders
            saved[product] = product_saved

        return result, 0, json.dumps(saved)
