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
    WALL_VOL   = 10
    EMA_ALPHA  = 0.12
    MEAN_ALPHA = 0.0001
    SKEW_DIV   = 30
    MR_THRESH  = 30
    MR_SIZE    = 20

    QUOTE_LEVELS = [
        (1, 60),
        (4, 35),
        (8, 25),
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

        mean = saved.get("mean", 10_000.0)
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

        orders: List[Order] = []

        # ── MULTI-LEVEL MAKER ─────────────────────────────────────────────
        remaining_buy  = buy_cap
        remaining_sell = sell_cap

        for ticks, size in self.QUOTE_LEVELS:
            if remaining_buy <= 0 and remaining_sell <= 0:
                break

            buy_size  = min(size, remaining_buy)
            sell_size = min(size, remaining_sell)

            bid_price = min(math.floor(skewed_fair) - ticks, best_ask - 1)
            ask_price = max(math.ceil(skewed_fair)  + ticks, best_bid + 1)

            if bid_price >= ask_price:
                bid_price = ask_price - 1

            if buy_size > 0 and deviation < self.MR_THRESH:
                orders.append(self.order(bid_price, buy_size))
                remaining_buy -= buy_size

            if sell_size > 0 and deviation > -self.MR_THRESH:
                orders.append(self.order(ask_price, -sell_size))
                remaining_sell -= sell_size

        # ── MR TAKER ─────────────────────────────────────────────────────
        if deviation < -self.MR_THRESH and buy_cap > 0:
            orders.append(self.order(best_ask, min(self.MR_SIZE, buy_cap)))
        elif deviation > self.MR_THRESH and sell_cap > 0:
            orders.append(self.order(best_bid, -min(self.MR_SIZE, sell_cap)))

        return orders, saved

class VelvetfruitStrategy(Strategy):
    """
    EMA + cost-basis market maker for VELVETFRUIT_EXTRACT.
    Same cost-basis logic as Hydrogel.
    """
    WALL_VOL     = 10
    EMA_ALPHA    = 0.15
    SKEW_DIV     = 50
    EDGE         = 1
    MM_SIZE      = 20
    MIN_PROFIT   = 1
    COST_WEIGHT  = 0.3

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

    def _update_cost_basis(self, state: TradingState, saved: dict) -> float:
        """Track average cost of current position via own_trades."""
        total_cost = saved.get("total_cost", 0.0)
        total_qty  = saved.get("total_qty", 0)

        trades = state.own_trades.get(self.product, [])
        for trade in trades:
            fill_qty   = trade.quantity
            fill_price = trade.price

            if total_qty == 0:
                total_cost = fill_price * abs(fill_qty)
                total_qty  = fill_qty
            elif (total_qty > 0 and fill_qty > 0) or (total_qty < 0 and fill_qty < 0):
                total_cost += fill_price * abs(fill_qty)
                total_qty  += fill_qty
            else:
                close_qty = min(abs(fill_qty), abs(total_qty))
                remaining_fill = abs(fill_qty) - close_qty

                if abs(fill_qty) >= abs(total_qty):
                    if remaining_fill > 0:
                        total_cost = fill_price * remaining_fill
                        total_qty  = fill_qty + total_qty
                    else:
                        total_cost = 0.0
                        total_qty  = 0
                else:
                    avg = total_cost / abs(total_qty) if total_qty != 0 else fill_price
                    total_qty  += fill_qty
                    total_cost = avg * abs(total_qty)

        saved["total_cost"] = total_cost
        saved["total_qty"]  = total_qty

        if total_qty != 0:
            return total_cost / abs(total_qty)
        return None

    def get_orders(self, state: TradingState, saved: dict) -> Tuple[List[Order], dict]:
        depth = state.order_depths[self.product]
        position = state.position.get(self.product, 0)

        if not depth.buy_orders or not depth.sell_orders:
            return [], saved

        best_bid = max(depth.buy_orders)
        best_ask = min(depth.sell_orders)

        wall_mid = self._get_wall_mid(depth)
        fair = saved.get("fair", wall_mid)
        fair = self.EMA_ALPHA * wall_mid + (1 - self.EMA_ALPHA) * fair
        saved["fair"] = fair

        cost_basis = self._update_cost_basis(state, saved)

        if cost_basis is not None and position != 0:
            blended_fair = (1 - self.COST_WEIGHT) * fair + self.COST_WEIGHT * cost_basis
        else:
            blended_fair = fair

        skew = -(position / self.SKEW_DIV)
        skewed_fair = blended_fair + skew

        our_bid = math.floor(skewed_fair) - self.EDGE
        our_ask = math.ceil(skewed_fair)  + self.EDGE

        # profit floor
        if cost_basis is not None:
            if position > 0:
                our_ask = max(our_ask, math.ceil(cost_basis + self.MIN_PROFIT))
            elif position < 0:
                our_bid = min(our_bid, math.floor(cost_basis - self.MIN_PROFIT))

        our_bid = min(our_bid, best_ask - 1)
        our_ask = max(our_ask, best_bid + 1)

        if our_bid >= our_ask:
            our_bid = our_ask - 1

        buy_cap  = self.position_limit - position
        sell_cap = self.position_limit + position

        buy_size  = min(self.MM_SIZE, int(self.MM_SIZE * (buy_cap  / self.position_limit)))
        sell_size = min(self.MM_SIZE, int(self.MM_SIZE * (sell_cap / self.position_limit)))

        orders: List[Order] = []
        if buy_size > 0 and buy_cap > 0:
            orders.append(self.order(our_bid, buy_size))
        if sell_size > 0 and sell_cap > 0:
            orders.append(self.order(our_ask, -sell_size))

        return orders, saved


# registry
STRATEGIES: Dict[str, Strategy] = {
    "HYDROGEL_PACK": HydrogelStrategy(),
    # "VELVETFRUIT_EXTRACT": VelvetfruitStrategy(),
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
