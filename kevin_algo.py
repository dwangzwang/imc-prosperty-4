from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List, Dict, Tuple, Optional
from round3_algo import VEV_BS_Strategy
import math
import json

# BLACK-SCHOLES HELPERS
def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)

def bs_call(S: float, K: float, T: float, sigma: float) -> float:
    if T <= 1e-8 or sigma <= 0:
        return max(0.0, S - K)
    d1 = (math.log(S / K) + 0.5 * sigma**2 * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * norm_cdf(d1) - K * norm_cdf(d2)

def bs_vega(S: float, K: float, T: float, sigma: float) -> float:
    if T <= 1e-8: return 0.0
    d1 = (math.log(S / K) + 0.5 * sigma**2 * T) / (sigma * math.sqrt(T))
    return S * math.sqrt(T) * norm_pdf(d1)

def implied_vol(market_price: float, S: float, K: float, T: float) -> float:
    if T <= 1e-8: return None
    intrinsic = max(0.0, S - K)
    if market_price <= intrinsic + 0.01: return None
    sigma = 0.04
    for _ in range(50):
        price = bs_call(S, K, T, sigma)
        vega = bs_vega(S, K, T, sigma)
        if vega < 1e-10: return None
        sigma -= (price - market_price) / vega
        if sigma <= 0.001: return None
        if abs(price - market_price) < 0.001: return sigma
    return None

# BASE CLASS
# all product strategies inherit from this
# get_orders() receives a `saved` dict (loaded from traderData) and
# returns (orders, updated_saved) so state survives across ticks.
class Strategy:

    def __init__(self, product: str, position_limit: int):
        self.product = product
        self.position_limit = position_limit

    def get_orders(self, state: TradingState, saved: dict) -> Tuple[List[Order], dict]:
        raise NotImplementedError

    def order(self, price: int, qty: int) -> Order:
        return Order(self.product, price, qty)


# INTARIAN_PEPPER_ROOT (hybrid: buy and hold + market making)
class IntarianPepperRootStrategy(Strategy):
    # PARAMETERS TO TUNE
    INIT_OVERPAY    = 7     # how much we're willing to overpay at the start
    BUY_EDGE        = 5     # max amount to overpay later
    SELL_EDGE       = 6     # minimum amount to oversell later
                            # sell_edge > buy_edge must hold to profit
    CRASH_THRESHOLD = 18    # crash threshold
    CRASH_RECOVER   =  8    # crash recovery

    def __init__(self):
        super().__init__("INTARIAN_PEPPER_ROOT", position_limit=80)
        self.base_value = None
        self.in_crash   = False
        self.crash_fv   = None

    def get_orders(self, state: TradingState, saved: dict) -> Tuple[List[Order], dict]:
        # get market data
        depth     = state.order_depths[self.product]
        position  = state.position.get(self.product, 0)
        timestamp = state.timestamp
        orders: List[Order] = []

        if not depth.buy_orders or not depth.sell_orders:
            return orders

        # calculate mid price, best bid, and best ask
        mid      = (max(depth.buy_orders) + min(depth.sell_orders)) / 2
        best_bid = max(depth.buy_orders)
        best_ask = min(depth.sell_orders)

        # anchor fair value on first tick
        if self.base_value is None:
            self.base_value = mid - timestamp / 1000

        fv = self.base_value + timestamp / 1000

        # crash detection and recovery
        if not self.in_crash and mid < fv - self.CRASH_THRESHOLD:
            self.in_crash = True
            self.crash_fv = fv

        if self.in_crash:
            if mid < fv:
                self.base_value = mid - timestamp / 1000
                fv = mid
            if mid >= self.crash_fv - self.CRASH_RECOVER:
                self.in_crash = False
                self.crash_fv = None

        # crash mode: dump position
        if self.in_crash:
            sell_cap = self.position_limit + position
            for bid in sorted(depth.buy_orders, reverse=True):
                if sell_cap <= 0:
                    break
                qty = min(depth.buy_orders[bid], sell_cap)
                orders.append(self.order(bid, -qty))
                sell_cap -= qty
            return orders, saved

        # normal operation: buy and hold + market making
        buy_fv  = math.floor(fv)
        sell_fv = math.ceil(fv)

        # adjust overpayment to remaining duration
        base_overpay = min(self.INIT_OVERPAY, math.ceil((100_000 - timestamp) / 1_000))
        desperation  = int(8 * (self.position_limit - position) / self.position_limit)
        overpay_amt  = base_overpay + desperation

        max_buy  = buy_fv  + max(overpay_amt, self.BUY_EDGE)
        min_sell = sell_fv + self.SELL_EDGE

        buy_cap  = self.position_limit - position
        sell_cap = self.position_limit + position

        # aggressive taking
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

        # passive resting
        our_bid = min(best_bid + 1, buy_fv - 1)
        our_ask = max(best_ask - 1, min_sell)

        if our_bid >= our_ask:
            our_bid = our_ask - 1

        if buy_cap > 0:
            orders.append(self.order(our_bid, buy_cap))
        if sell_cap > 0:
            orders.append(self.order(our_ask, -sell_cap))

        return orders, {}

# OSMIUM_STRATEGY
class OsmiumStrategy(Strategy):
    # PARAMETERS TO TUNE
    SKEW_DIVISOR = 40       # skew limits to prevent overexposure
    MIN_EDGE     = 1        # ensure profit on market making
    WALL_VOL     = 15       # filter noise
    TRUE_VALUE   = 10_000   # true fair value
    ARB_THRESH   = 3        # deviation from fv to start taking positions

    def __init__(self, product: str, position_limit: int):
        super().__init__(product, position_limit)

    def _get_wall_mid(self, depth: OrderDepth) -> float:
        # calculate approximate mid price
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
        # get market data
        depth = state.order_depths[self.product]
        position = state.position.get(self.product, 0)
        
        if not depth.buy_orders or not depth.sell_orders:
            return []

        wall_mid = self._get_wall_mid(depth)
        
        # structural mean-reversion arbitrage
        diff_from_true = wall_mid - self.TRUE_VALUE
        
        if diff_from_true > self.ARB_THRESH:
            # overvalued: adjust fair value down to go short
            wall_mid -= (diff_from_true - self.ARB_THRESH)
        elif diff_from_true < -self.ARB_THRESH:
            # undervalued: adjust fair value up to go long
            wall_mid -= (diff_from_true + self.ARB_THRESH)
        
        # linear skew
        inventory_skew = position / self.SKEW_DIVISOR
        
        buy_fv = math.floor(wall_mid - inventory_skew)
        sell_fv = math.ceil(wall_mid - inventory_skew)

        buy_cap = self.position_limit - position
        sell_cap = self.position_limit + position
        orders: List[Order] = []
        
        # market taking
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

        # market making
        best_bid = max(depth.buy_orders.keys())
        best_ask = min(depth.sell_orders.keys())
        
        our_bid = min(best_bid + 1, buy_fv - self.MIN_EDGE)
        our_ask = max(best_ask - 1, sell_fv + self.MIN_EDGE)
        
        # ensure spread doesn't invert
        if our_bid >= our_ask:
            our_bid = our_ask - 1
            
        if buy_cap > 0:
            orders.append(self.order(our_bid, buy_cap))
        if sell_cap > 0:
            orders.append(self.order(our_ask, -sell_cap))

        return orders, {}

class HydrogelStrategy(Strategy):
    WALL_VOL    = 5      # min size to count as a wall
    EMA_ALPHA   = 0.18    # fast EMA for fair value
    MEAN_ALPHA  = 0.0001  # very slow EMA anchored at 10,000
    SKEW_DIV    = 30      # inventory skew divisor
    EDGE        = 10       # maker half-spread
    MM_SIZE     = 30      # base quote size per side
    MR_THRESH   = 30       # ticks from mean to trigger MR taker
    MR_SIZE     = 20      # units per MR order

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
            orders.append(self.order(our_bid, buy_size/2))
            orders.append(self.order(our_bid, buy_size/2))
        if sell_size > 0 and sell_cap > 0 and mr_bias > -self.MR_THRESH:
            orders.append(self.order(our_ask, -sell_size/2))
            orders.append(self.order(our_ask, -sell_size/2))

    

        # MR taker
        deviation = wall_mid - mean

        if deviation < -self.MR_THRESH and buy_cap > 0:
            orders.append(self.order(best_ask, min(self.MR_SIZE, buy_cap)/2))
            orders.append(self.order(best_ask, min(self.MR_SIZE, buy_cap)/2))
        elif deviation > self.MR_THRESH and sell_cap > 0:
            orders.append(self.order(best_bid, -min(self.MR_SIZE, sell_cap)/2))
            orders.append(self.order(best_bid, -min(self.MR_SIZE, sell_cap)/2))

        return orders, saved

class VelvetfruitStrategy(Strategy):
    # Market Making & Smoothing
    WALL_VOL   = 10
    EMA_ALPHA  = 0.12
    MEAN_ALPHA = 0.0001 
    SKEW_DIV   = 50
    EDGE       = 1
    MM_SIZE    = 30

    # Mean Reversion
    MR_THRESH  = 30
    MR_SIZE    = 20

    # Voucher Signal (The "Lead-Lag" Component)
    VOUCHER_EMA_ALPHA = 0.15
    VOUCHER_BETA      = 0.8  # Sensitivity of VEV to Voucher moves

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

        # 1. Update Long-term Mean
        mean = saved.get("mean", wall_mid)
        mean = self.MEAN_ALPHA * wall_mid + (1 - self.MEAN_ALPHA) * mean
        saved["mean"] = mean

        # 2. Update Short-term EMA (Fair Value)
        fair = saved.get("fair", wall_mid)
        fair = self.EMA_ALPHA * wall_mid + (1 - self.EMA_ALPHA) * fair

        # 3. Incorporate Voucher Lead Signal
        # We look for price pressure in the vouchers to predict VEV's next move
        v_depth = state.order_depths.get("VEV_VOUCHER")
        if v_depth and v_depth.buy_orders and v_depth.sell_orders:
            v_mid = (max(v_depth.buy_orders) + min(v_depth.sell_orders)) / 2
            v_ema = saved.get("v_ema", v_mid)
            v_ema = self.VOUCHER_EMA_ALPHA * v_mid + (1 - self.VOUCHER_EMA_ALPHA) * v_ema
            saved["v_ema"] = v_ema

            # If Voucher is above its EMA, it suggests upward pressure on the underlying
            v_diff = v_mid - v_ema
            fair += (v_diff * self.VOUCHER_BETA)
        
        saved["fair"] = fair

        # 4. Inventory and Deviation Logic
        skew        = -(position / self.SKEW_DIV)
        skewed_fair = fair + skew
        deviation   = wall_mid - mean

        buy_cap  = self.position_limit - position
        sell_cap = self.position_limit + position

        # 5. Market Making Quotes
        our_bid = min(math.floor(skewed_fair) - self.EDGE, best_ask - 1)
        our_ask = max(math.ceil(skewed_fair)  + self.EDGE, best_bid + 1)
        
        if our_bid >= our_ask:
            our_bid = our_ask - 1

        buy_size  = min(self.MM_SIZE, int(self.MM_SIZE * (buy_cap  / self.position_limit)))
        sell_size = min(self.MM_SIZE, int(self.MM_SIZE * (sell_cap / self.position_limit)))

        orders: List[Order] = []

        # Maker: Suppress one side if we are too far from the long-term mean
        if buy_size > 0 and buy_cap > 0 and deviation < self.MR_THRESH:
            orders.append(self.order(our_bid, buy_size))
        if sell_size > 0 and sell_cap > 0 and deviation > -self.MR_THRESH:
            orders.append(self.order(our_ask, -sell_size))

        # Mean Reversion Taker: Only fire on high-confidence deviations
        if deviation < -self.MR_THRESH and buy_cap > 0:
            # Aggressively lift the ask
            orders.append(self.order(best_ask, min(self.MR_SIZE, buy_cap)))
        elif deviation > self.MR_THRESH and sell_cap > 0:
            # Aggressively hit the bid
            orders.append(self.order(best_bid, -min(self.MR_SIZE, sell_cap)))

        return orders, saved

# options trading bs
class VEV_Strategy(Strategy):
    """
    Enhanced multi-signal options strategy – RISKY version.
    Same algorithm structure, but more aggressive parameters:
    - Lower thresholds for entry
    - Larger position limits and sizes
    - Wider stops, higher take-profit targets
    - Stronger momentum and vertical signal boosts
    """
    # Imbalance parameters – lower threshold = more trades
    IMB_TH_BASE = 0.12          # Was 0.18 – more sensitive
    IMB_EMA = 0.2               # Slightly faster reaction

    # Vertical spread parameters – lower threshold = more relative value trades
    VERTICAL_SIZE = 5
    VERTICAL_THRESH = 3.0       # Was 4.0 – easier to trigger

    # Momentum and risk – more aggressive
    MOMENTUM_WINDOW = 2         # Shorter = faster reaction
    MAX_POSITION = 240          # Was 180 – let positions run larger
    STOP_LOSS_PCT = 0.12        # Was 0.08 – wider stop, less cutting
    TAKE_PROFIT_PCT = 0.18      # Was 0.12 – let winners run
    VOL_HIGH_THRESH = 0.45      # Was 0.35 – less penalty for volatility
    MAX_TS = 90000

    # Size parameters – LARGER POSITIONS (modified)
    BASE_SIZE = 15              # Was 8 – much larger per tick
    MAX_SIZE = 60               # Was 30 – much larger cap

    def __init__(self, strike: int):
        super().__init__(f"VEV_{strike}", position_limit=300)
        self.strike = strike

    def _get_underlying_data(self, state: TradingState, saved: dict):
        under_depth = state.order_depths.get("VELVETFRUIT_EXTRACT")
        if not under_depth or not under_depth.buy_orders or not under_depth.sell_orders:
            return None, None, 0.0
        best_bid = max(under_depth.buy_orders)
        best_ask = min(under_depth.sell_orders)
        under_mid = (best_bid + best_ask) / 2.0

        prices = saved.get("under_prices", [])
        prices.append(under_mid)
        if len(prices) > self.MOMENTUM_WINDOW + 1:
            prices = prices[-(self.MOMENTUM_WINDOW + 1):]
        saved["under_prices"] = prices

        momentum = 0.0
        volatility = 0.0
        if len(prices) >= 2:
            momentum = (prices[-1] - prices[-2]) / prices[-2]
        if len(prices) >= 3:
            max_p = max(prices)
            min_p = min(prices)
            volatility = (max_p - min_p) / min_p if min_p > 0 else 0.0
        return under_mid, momentum, volatility

    def _get_adjacent_strikes_mids(self, state: TradingState):
        all_mids = {}
        for prod in state.order_depths:
            if prod.startswith("VEV_"):
                depth = state.order_depths[prod]
                if depth.buy_orders and depth.sell_orders:
                    best_bid = max(depth.buy_orders)
                    best_ask = min(depth.sell_orders)
                    all_mids[int(prod.split("_")[1])] = (best_bid + best_ask) / 2.0
        return all_mids

    def _vertical_spread_signal(self, state: TradingState, current_mid: float) -> int:
        all_mids = self._get_adjacent_strikes_mids(state)
        if self.strike not in all_mids:
            return 0
        strikes = sorted(all_mids.keys())
        idx = strikes.index(self.strike)
        signals = []

        if idx > 0:
            K_low = strikes[idx-1]
            spread_market = all_mids[self.strike] - all_mids[K_low]
            fair_spread = max(0, self.strike - K_low)
            mispricing = spread_market - fair_spread
            if mispricing > self.VERTICAL_THRESH:
                signals.append(-1)
            elif mispricing < -self.VERTICAL_THRESH:
                signals.append(1)

        if idx + 1 < len(strikes):
            K_high = strikes[idx+1]
            spread_market = all_mids[K_high] - all_mids[self.strike]
            fair_spread = max(0, K_high - self.strike)
            mispricing = spread_market - fair_spread
            if mispricing > self.VERTICAL_THRESH:
                signals.append(1)
            elif mispricing < -self.VERTICAL_THRESH:
                signals.append(-1)

        if not signals:
            return 0
        total = sum(signals)
        return 1 if total > 0 else (-1 if total < 0 else 0)

    def get_orders(self, state: TradingState, saved: dict):
        depth = state.order_depths.get(self.product)
        if not depth or not depth.buy_orders or not depth.sell_orders:
            return [], saved

        best_bid = max(depth.buy_orders)
        best_ask = min(depth.sell_orders)
        spread = best_ask - best_bid
        mid = (best_bid + best_ask) / 2.0
        position = state.position.get(self.product, 0)
        ts = state.timestamp

        buy_cap = self.position_limit - position
        sell_cap = self.position_limit + position
        orders = []

        # Order book imbalance
        bid_vol = sum(depth.buy_orders.values())
        ask_vol = sum(abs(v) for v in depth.sell_orders.values())
        total = bid_vol + ask_vol
        if total == 0:
            raw_imb = 0.0
        else:
            raw_imb = (bid_vol - ask_vol) / total
        prev_imb = saved.get("smooth_imb", raw_imb)
        smooth_imb = self.IMB_EMA * raw_imb + (1 - self.IMB_EMA) * prev_imb
        saved["smooth_imb"] = smooth_imb

        # Underlying momentum & volatility filter
        under_mid, momentum, volatility = self._get_underlying_data(state, saved)
        momentum_boost = 1.0
        vol_penalty = 1.0
        if under_mid is not None and momentum is not None:
            if (smooth_imb > 0 and momentum > 0) or (smooth_imb < 0 and momentum < 0):
                momentum_boost = 1.6      # Was 1.4 – stronger boost
            elif (smooth_imb > 0 and momentum < -0.0005) or (smooth_imb < 0 and momentum > 0.0005):
                momentum_boost = 0.6      # Was 0.5 – less penalty
            if volatility > self.VOL_HIGH_THRESH:
                vol_penalty = 0.8         # Was 0.6 – less reduction

        # Vertical spread signal
        vert_signal = self._vertical_spread_signal(state, mid)

        # Combine signals
        combined = smooth_imb * momentum_boost * vol_penalty
        if vert_signal != 0:
            if (vert_signal > 0 and combined > 0) or (vert_signal < 0 and combined < 0):
                combined *= 1.4           # Was 1.25 – stronger reinforcement
            else:
                combined *= 0.8           # Was 0.7 – less conflict penalty

        # Adaptive threshold
        dynamic_th = self.IMB_TH_BASE * (0.7 if vert_signal != 0 else 1.0)

        # Dead zone / market making (larger mm size)
        if abs(combined) < dynamic_th:
            if spread >= 2:
                mm_size = min(10, buy_cap, sell_cap)   # Was 5 – larger market making
                if buy_cap > 0:
                    orders.append(self.order(best_bid + 1, mm_size))
                if sell_cap > 0:
                    orders.append(self.order(best_ask - 1, -mm_size))
            return orders, saved

        # Signal trading with aggressive sizing (now BASE_SIZE and MAX_SIZE are larger)
        size = self.BASE_SIZE
        size += int(spread * 2)             # Was 1.5 – more sensitive to spread
        size += int(abs(combined) * 20)     # Was 15 – larger from strong signals
        if vert_signal != 0:
            size += 3                       # Was 2
        size = int(size * vol_penalty)
        size = min(self.MAX_SIZE, max(4, size))  # Minimum 4 (was 3)

        if combined > dynamic_th and buy_cap > 0:
            orders.append(self.order(best_ask, min(size, buy_cap)))
        elif combined < -dynamic_th and sell_cap > 0:
            orders.append(self.order(best_bid, -min(size, sell_cap)))

        # Position management: slower reduction (more risk)
        if position != 0 and under_mid is not None:
            if "entry_price" not in saved and abs(position) > 0:
                saved["entry_price"] = mid
                saved["entry_pos"] = position
            entry = saved.get("entry_price")
            if entry:
                if position > 0:
                    pnl_pct = (mid - entry) / entry
                    if pnl_pct < -self.STOP_LOSS_PCT:
                        # Stop loss: reduce only 1/3 (was 1/2) – less aggressive cutting
                        reduce = min(position // 3, sell_cap)
                        if reduce > 0:
                            orders.append(self.order(best_bid, -reduce))
                            if position - reduce <= 15:
                                saved.pop("entry_price", None)
                    elif pnl_pct > self.TAKE_PROFIT_PCT:
                        # Take profit: reduce 1/2 (was 1/3) – take more profit sooner
                        reduce = min(position // 2, sell_cap)
                        if reduce > 0:
                            orders.append(self.order(best_bid, -reduce))
                elif position < 0:
                    pnl_pct = (entry - mid) / entry
                    if pnl_pct < -self.STOP_LOSS_PCT:
                        reduce = min((-position) // 3, buy_cap)
                        if reduce > 0:
                            orders.append(self.order(best_ask, reduce))
                            if -position - reduce <= 15:
                                saved.pop("entry_price", None)
                    elif pnl_pct > self.TAKE_PROFIT_PCT:
                        reduce = min((-position) // 2, buy_cap)
                        if reduce > 0:
                            orders.append(self.order(best_ask, reduce))

        # Soft position limit – looser
        if abs(position) > self.MAX_POSITION:
            reduce = min((abs(position) - self.MAX_POSITION) // 1, 
                        sell_cap if position > 0 else buy_cap)
            if reduce > 0:
                if position > 0:
                    orders.append(self.order(best_bid, -reduce))
                else:
                    orders.append(self.order(best_ask, reduce))

        # Final unwind before expiry (same timing)
        if ts >= self.MAX_TS - 3000:
            if position > 0:
                orders.append(self.order(best_bid, -position))
            elif position < 0:
                orders.append(self.order(best_ask, -position))

        # Update saved state
        saved["last_mid"] = mid
        if under_mid:
            saved["last_under"] = under_mid

        return orders, saved

# ── STRATEGY REGISTRY ────────────────────────────────────────────────────────
STRATEGIES: Dict[str, Strategy] = {
    "HYDROGEL_PACK": HydrogelStrategy(),
    "VELVETFRUIT_EXTRACT": VelvetfruitStrategy(),

    "VEV_4000": VEV_Strategy(4000),
    "VEV_4500": VEV_Strategy(4500),
    "VEV_5000": VEV_Strategy(5000),
    "VEV_5100": VEV_Strategy(5100),
    "VEV_5200": VEV_Strategy(5200),
    "VEV_5300": VEV_Strategy(5300),
    "VEV_5400": VEV_Strategy(5400),
    "VEV_5500": VEV_Strategy(5500),
    "VEV_6000": VEV_Strategy(6000),
    "VEV_6500": VEV_Strategy(6500),
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