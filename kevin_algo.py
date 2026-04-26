from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List, Dict, Tuple
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
    WALL_VOL     = 10       # min size to count as a wall
    SMOOTH_A     = 0.15    # EMA alpha for fair value smoothing
    ROLLING_WINDOW = 45_000
    DOWNSAMPLE   = 4         # max timesteps for pseudo-rolling mean (circular buffer — O(1) writes)
    ARB_THRESH   = 7       # Ticks away from true value before anchor kicks in
    SKEW_DIVISOR = 50      # Higher divisor = weaker skew, allows more volume buildup
    EDGE         = 1       # Min edge from skewed fair value
    MM_SIZE      = 20      # base quote size

    def __init__(self):
        super().__init__("HYDROGEL_PACK", position_limit=200)

    def _get_wall_mid(self, depth: OrderDepth) -> float:
        # Walk from best bid inward until we hit a wall
        valid_bid = max(depth.buy_orders.keys())
        for bid in sorted(depth.buy_orders.keys(), reverse=True):
            if depth.buy_orders[bid] >= self.WALL_VOL:
                valid_bid = bid
                break

        # Walk from best ask inward until we hit a wall
        valid_ask = min(depth.sell_orders.keys())
        for ask in sorted(depth.sell_orders.keys()):
            if abs(depth.sell_orders[ask]) >= self.WALL_VOL:
                valid_ask = ask
                break

        return (valid_bid + valid_ask) / 2

    def get_orders(self, state: TradingState, saved: dict):
        depth = state.order_depths[self.product]
        position = state.position.get(self.product, 0)

        if not depth.buy_orders or not depth.sell_orders:
            return [], saved

        best_bid = max(depth.buy_orders)
        best_ask = min(depth.sell_orders)

        # --- WALL MID ---
        wall_mid = self._get_wall_mid(depth)

        # --- SMOOTHED FAIR VALUE (EMA via traderdata) ---
        fair = saved.get("fair", wall_mid)
        fair = self.SMOOTH_A * wall_mid + (1 - self.SMOOTH_A) * fair
        saved["fair"] = fair

        # --- DYNAMIC TRUE VALUE ANCHOR (Robust Mean Reversion) ---
        # Instead of hardcoding 10,000, we maintain a very slow EMA (alpha=0.001).
        # We initialize it at 10,000. If the actual simulation day is abnormally low 
        # (e.g. centered around 9980), the slow EMA gracefully drifts down to 9980,
        # preventing us from getting stuck continuously buying a "dip" that is actually the new normal.
        prices    = saved.get("prices", [])
        ptr       = saved.get("ptr", 0)
        tick_buf  = saved.get("tick_buf", [])
        tick_buf.append(wall_mid)
        if len(tick_buf) == self.DOWNSAMPLE:
            if len(prices) < self.ROLLING_WINDOW:
                prices.append(sum(tick_buf) / self.DOWNSAMPLE)
            else:
                prices[ptr] = sum(tick_buf) / self.DOWNSAMPLE
            ptr = (ptr + 1) % self.ROLLING_WINDOW
            tick_buf = []
        true_value = sum(prices) / len(prices) if prices else wall_mid
        saved["prices"]   = prices
        saved["ptr"]      = ptr
        saved["tick_buf"] = tick_buf

        diff = fair - true_value
        if diff > self.ARB_THRESH:
            fair -= (diff - self.ARB_THRESH)
        elif diff < -self.ARB_THRESH:
            fair -= (diff + self.ARB_THRESH)

        # --- INVENTORY SKEW ---
        # We previously used a cubic skew which was too "flat" in the middle. 
        # A strong linear skew actively shifts our pricing down as we get long, and up as we get short.
        # We use a divisor of 40. At max position (200), skew is -5 ticks.
        # This provides plenty of room to build up an inventory before we drastically panic our quotes.
        skew = -(position / self.SKEW_DIVISOR)

        skewed_fair = fair + skew
        
        # A static edge creates a tighter spread to ensure we are actually capturing trade volume.
        # Combined with the linear skew, we remain perfectly safe while actively market making.
        our_bid = math.floor(skewed_fair) - self.EDGE
        our_ask = math.ceil(skewed_fair)  + self.EDGE

        # --- PURE MAKER LOGIC ---
        # Allow our quotes to improve the spread, but strictly cap them to not cross the existing market.
        # We never take liquidity (pay the spread); we only provide it (earn the spread).
        our_bid = min(our_bid, best_ask - 1)
        our_ask = max(our_ask, best_bid + 1)
        
        if our_bid >= our_ask:
            our_bid = our_ask - 1

        # Shrink quote size as we approach the limit 
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

# VELVETFRUIT_EXTRACT: Pure MM (no true value anchor)
#
# Velvetfruit drifts slowly (+6, +20, +28 per day — inconsistent)
# and mean-reverts around the drift (AC1=-0.16). We don't try to
# predict direction — just MM around the market mid.
#
# Like Hydrogel, uses jump detection and inventory skew for defense.
# NO EMA smoothing because a genuine drift would leave us lagging behind
# and executing losing orders.

class VelvetfruitStrategy(Strategy):
    WALL_VOL     = 10       # min size to count as a wall
    SMOOTH_A     = 0.15     # EMA alpha for fair value smoothing
    ROLLING_WINDOW = 40_000
    DOWNSAMPLE   = 4         # shorter window than Hydrogel — Velvetfruit drifts faster, so we track tighter (circular buffer — O(1) writes)
    ARB_THRESH   = 3        # Tighter threshold given Velvetfruit's lower std dev (15 vs 32)
    SKEW_DIVISOR = 50       # Higher divisor = weaker skew, allows more volume buildup
    EDGE         = 1        # Min edge from skewed fair value
    MM_SIZE      = 20       # base quote size

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
        depth = state.order_depths[self.product]
        position = state.position.get(self.product, 0)

        if not depth.buy_orders or not depth.sell_orders:
            return [], saved

        best_bid = max(depth.buy_orders)
        best_ask = min(depth.sell_orders)

        # --- WALL MID ---
        wall_mid = self._get_wall_mid(depth)

        # --- SMOOTHED FAIR VALUE (EMA via traderdata) ---
        fair = saved.get("fair", wall_mid)
        fair = self.SMOOTH_A * wall_mid + (1 - self.SMOOTH_A) * fair
        saved["fair"] = fair

        # --- DYNAMIC TRUE VALUE ANCHOR (Robust Mean Reversion) ---
        # Absorbs the slow linear drift of Velvetfruit. 
        # Initializes at the first tick's mid-price so it perfectly traces the day's baseline.
        prices    = saved.get("prices", [])
        ptr       = saved.get("ptr", 0)
        tick_buf  = saved.get("tick_buf", [])
        tick_buf.append(wall_mid)
        if len(tick_buf) == self.DOWNSAMPLE:
            if len(prices) < self.ROLLING_WINDOW:
                prices.append(sum(tick_buf) / self.DOWNSAMPLE)
            else:
                prices[ptr] = sum(tick_buf) / self.DOWNSAMPLE
            ptr = (ptr + 1) % self.ROLLING_WINDOW
            tick_buf = []
        true_value = sum(prices) / len(prices) if prices else wall_mid
        saved["prices"]   = prices
        saved["ptr"]      = ptr
        saved["tick_buf"] = tick_buf

        diff = fair - true_value
        if diff > self.ARB_THRESH:
            fair -= (diff - self.ARB_THRESH)
        elif diff < -self.ARB_THRESH:
            fair -= (diff + self.ARB_THRESH)

        # --- INVENTORY SKEW ---
        # A strong linear skew actively shifts our pricing down as we get long, and up as we get short.
        skew = -(position / self.SKEW_DIVISOR)

        skewed_fair = fair + skew
        
        # A static edge creates a tighter spread to ensure we are actually capturing trade volume.
        our_bid = math.floor(skewed_fair) - self.EDGE
        our_ask = math.ceil(skewed_fair)  + self.EDGE

        # --- PURE MAKER LOGIC ---
        our_bid = min(our_bid, best_ask - 1)
        our_ask = max(our_ask, best_bid + 1)
        
        if our_bid >= our_ask:
            our_bid = our_ask - 1

        # Shrink quote size as we approach the limit 
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


class VEV_BS_Strategy(Strategy):
    CURRENT_DAY  = 3            # Set this for Round 3 submission
    TOTAL_LIFE   = 8            # Total option life
    MIN_EDGE     = 1.5          # Minimum mispricing edge
    SKEW_DIVISOR = 40           # Inventory skew divisor
    EMA_ALPHA    = 0.1          # Sluggish tracking of IV to capture fast deviations

    def __init__(self, strike: int):
        super().__init__(f"VEV_{strike}", position_limit=300)
        self.strike = strike

    def _get_T(self, timestamp: int) -> float:
        tte_days = (self.TOTAL_LIFE - self.CURRENT_DAY) - timestamp / 1_000_000
        return max(1e-8, tte_days / self.TOTAL_LIFE)

    def get_orders(self, state: TradingState, saved: dict) -> Tuple[List[Order], dict]:
        vf_depth = state.order_depths.get("VELVETFRUIT_EXTRACT")
        if not vf_depth or not vf_depth.buy_orders or not vf_depth.sell_orders:
            return [], saved
        S = (max(vf_depth.buy_orders) + min(vf_depth.sell_orders)) / 2

        depth = state.order_depths.get(self.product)
        if not depth or not depth.buy_orders or not depth.sell_orders:
            return [], saved

        best_bid = max(depth.buy_orders.keys())
        best_ask = min(depth.sell_orders.keys())
        market_mid = (best_bid + best_ask) / 2
        position = state.position.get(self.product, 0)
        T = self._get_T(state.timestamp)

        # 1. DYNAMIC IV ESTIMATION (The "Fitted Parabola" Tracking)
        live_iv = implied_vol(market_mid, S, self.strike, T)
        ema_iv = saved.get("ema_iv")
        
        if live_iv is not None:
            if ema_iv is None:
                ema_iv = live_iv
            else:
                ema_iv = self.EMA_ALPHA * live_iv + (1 - self.EMA_ALPHA) * ema_iv
                
        if ema_iv is None:
            ema_iv = 0.04 # Fallback
            
        saved["ema_iv"] = ema_iv

        # 2. SCALPING EDGE + INVENTORY SKEW
        fair_base = bs_call(S, self.strike, T, ema_iv)
        skew = -(position / self.SKEW_DIVISOR)
        fair = fair_base + skew

        buy_cap = self.position_limit - position
        sell_cap = self.position_limit + position
        orders: List[Order] = []

        # Taker sweeps
        for ask in sorted(depth.sell_orders.keys()):
            if ask < fair - self.MIN_EDGE and buy_cap > 0:
                qty = min(buy_cap, abs(depth.sell_orders[ask]))
                orders.append(self.order(ask, qty))
                buy_cap -= qty

        for bid in sorted(depth.buy_orders.keys(), reverse=True):
            if bid > fair + self.MIN_EDGE and sell_cap > 0:
                qty = min(sell_cap, depth.buy_orders[bid])
                orders.append(self.order(bid, -qty))
                sell_cap -= qty

        # Maker quotes
        our_bid = min(best_bid + 1, math.floor(fair - self.MIN_EDGE))
        our_ask = max(best_ask - 1, math.ceil(fair + self.MIN_EDGE))
        if our_bid >= our_ask:
            our_bid = our_ask - 1

        if our_bid >= 1 and buy_cap > 0:
            orders.append(self.order(our_bid, buy_cap))
        if sell_cap > 0:
            orders.append(self.order(our_ask, -sell_cap))

        return orders, saved


class VEV_VolArb_Strategy(Strategy):
    SKEW_DIVISOR = 40           # Force the bot to buy back its shorts
    STATIC_BIAS  = 1.5          # Tick bias to force shorting/buying structure
    EDGE         = 1.0          # Spread edge

    def __init__(self, strike: int):
        super().__init__(f"VEV_{strike}", position_limit=300)
        self.strike = strike

    def get_orders(self, state: TradingState, saved: dict) -> Tuple[List[Order], dict]:
        depth = state.order_depths.get(self.product)
        if not depth or not depth.buy_orders or not depth.sell_orders:
            return [], saved

        position = state.position.get(self.product, 0)
        buy_cap = self.position_limit - position
        sell_cap = self.position_limit + position
        
        best_bid = max(depth.buy_orders.keys())
        best_ask = min(depth.sell_orders.keys())
        
        # Base fair is the pure micro-price
        fair = (best_bid + best_ask) / 2
        
        # Add Structural Bias
        if self.strike >= 5300:
            fair -= self.STATIC_BIAS # Force shorting structure
        elif self.strike <= 5200:
            fair += self.STATIC_BIAS # Force buying structure
            
        # Add Inventory Skew to enforce profit taking churn
        skew = -(position / self.SKEW_DIVISOR)
        skewed_fair = fair + skew
        
        our_bid = math.floor(skewed_fair) - self.EDGE
        our_ask = math.ceil(skewed_fair) + self.EDGE
        
        our_bid = min(our_bid, best_ask - 1)
        our_ask = max(our_ask, best_bid + 1)
        if our_bid >= our_ask:
            our_bid = our_ask - 1
            
        orders: List[Order] = []
        if our_bid >= 1 and buy_cap > 0:
            orders.append(self.order(our_bid, buy_cap))
        if sell_cap > 0:
            orders.append(self.order(our_ask, -sell_cap))

        return orders, saved


# registry to handle matching products to strategies
STRATEGIES: Dict[str, Strategy] = {
    # "INTARIAN_PEPPER_ROOT": IntarianPepperRootStrategy("INTARIAN_PEPPER_ROOT", position_limit = 80),
    # "ASH_COATED_OSMIUM": OsmiumStrategy("ASH_COATED_OSMIUM", position_limit = 80),
    "HYDROGEL_PACK": HydrogelStrategy(),
    "VELVETFRUIT_EXTRACT": VelvetfruitStrategy(),

    # --- OPTIONS: VOLATILITY ARBITRAGE (SMIRK SNIPER) ---
    # "VEV_4000": VEV_VolArb_Strategy(4000),
    # "VEV_4500": VEV_VolArb_Strategy(4500),
    # "VEV_5000": VEV_VolArb_Strategy(5000),
    # "VEV_5100": VEV_VolArb_Strategy(5100),
    # "VEV_5200": VEV_VolArb_Strategy(5200),
    # "VEV_5300": VEV_VolArb_Strategy(5300),
    # "VEV_5400": VEV_VolArb_Strategy(5400),
    # "VEV_5500": VEV_VolArb_Strategy(5500),
    # "VEV_6000": VEV_VolArb_Strategy(6000),
    # "VEV_6500": VEV_VolArb_Strategy(6500),

    # --- OPTIONS: BLACK-SCHOLES MAKER (DYNAMIC IV) ---
    "VEV_4000": VEV_BS_Strategy(4000),
    "VEV_4500": VEV_BS_Strategy(4500),
    "VEV_5000": VEV_BS_Strategy(5000),
    "VEV_5100": VEV_BS_Strategy(5100),
    "VEV_5200": VEV_BS_Strategy(5200),
    "VEV_5300": VEV_BS_Strategy(5300),
    "VEV_5400": VEV_BS_Strategy(5400),
    "VEV_5500": VEV_BS_Strategy(5500),
    "VEV_6000": VEV_BS_Strategy(6000),
    "VEV_6500": VEV_BS_Strategy(6500),
}


# trader class for simulation to interact with algo
class Trader:

    def run(self, state: TradingState):
        # load persisted state from previous tick
        try:
            saved = json.loads(state.traderData)
        except:
            saved = {}

        result = {}

        for product, strategy in STRATEGIES.items():
            if product not in state.order_depths:
                continue

            # each strategy gets its own slice of saved state
            product_saved = saved.get(product, {})
            orders, product_saved = strategy.get_orders(state, product_saved)
            result[product] = orders
            saved[product] = product_saved

        # serialize state back for next tick
        trader_data = json.dumps(saved)
        return result, 0, trader_data