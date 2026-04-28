from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List, Dict, Tuple, Optional
import math
import json

# BLACK-SCHOLES HELPERS
def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)

def bs_delta(S: float, K: float, T: float, sigma: float) -> float:
    if T <= 1e-8 or sigma <= 0:
        return 1.0 if S > K else 0.0
    d1 = (math.log(S / K) + 0.5 * sigma**2 * T) / (sigma * math.sqrt(T))
    return norm_cdf(d1)

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
    ROLLING_WINDOW = 80_000
    DOWNSAMPLE   = 2         # shorter window than Hydrogel — Velvetfruit drifts faster, so we track tighter (circular buffer — O(1) writes)
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
        options_delta = saved.get("options_delta", 0.0)
        target_position = -options_delta
        skew = -((position - target_position) / self.SKEW_DIVISOR)

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
        # tte_days: days remaining until expiry (e.g. 5.0 at start of day 0 if TOTAL_LIFE-CURRENT_DAY=5)
        # Divide by 365 to convert to years for Black-Scholes (NOT by TOTAL_LIFE)
        tte_days = (self.TOTAL_LIFE - self.CURRENT_DAY) - timestamp / 1_000_000
        return max(1e-8, tte_days / 365.0)

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

class VEV_FittedSmileStrategy:
    """
    Multi-strategy options handler combining three complementary approaches:
    
    1. DEEP ITM MARKET-MAKING (K=4000, 4500):
       These have 16-21 tick spreads. Pure spread capture — post bids/asks
       inside the market spread with strong inventory skew. No BS needed.
    
    2. GAMMA SCALPING (K=5200, 5300 — ATM):
       Realized vol (0.41) >> implied vol (0.29). Maintain a long gamma
       position in ATM options, delta-hedge via VELVETFRUIT_EXTRACT.
       Profits from the vol premium captured through rebalancing.
    
    3. CROSS-STRIKE RELATIVE VALUE (all calibrated strikes):
       Compare live IV ratios between adjacent strikes against the
       calibrated table. Trade when a strike's IV is out of line.
    """

    ALL_STRIKES = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
    
    # Sub-strategy strike assignments
    DEEP_ITM_STRIKES = [4000, 4500]
    GAMMA_STRIKES    = [5200, 5300]
    RV_STRIKES       = [5000, 5100, 5200, 5300, 5400, 5500]
    
    TOTAL_LIFE = 8
    POS_LIMIT  = 300

    ROUND_START_DAY = 3

    # IV table — used for gamma scalping fair value and relative value baseline
    STRIKE_IV_TABLE = {4000: 0.857084699331472, 
                        4500: 0.49681761537507124, 
                        5000: 0.24231478021589378, 
                        5100: 0.238441413017432, 
                        5200: 0.2425860862856522, 
                        5300: 0.24613916196117092, 
                        5400: 0.23002932880839103, 
                        5500: 0.25002853180727824}

    # --- Deep ITM MM params ---
    DITM_SKEW_DIV  = 30    # inventory skew divisor
    DITM_MAX_POS   = 40    # conservative position cap
    DITM_MM_SIZE   = 8     # quote size per side
    DITM_EDGE      = 2     # ticks inside the spread

    # --- Gamma scalping params ---
    GAMMA_TARGET   = 20    # target long position per ATM strike
    GAMMA_REBAL    = 5     # rebalance when position drifts by this much
    GAMMA_EMA_ALPHA = 0.1  # IV tracking for gamma fair value
    GAMMA_WARMUP   = 30    # ticks before gamma trading starts

    # --- Relative value params ---
    RV_IV_THRESHOLD = 0.015  # min IV ratio deviation to trigger RV trade
    RV_MAX_POS      = 30     # max position per strike from RV
    RV_EMA_ALPHA    = 0.08   # IV tracking for RV

    def _get_T(self, timestamp: int, saved: dict) -> float:
        last_ts    = saved.get("last_ts", -1)
        day_offset = saved.get("day_offset", 0)
        if last_ts >= 0 and timestamp < last_ts:
            day_offset += 1
        saved["last_ts"]    = timestamp
        saved["day_offset"] = day_offset

        current_day = self.ROUND_START_DAY + day_offset
        tte_days    = (self.TOTAL_LIFE - current_day) - timestamp / 1_000_000
        return max(1e-8, tte_days / 365.0)

    def _get_S(self, state: TradingState) -> Optional[float]:
        vf = state.order_depths.get("VELVETFRUIT_EXTRACT")
        if not vf or not vf.buy_orders or not vf.sell_orders:
            return None
        bb  = max(vf.buy_orders)
        ba  = min(vf.sell_orders)
        bsz = vf.buy_orders[bb]
        asz = abs(vf.sell_orders[ba])
        if bsz + asz == 0:
            return None
        return (bb * asz + ba * bsz) / (bsz + asz)

    # ─── SUB-STRATEGY 1: Deep ITM Market-Making ───────────────────────
    def _deep_itm_mm(self, K, state, saved):
        """
        Pure spread capture on wide-spread deep ITM options.
        Fair value = mid price + inventory skew. No BS needed.
        """
        product = f"VEV_{K}"
        depth = state.order_depths.get(product)
        if not depth or not depth.buy_orders or not depth.sell_orders:
            return []

        bb = max(depth.buy_orders)
        ba = min(depth.sell_orders)
        mid = (bb + ba) / 2
        spread = ba - bb

        position = state.position.get(product, 0)
        buy_cap  = min(self.POS_LIMIT, self.DITM_MAX_POS) - position
        sell_cap = min(self.POS_LIMIT, self.DITM_MAX_POS) + position
        buy_cap  = max(0, buy_cap)
        sell_cap = max(0, sell_cap)

        # Inventory skew pushes fair value to encourage position reduction
        skew = -(position / self.DITM_SKEW_DIV)
        fair = mid + skew

        # Post inside the spread, but not crossing
        our_bid = max(bb + self.DITM_EDGE, math.floor(fair - 1))
        our_ask = min(ba - self.DITM_EDGE, math.ceil(fair + 1))

        # Don't cross the market
        our_bid = min(our_bid, ba - 1)
        our_ask = max(our_ask, bb + 1)

        if our_bid >= our_ask:
            our_bid = our_ask - 1

        # Scale size with spread width (wider spread = more edge = more size)
        mm_size = min(self.DITM_MM_SIZE, max(1, spread // 4))

        orders = []
        if buy_cap > 0 and our_bid >= 1:
            orders.append(Order(product, our_bid, min(mm_size, buy_cap)))
        if sell_cap > 0:
            orders.append(Order(product, our_ask, -min(mm_size, sell_cap)))
        return orders

    # ─── SUB-STRATEGY 2: Gamma Scalping ───────────────────────────────
    def _gamma_scalp(self, K, S, T, state, saved):
        """
        Maintain a long gamma position in ATM options. The delta-hedging
        on VELVETFRUIT_EXTRACT will capture the vol premium:
        realized vol (0.41) >> implied vol (0.29).
        
        We accumulate a target long position (GAMMA_TARGET) by buying
        when price is at or below fair, then rely on delta-hedging to
        generate the actual P&L.
        """
        product = f"VEV_{K}"
        depth = state.order_depths.get(product)
        if not depth or not depth.buy_orders or not depth.sell_orders:
            return []

        bb = max(depth.buy_orders)
        ba = min(depth.sell_orders)
        mid = (bb + ba) / 2
        position = state.position.get(product, 0)

        # Track live IV for fair value
        ema_key = f"gamma_iv_{K}"
        count_key = f"gamma_count_{K}"
        count = saved.get(count_key, 0)
        
        live_iv = implied_vol(mid, S, K, T)
        if live_iv is not None and 0.01 < live_iv < 2.0:
            if count == 0:
                saved[ema_key] = live_iv
            else:
                old = saved.get(ema_key, live_iv)
                saved[ema_key] = self.GAMMA_EMA_ALPHA * live_iv + (1 - self.GAMMA_EMA_ALPHA) * old
            count += 1
            saved[count_key] = count

        if count < self.GAMMA_WARMUP:
            return []

        iv = saved.get(ema_key, 0.25)
        fair = bs_call(S, K, T, iv)

        orders = []

        # Accumulate toward target long position
        deficit = self.GAMMA_TARGET - position
        if deficit > 0:
            # Buy at or below fair value — patient accumulation
            buy_price = min(bb + 1, math.floor(fair))
            buy_price = min(buy_price, ba - 1)  # don't cross
            buy_cap = min(self.POS_LIMIT, self.GAMMA_TARGET + 10) - position
            if buy_cap > 0 and buy_price >= 1:
                qty = min(deficit, buy_cap, 5)  # small increments
                orders.append(Order(product, buy_price, qty))

        # If we're over-accumulated, trim by selling above fair
        elif position > self.GAMMA_TARGET + self.GAMMA_REBAL:
            sell_price = max(ba - 1, math.ceil(fair + 1))
            sell_price = max(sell_price, bb + 1)
            excess = position - self.GAMMA_TARGET
            sell_cap = self.POS_LIMIT + position
            if sell_cap > 0:
                qty = min(excess, sell_cap, 5)
                orders.append(Order(product, sell_price, -qty))

        return orders

    # ─── SUB-STRATEGY 3: Cross-Strike Relative Value ──────────────────
    def _relative_value(self, S, T, state, saved):
        """
        Compare each strike's live IV ratio (vs ATM) against the calibrated
        table ratio. If a strike's IV is elevated relative to where it
        should be (per the smile shape), sell it; if depressed, buy it.
        """
        # Collect live IVs for all RV strikes
        live_ivs = {}
        for K in self.RV_STRIKES:
            product = f"VEV_{K}"
            depth = state.order_depths.get(product)
            if not depth or not depth.buy_orders or not depth.sell_orders:
                continue
            bb = max(depth.buy_orders)
            ba = min(depth.sell_orders)
            mid = (bb + ba) / 2
            
            # Track via EMA
            ema_key = f"rv_iv_{K}"
            count_key = f"rv_count_{K}"
            count = saved.get(count_key, 0)
            
            iv = implied_vol(mid, S, K, T)
            if iv is not None and 0.01 < iv < 2.0:
                if count == 0:
                    saved[ema_key] = iv
                else:
                    old = saved.get(ema_key, iv)
                    saved[ema_key] = self.RV_EMA_ALPHA * iv + (1 - self.RV_EMA_ALPHA) * old
                saved[count_key] = count + 1
            
            if saved.get(count_key, 0) >= 50:
                live_ivs[K] = saved.get(ema_key)

        if len(live_ivs) < 3:
            return {}

        # Compute ATM IV (average of most ATM strikes we have)
        atm_candidates = [K for K in [5200, 5300] if K in live_ivs]
        if not atm_candidates:
            return {}
        live_atm = sum(live_ivs[K] for K in atm_candidates) / len(atm_candidates)
        
        table_atm_candidates = [K for K in [5200, 5300] if K in self.STRIKE_IV_TABLE]
        table_atm = sum(self.STRIKE_IV_TABLE[K] for K in table_atm_candidates) / len(table_atm_candidates)

        result = {}
        for K in self.RV_STRIKES:
            if K not in live_ivs or K in atm_candidates:
                continue
            if K not in self.STRIKE_IV_TABLE:
                continue

            product = f"VEV_{K}"
            depth = state.order_depths.get(product)
            if not depth or not depth.buy_orders or not depth.sell_orders:
                continue

            # Compare IV ratios: live_iv/live_atm vs table_iv/table_atm
            live_ratio  = live_ivs[K] / live_atm
            table_ratio = self.STRIKE_IV_TABLE[K] / table_atm

            ratio_dev = live_ratio - table_ratio  # positive = strike IV relatively elevated

            if abs(ratio_dev) < self.RV_IV_THRESHOLD:
                continue

            bb = max(depth.buy_orders)
            ba = min(depth.sell_orders)
            position = state.position.get(product, 0)
            buy_cap  = min(self.POS_LIMIT, self.RV_MAX_POS) - position
            sell_cap = min(self.POS_LIMIT, self.RV_MAX_POS) + position
            buy_cap  = max(0, buy_cap)
            sell_cap = max(0, sell_cap)

            orders = []
            if ratio_dev > self.RV_IV_THRESHOLD and sell_cap > 0:
                # Strike IV relatively high → sell
                qty = min(sell_cap, depth.buy_orders.get(bb, 0), 5)
                if qty > 0:
                    orders.append(Order(product, bb, -qty))
            elif ratio_dev < -self.RV_IV_THRESHOLD and buy_cap > 0:
                # Strike IV relatively low → buy
                qty = min(buy_cap, abs(depth.sell_orders.get(ba, 0)), 5)
                if qty > 0:
                    orders.append(Order(product, ba, qty))

            if orders:
                result[product] = orders

        return result

    # ─── MAIN ENTRY POINT ─────────────────────────────────────────────
    def get_orders(self, state: TradingState, saved: dict):
        S = self._get_S(state)
        if S is None:
            return {}, saved

        T = self._get_T(state.timestamp, saved)
        result = {}

        # 1. Deep ITM market-making
        for K in self.DEEP_ITM_STRIKES:
            orders = self._deep_itm_mm(K, state, saved)
            if orders:
                result[f"VEV_{K}"] = orders

        # 2. Gamma scalping on ATM strikes
        for K in self.GAMMA_STRIKES:
            orders = self._gamma_scalp(K, S, T, state, saved)
            if orders:
                product = f"VEV_{K}"
                result[product] = result.get(product, []) + orders

        # 3. Cross-strike relative value
        rv_orders = self._relative_value(S, T, state, saved)
        for product, orders in rv_orders.items():
            result[product] = result.get(product, []) + orders

        # Calculate total options delta for hedging via VELVETFRUIT_EXTRACT
        total_options_delta = 0.0
        for K in self.ALL_STRIKES:
            product = f"VEV_{K}"
            pos = state.position.get(product, 0)
            if pos == 0:
                continue
            # Use tracked IV if available, otherwise table
            iv = (saved.get(f"gamma_iv_{K}") or
                  saved.get(f"rv_iv_{K}") or
                  self.STRIKE_IV_TABLE.get(K, 0.25))
            delta = bs_delta(S, K, T, iv)
            total_options_delta += pos * delta
        
        saved["options_delta"] = total_options_delta

        return result, saved

VEV_SMILE = VEV_FittedSmileStrategy()


# registry to handle matching products to strategies
STRATEGIES: Dict[str, Strategy] = {
    # "INTARIAN_PEPPER_ROOT": IntarianPepperRootStrategy("INTARIAN_PEPPER_ROOT", position_limit = 80),
    # "ASH_COATED_OSMIUM": OsmiumStrategy("ASH_COATED_OSMIUM", position_limit = 80),
    # "HYDROGEL_PACK": HydrogelStrategy(),
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
    # "VEV_4000": VEV_BS_Strategy(4000),
    # "VEV_4500": VEV_BS_Strategy(4500),
    # "VEV_5000": VEV_BS_Strategy(5000),
    # "VEV_5100": VEV_BS_Strategy(5100),
    # "VEV_5200": VEV_BS_Strategy(5200),
    # "VEV_5300": VEV_BS_Strategy(5300),
    # "VEV_5400": VEV_BS_Strategy(5400),
    # "VEV_5500": VEV_BS_Strategy(5500),
    # "VEV_6000": VEV_BS_Strategy(6000),
    # "VEV_6500": VEV_BS_Strategy(6500),
}


# trader class for simulation to interact with algo
# class Trader:

#     def run(self, state: TradingState):
#         # load persisted state from previous tick
#         try:
#             saved = json.loads(state.traderData)
#         except:
#             saved = {}

#         result = {}

#         for product, strategy in STRATEGIES.items():
#             if product not in state.order_depths:
#                 continue

#             # each strategy gets its own slice of saved state
#             product_saved = saved.get(product, {})
#             orders, product_saved = strategy.get_orders(state, product_saved)
#             result[product] = orders
#             saved[product] = product_saved

#         # serialize state back for next tick
#         trader_data = json.dumps(saved)
#         return result, 0, trader_data

class Trader:
    def run(self, state: TradingState):
        try:
            saved = json.loads(state.traderData)
        except:
            saved = {}

        result = {}

        # VEV smile handles all options + velvetfruit underlying
        vev_saved             = saved.get("VEV_SMILE", {})
        vev_orders, vev_saved = VEV_SMILE.get_orders(state, vev_saved)
        result.update(vev_orders)
        saved["VEV_SMILE"]    = vev_saved

        # pass options_delta to VELVETFRUIT_EXTRACT
        options_delta = vev_saved.get("options_delta", 0.0)

        # Remaining single-product strategies
        for product, strategy in STRATEGIES.items():
            if product not in state.order_depths:
                continue
            product_saved         = saved.get(product, {})
            if product == "VELVETFRUIT_EXTRACT":
                product_saved["options_delta"] = options_delta
            orders, product_saved = strategy.get_orders(state, product_saved)
            result[product]       = orders
            saved[product]        = product_saved

        return result, 0, json.dumps(saved)