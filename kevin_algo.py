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

def bs_fair(S, K, T, iv):
    if T <= 1e-8 or iv <= 0: return max(0.0, S - K)
    d1 = (math.log(S/K) + 0.5*iv*iv*T) / (iv*math.sqrt(T))
    d2 = d1 - iv*math.sqrt(T)
    return S*norm_cdf(d1) - K*norm_cdf(d2)

def bs_delta(S, K, T, iv):
    if T <= 1e-8 or iv <= 0: return 1.0 if S > K else 0.0
    d1 = (math.log(S/K) + 0.5*iv*iv*T) / (iv*math.sqrt(T))
    return norm_cdf(d1)

def fit_iv(mid, S, K, T):
    intrinsic = max(0.0, S - K)
    if mid - intrinsic < 0.5 or T < 1e-8: return None
    iv = math.sqrt(2*math.pi/T) * (mid - intrinsic) / S
    iv = max(0.01, min(2.0, iv))
    for _ in range(30):
        d1 = (math.log(S/K) + 0.5*iv*iv*T) / (iv*math.sqrt(T))
        d2 = d1 - iv*math.sqrt(T)
        p  = S*norm_cdf(d1) - K*norm_cdf(d2)
        v  = S*math.sqrt(T)*norm_pdf(d1)
        if v < 1e-10: break
        iv -= (p - mid) / v
        iv  = max(0.01, min(2.0, iv))
        if abs(p - mid) < 0.01: return iv
    return iv if 0.01 < iv < 2.0 else None


class VEV_Strategy(Strategy):
    TOTAL_LIFE  = 8
    CURRENT_DAY = 4
    DUMP_TS     = 85_000
    SKEW_DIV    = 40
    TAKER_EDGE  = 2.0
    MAKER_EDGE  = 1.0
    IV_ALPHA    = 0.15
    MR_WINDOW   = 30
    MR_ALPHA    = 0.05

    def __init__(self, strike):
        super().__init__(f"VEV_{strike}", position_limit=300)
        self.K = strike

    def _T(self, ts):
        return max(1e-6, ((self.TOTAL_LIFE - self.CURRENT_DAY) - ts/1_000_000) / self.TOTAL_LIFE)

    def _mlr_fair(self, S, T, iv, saved):
        # Rolling MLR: regress option mid on [BS(S,K,T,iv), S, iv] using
        # exponentially weighted covariance update (no numpy, no matrices).
        # State: means, variances, covariances of 3 features vs target.
        # Prediction: fair = w0*bs + w1*S_dev + w2*iv_dev + intercept
        # where deviations are from rolling means.
        bs = bs_fair(S, self.K, T, iv)
        feats = [bs, S, iv]

        fm  = saved.get("fm",  [bs, S, iv])
        fv  = saved.get("fv",  [1.0, 1.0, 0.001])
        fcv = saved.get("fcv", [1.0, 0.0, 0.0])
        tm  = saved.get("tm",  bs)
        a   = self.MR_ALPHA

        old_fm = fm[:]
        old_tm = tm

        for i in range(3):
            fm[i] = (1-a)*fm[i] + a*feats[i]
        tm = (1-a)*tm + a*bs

        for i in range(3):
            df = feats[i] - old_fm[i]
            dt = bs     - old_tm
            fv[i]  = (1-a)*fv[i]  + a*df*df
            fcv[i] = (1-a)*fcv[i] + a*df*dt

        saved["fm"]  = fm
        saved["fv"]  = fv
        saved["fcv"] = fcv
        saved["tm"]  = tm

        weights = [fcv[i] / max(fv[i], 1e-8) for i in range(3)]
        pred = sum(weights[i] * (feats[i] - fm[i]) for i in range(3)) + tm
        return pred, bs, weights

    def _mr_signal(self, fair, saved):
        # Rolling mean reversion: track residual of (market_mid - mlr_fair)
        # z-score the residual against its rolling std.
        # High positive z → market overpriced → short signal
        # High negative z → market underpriced → long signal
        buf = saved.get("mr_buf", [])
        buf.append(fair)
        if len(buf) > self.MR_WINDOW:
            buf = buf[-self.MR_WINDOW:]
        saved["mr_buf"] = buf
        if len(buf) < 5:
            return 0.0
        n    = len(buf)
        mean = sum(buf) / n
        var  = sum((x-mean)**2 for x in buf) / n
        std  = math.sqrt(var) if var > 1e-10 else 1e-5
        return (fair - mean) / std

    def get_orders(self, state, saved):
        vf = state.order_depths.get("VELVETFRUIT_EXTRACT")
        if not vf or not vf.buy_orders or not vf.sell_orders:
            return [], saved
        S = (max(vf.buy_orders) + min(vf.sell_orders)) / 2

        depth = state.order_depths.get(self.product)
        if not depth or not depth.buy_orders or not depth.sell_orders:
            return [], saved

        best_bid = max(depth.buy_orders)
        best_ask = min(depth.sell_orders)
        mid      = (best_bid + best_ask) / 2
        position = state.position.get(self.product, 0)
        ts       = state.timestamp
        T        = self._T(ts)
        buy_cap  = self.position_limit - position
        sell_cap = self.position_limit + position
        orders   = []

        iv = saved.get("iv", 0.20)
        live_iv = fit_iv(mid, S, self.K, T)
        if live_iv:
            iv = self.IV_ALPHA * live_iv + (1 - self.IV_ALPHA) * iv
        saved["iv"] = iv

        mlr_fair, bs, weights = self._mlr_fair(S, T, iv, saved)

        resid = mid - mlr_fair
        z     = self._mr_signal(resid, saved)

        delta = bs_delta(S, self.K, T, iv)
        taker_edge = max(self.TAKER_EDGE, self.TAKER_EDGE / (1 - delta + 0.15))
        taker_edge = min(taker_edge, 10.0)

        mr_bias = z * 1.5
        fair    = mlr_fair - mr_bias - (position / self.SKEW_DIV)

        if ts >= self.DUMP_TS:
            if position > 0:
                r = position
                for bid in sorted(depth.buy_orders, reverse=True):
                    if r <= 0: break
                    q = min(r, depth.buy_orders[bid])
                    orders.append(self.order(bid, -q))
                    r -= q
                if r > 0: orders.append(self.order(best_bid, -r))
            elif position < 0:
                r = -position
                for ask in sorted(depth.sell_orders):
                    if r <= 0: break
                    q = min(r, abs(depth.sell_orders[ask]))
                    orders.append(self.order(ask, q))
                    r -= q
                if r > 0: orders.append(self.order(best_ask, r))
            return orders, saved

        for ask in sorted(depth.sell_orders):
            if ask >= fair - taker_edge or buy_cap <= 0: break
            q = min(buy_cap, abs(depth.sell_orders[ask]))
            orders.append(self.order(ask, q))
            buy_cap -= q

        for bid in sorted(depth.buy_orders, reverse=True):
            if bid <= fair + taker_edge or sell_cap <= 0: break
            q = min(sell_cap, depth.buy_orders[bid])
            orders.append(self.order(bid, -q))
            sell_cap -= q

        maker_bid = math.floor(fair - self.MAKER_EDGE)
        maker_ask = math.ceil(fair  + self.MAKER_EDGE)
        maker_bid = min(maker_bid, best_ask - 1)
        maker_ask = max(maker_ask, best_bid + 1)
        if maker_bid >= maker_ask: maker_bid = maker_ask - 1
        if maker_bid < 1: maker_bid = 1

        if buy_cap > 0:  orders.append(self.order(maker_bid,  buy_cap))
        if sell_cap > 0: orders.append(self.order(maker_ask, -sell_cap))

        return orders, saved


STRATEGIES: Dict[str, Strategy] = {
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