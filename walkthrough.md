# `kevin_algo.py` — Walkthrough & Developer Guide

## Overview

`kevin_algo.py` is the main trading algorithm file submitted to the IMC Prosperity exchange simulator. Each timestep, the exchange calls `Trader.run(state)`, which dispatches to a per-product strategy that returns a list of `Order` objects.

The file is structured into four layers:

```
Strategy          ← base class (shared interface + helpers)
  └── EmeraldsStrategy   ← first live strategy (EMERALDS / Rainforest Resin)
  └── TomatoesStrategy   ← placeholder for TOMATOES
STRATEGIES        ← registry dict mapping product name → strategy instance
Trader            ← entry point called by the exchange every timestep
```

---

## How the Exchange Works (Context)

Each timestep the exchange provides a `TradingState` snapshot containing:

| Field | Type | Description |
|---|---|---|
| `order_depths` | `Dict[str, OrderDepth]` | Current order book per product |
| `position` | `Dict[str, int]` | Your current inventory per product |
| `own_trades` | `Dict[str, List[Trade]]` | Your fills from the last step |
| `market_trades` | `Dict[str, List[Trade]]` | Other participants' fills |
| `traderData` | `str` | Persistent string state you can write/read |

`OrderDepth` contains two dicts:
- `buy_orders: Dict[int, int]` — price → positive quantity (bids)
- `sell_orders: Dict[int, int]` — price → **negative** quantity (asks)

`Trader.run()` must return `(result, conversions, traderData)` where `result` is `Dict[str, List[Order]]`.

---

## `Strategy` — Base Class

```python
class Strategy:
    def __init__(self, product: str, position_limit: int): ...
    def get_orders(self, depth: OrderDepth, position: int) -> List[Order]: ...
    def order(self, price: int, qty: int) -> Order: ...
```

Every product strategy inherits from `Strategy`. The key contract:

- **`get_orders(depth, position)`** — implement this in subclasses. Receives the live order book and current inventory, returns orders for this timestep. Raises `NotImplementedError` if not overridden.
- **`self.order(price, qty)`** — convenience helper. Positive `qty` = buy, negative `qty` = sell. Automatically fills in `self.product` so subclasses don't repeat the symbol string.
- **`self.position_limit`** — stored on the instance; each subclass sets this in `super().__init__()`.

> **No `abc` module is used.** The `NotImplementedError` pattern gives the same developer-facing safety without requiring any additional imports.

---

## `EmeraldsStrategy` — Rainforest Resin

### Background

EMERALDS (called "Rainforest Resin" in the challenge) has a stable true fair value of **10,000**. Because the price rarely strays far, a simple edge-taking + passive quoting approach captures consistent profit without needing any price prediction.

### Parameters

Defined as class-level constants at the top of `EmeraldsStrategy` for easy tuning:

| Constant | Default | Meaning |
|---|---|---|
| `FAIR_VALUE` | `10000` | Known true price of EMERALDS |
| `SKEW_THRESHOLD` | `60` | Absolute position level at which we flatten inventory |
| `PASSIVE_SPREAD` | `3` | Fallback offset from fair value when the book is empty |
| `MAX_PASSIVE_QTY` | `25` | Maximum quantity per passive resting quote |

The position limit (`80`) is set in `__init__` via `super().__init__("EMERALDS", position_limit=80)`.

### Three-Step Logic (per timestep)

#### Step 1 — Take Favorable Resting Orders

Scan the live order book for any orders that cross fair value:

```
For each ask price (cheapest first):
    If ask < 10,000 → BUY (we gain edge immediately)

For each bid price (highest first):
    If bid > 10,000 → SELL (we gain edge immediately)
```

Quantity is capped by remaining position capacity (`position_limit ± current_position`). We reduce `buy_cap` / `sell_cap` as we fill, so we never breach the position limit across multiple fills in one step.

**Why sort?** We want to take the most favorable price first (cheapest asks, richest bids), maximizing edge per unit of position used.

#### Step 2 — Post Passive Quotes

After taking what's available, we post resting orders to capture edge from future market orders:

```
our_bid = min(best_existing_bid + 1, FAIR_VALUE - 1)
our_ask = max(best_existing_ask - 1, FAIR_VALUE + 1)
```

- **Overbidding** (`best_bid + 1`): we sit at the top of the bid queue, so incoming sell market orders hit us first.
- **Undercutting** (`best_ask - 1`): we sit at the bottom of the ask queue.
- **Hard constraints**: `our_bid` is capped at `FAIR_VALUE - 1` and `our_ask` is floored at `FAIR_VALUE + 1`. This ensures we always have positive edge on passive fills — we never accidentally post at or through fair value.
- If the book is empty, `best_bid` / `best_ask` fall back to `FAIR_VALUE ± PASSIVE_SPREAD`.

Quantity is capped at `MAX_PASSIVE_QTY` and the remaining position capacity after Step 1.

#### Step 3 — Flatten When Inventory Is Too Skewed

```
If |position| > SKEW_THRESHOLD:
    Post order for -position qty @ FAIR_VALUE
```

If inventory becomes too one-sided (e.g., we've bought a lot and the market isn't offering sell opportunities with edge), we flatten at exactly fair value. This is zero-edge but frees up position capacity for the next timestep's profitable opportunities. Trading at fair value is never a loss on EMERALDS because the true value is known.

---

## `KelpStrategy` — Applied to TOMATOES

### Background

Kelp (applied to TOMATOES in this codebase) differs from EMERALDS in one key way: **there is no known fixed fair value**. The true price follows a slow random walk, drifting slightly between timesteps. However, because the drift is small and unpredictable, the best estimate of the current true price is simply what the market is showing right now.

The optimal approach is therefore identical to EMERALDS in structure — take edge, post passives, flatten if skewed — with one substitution: instead of a constant `FAIR_VALUE`, we compute the **WallMid** from the live order book at each timestep and use that as the dynamic fair price.

### What Is WallMid?

**WallMid** is the midpoint between the best resting bid and the best resting ask:

```
WallMid = (best_bid + best_ask) / 2
```

For example, if the order book shows:
```
Bids:  5006 × 8,  5005 × 15
Asks:  5009 × 5,  5010 × 12
```
Then `WallMid = (5006 + 5009) / 2 = 5007.5`

WallMid can be a **half-integer** (e.g., 5007.5) when the bid/ask sum is odd. This is important because exchange orders must be placed at integer prices. A strict positive-edge constraint means:

- Buying at price `p` is profitable only if `p < WallMid`
- Selling at price `p` is profitable only if `p > WallMid`

### Integer Arithmetic — No Float Needed

Rather than computing `WallMid` as a float, the strategy stores `mid_2x = best_bid + best_ask` (which equals `2 × WallMid` exactly as an integer). All comparisons are done in integer space to avoid floating-point rounding issues:

| Formula | Meaning | Example (mid_2x=10015, so WallMid=5007.5) | Example (mid_2x=10014, so WallMid=5007.0) |
|---|---|---|---|
| `bid_limit = (mid_2x - 1) // 2` | Largest integer strictly below WallMid | 5007 | 5006 |
| `ask_limit = mid_2x // 2 + 1` | Smallest integer strictly above WallMid | 5008 | 5008 |
| `flatten_price = (mid_2x + 1) // 2` | Nearest integer to WallMid | 5008 | 5007 |

Note that when WallMid is a half-integer (odd `mid_2x`), `bid_limit` and `ask_limit` are separated by 1 tick with 0.5 edge each. When WallMid is a whole number (even `mid_2x`), they are separated by 2 ticks with 1.0 edge each — the algorithm naturally avoids quoting at zero-edge prices.

### Parameters

Defined as class-level constants in `KelpStrategy`:

| Constant | Default | Meaning |
|---|---|---|
| `SKEW_THRESHOLD` | `40` | Absolute position level at which we flatten |
| `PASSIVE_SPREAD` | `2` | Fallback half-spread from WallMid when book is empty |
| `MAX_PASSIVE_QTY` | `20` | Maximum quantity per passive resting quote |

The product name and position limit are passed at instantiation in the `STRATEGIES` registry, making `KelpStrategy` reusable across any Kelp-like asset:
```python
KelpStrategy("TOMATOES", position_limit=80)
```

### Three-Step Logic

**Step 1 — Take Favorable Resting Orders**

Same structure as EMERALDS, but using `bid_limit` / `ask_limit` instead of a fixed `FAIR_VALUE`:
```
For each ask (cheapest first): buy if ask <= bid_limit
For each bid (highest first): sell if bid >= ask_limit
```

**Step 2 — Post Passive Quotes**

Same overbid/undercut logic, clamped so quotes never cross WallMid:
```
our_bid = min(best_bid + 1, bid_limit)   # never bid at or above WallMid
our_ask = max(best_ask - 1, ask_limit)   # never ask at or below WallMid
```
A `fallback_bid` / `fallback_ask` computed from `flatten_price ± PASSIVE_SPREAD` is used if the book is so one-sided that the above formula would invert. A final `our_bid < our_ask` sanity check prevents crossed quotes from ever being submitted.

**Step 3 — Flatten If Skewed**

Identical to EMERALDS, but the flatten price is `flatten_price = round(WallMid)` rather than a constant. Because the true price is drifting, we flatten at whatever the current best estimate is — this is zero expected edge, not a loss.

---

## `TomatoesStrategy` — Placeholder

```python
class TomatoesStrategy(Strategy):
    def get_orders(self, depth, position) -> List[Order]:
        return []   # TODO
```

Returns no orders each timestep. Replace the body with your actual TOMATOES strategy. The position limit is set to `20` — adjust if the exchange enforces a different limit for this product.

---

## `STRATEGIES` — Registry

```python
STRATEGIES: Dict[str, Strategy] = {
    "EMERALDS": EmeraldsStrategy(),
    "TOMATOES": TomatoesStrategy(),
}
```

This is the single place where products are enabled or disabled. `Trader.run()` iterates this dict — if a product is not in `STRATEGIES`, it is silently ignored. If a product is in `STRATEGIES` but not in `state.order_depths` (i.e., not active this round), it is also skipped.

### Adding a New Product

1. Write a new subclass of `Strategy`:
```python
class KelpStrategy(Strategy):
    FAIR_VALUE = 2000  # example

    def __init__(self):
        super().__init__("KELP", position_limit=50)

    def get_orders(self, depth: OrderDepth, position: int) -> List[Order]:
        orders = []
        # ... your logic ...
        return orders
```

2. Register it:
```python
STRATEGIES = {
    "EMERALDS": EmeraldsStrategy(),
    "TOMATOES": TomatoesStrategy(),
    "KELP":     KelpStrategy(),    # ← one line
}
```

---

## `Trader` — Exchange Entry Point

```python
class Trader:
    def run(self, state: TradingState):
        result = {}
        for product, strategy in STRATEGIES.items():
            if product not in state.order_depths:
                continue
            depth    = state.order_depths[product]
            position = state.position.get(product, 0)
            result[product] = strategy.get_orders(depth, position)
        return result, 0, ""
```

- **`conversions`** is always `0` (not used yet).
- **`traderData`** is always `""` — currently no cross-timestep state is needed. If you want to persist data (e.g., a rolling mid-price history), serialize it to a JSON string here and parse it at the start of `run()` from `state.traderData`.

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| Class-level constants for parameters | Visible at a glance at the top of each strategy; easy to tune without reading logic |
| `buy_cap` / `sell_cap` tracking | Ensures Step 1 and Step 2 together never exceed the position limit in a single timestep |
| Sort order for taking | Cheapest asks / highest bids first maximizes edge per unit of position consumed |
| Hard clamp on passive quotes | Eliminates the risk of accidentally posting at or through fair value due to an unusual book state |
| Flatten at fair value, not market | Avoids crossing the spread unnecessarily; on a stable-priced asset this is cost-free |
| `NotImplementedError` instead of `abc` | Same safety guarantee with only standard IMC-allowed imports |
