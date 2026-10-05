"""All tunable parameters in one place."""

from __future__ import annotations

from dataclasses import dataclass, field


TREND = {"ma_cross_50_200": 1.0, "ma_ratio_200": 1.0, "mom_6_1": 1.0, "mom_12_1": 1.0}

# v2: chosen on 2009-2017 only (python -m strat.research), then checked once on 2018-2026.
SIGNALS_V2 = {
    "stocks": TREND,                       # best top-5 excess return in train, confirmed in test
    "etf": TREND,                          # the four best rank ICs in train
    "bonds": {"bollinger_20": 1.0},        # short-term reversal: only bond alpha with t > 2 in train
    "commodities": TREND,                  # nothing significant on 7 names: prior from the literature
}
# v1: the first version of the score (12-1 momentum + low volatility in every sleeve)
SIGNALS_V1 = {g: {"mom_12_1": 0.6, "lowvol_63": 0.4} for g in ("stocks", "etf", "bonds", "commodities")}


@dataclass(frozen=True)
class StratConfig:
    # ---- data / walk-forward grid
    start: str = "2008-01-01"
    in_sample: int = 252            # fit window (observations)
    hold: int = 21                  # holding period: 21 = monthly, 5 = weekly

    # ---- signals (all computed inside the in-sample window, so no look-ahead)
    signal_weights: dict = field(default_factory=lambda: dict(SIGNALS_V2))   # sleeve -> {alpha: weight}
    winsor: float = 0.01

    # ---- selection: number of lines per sleeve (10 in total)
    quotas: dict = field(default_factory=lambda: {"stocks": 5, "etf": 2, "bonds": 2, "commodities": 1})
    exit_quantile: float = 0.40     # hysteresis: a held line leaves only below the top 40% of its sleeve

    # ---- allocation: risk budget per sleeve (share of total portfolio risk)
    risk_budgets: dict = field(default_factory=lambda: {"stocks": 0.40, "etf": 0.24, "bonds": 0.24, "commodities": 0.12})
    cov_method: str = "ledoit_wolf"
    w_min: float = 0.05
    w_max: float = 0.20
    w_max_by_group: dict = field(default_factory=dict)   # e.g. {"stocks": 0.10}
    w_min_by_group: dict = field(default_factory=lambda: {"stocks": 0.03})
    capped_group: str = "stocks"    # sleeve whose risk budget is enforced as a ceiling

    # ---- turnover constraints
    max_swaps: int = 2              # held lines replaced per rebalancing
    no_trade_band: float = 0.02     # a kept line is not resized below this gap
    max_turnover: float = 0.40      # sum |w_target - w_held| per rebalancing (rotations + resizes)

    # ---- special tickers
    cash: str = "SHY"               # receives the budget of empty slots (long-only, fully invested)
    market: str = "SPY"             # single factor of the variance decomposition

    # ---- costs
    nav0: float = 10_000.0          # EUR, starting capital
    fixed_fee: float = 1.0          # EUR per order
    spread_bps: float = 5.0         # proportional cost on traded notional

    @property
    def n_lines(self) -> int:
        return int(sum(self.quotas.values()))
