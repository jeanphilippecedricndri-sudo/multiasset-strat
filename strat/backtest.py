"""Walk-forward backtester aware of the held portfolio and of fixed fees per order.

Same timeline and no-look-ahead rule as ``portopt.Backtester`` (weights computed on r[t-L : t),
applied from r[t]). Three additions:

* the drifted weights held just before the rebalancing are handed to the estimator
  (``estimator.prev_weights``), which makes turnover rules possible;
* cost paid at the rebalancing = spread_bps * turnover + fixed_fee * n_orders / NAV_t,
  with NAV_t = nav0 * net wealth, so a fixed fee weighs more on a small account;
* per-line daily P&L contributions are stored (``contributions_``) for the risk checks.
"""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
from portopt import Backtester


class TurnoverBacktester(Backtester):
    def __init__(self, returns, estimator, in_sample, out_of_sample, window="rolling",
                 transaction_cost_bps: float = 0.0, fixed_fee: float = 0.0, nav0: float = 10_000.0,
                 order_eps: float = 1e-4, min_assets: int = 1):
        super().__init__(returns, estimator, in_sample, out_of_sample, window=window,
                         transaction_cost_bps=transaction_cost_bps, drift=True, min_assets=min_assets)
        self.fixed_fee, self.nav0, self.order_eps = fixed_fee, nav0, order_eps
        self.orders_: pd.Series | None = None          # number of orders at each rebalancing
        self.costs_: pd.Series | None = None           # cost paid, as a fraction of NAV
        self.nav_: pd.Series | None = None
        self.contributions_: pd.DataFrame | None = None  # daily w_{i,d-1} * r_{i,d}
        self.final_weights_: pd.Series | None = None   # drifted weights after the last observation
        self.fits_: dict = {}                          # rebalancing date -> fitted estimator

    def run(self) -> pd.Series:
        r = self.returns
        cols = r.columns
        values = r.to_numpy(dtype=float)
        T, N = values.shape
        gross, net = np.full(T, np.nan), np.full(T, np.nan)
        contrib = np.zeros((T, N), dtype=np.float32)
        w_held = np.zeros(N)
        wealth = 1.0
        dates, targets, turnovers, orders, costs = [], [], [], [], []

        t = self.in_sample
        while t < T:
            start = 0 if self.window == "expanding" else t - self.in_sample
            window = r.iloc[start:t]
            eligible = window.columns[window.notna().all().to_numpy()]
            if len(eligible) >= self.min_assets:
                est = copy.deepcopy(self.estimator)
                if hasattr(est, "prev_weights"):
                    est.prev_weights = pd.Series(w_held, index=cols)
                est.fit(window[eligible])
                target = est.weights_.reindex(cols, fill_value=0.0).to_numpy()
                self.fits_[r.index[t]] = est
            else:
                target = w_held.copy()

            trade = np.abs(target - w_held)
            turnover = trade.sum()
            n_orders = int((trade > self.order_eps).sum())
            cost = self.tc * turnover + self.fixed_fee * n_orders / (self.nav0 * wealth)

            end = min(t + self.out_of_sample, T)
            oos = np.nan_to_num(values[t:end], nan=0.0)
            growth = np.cumprod(1.0 + oos, axis=0)
            cash = 1.0 - target.sum()
            v = growth @ target + cash
            v_prev = np.concatenate(([1.0], v[:-1]))
            g_prev = np.vstack((np.ones(N), growth[:-1]))
            rp = v / v_prev - 1.0
            contrib[t:end] = target * g_prev / v_prev[:, None] * oos
            gross[t:end] = rp
            net[t:end] = rp
            net[t] = (1.0 - cost) * (1.0 + rp[0]) - 1.0
            wealth *= float(np.prod(1.0 + net[t:end]))

            dates.append(r.index[t]); targets.append(target)
            turnovers.append(turnover); orders.append(n_orders); costs.append(cost)
            w_held = target * growth[-1] / v[-1]
            t = end

        idx = r.index[self.in_sample:]
        di = pd.DatetimeIndex(dates)
        self.gross_returns_ = pd.Series(gross[self.in_sample:], index=idx, name=self.name)
        self.returns_ = pd.Series(net[self.in_sample:], index=idx, name=self.name)
        self.weights_ = pd.DataFrame(targets, index=di, columns=cols)
        self.turnover_ = pd.Series(turnovers, index=di, name=self.name)
        self.orders_ = pd.Series(orders, index=di, name=self.name)
        self.costs_ = pd.Series(costs, index=di, name=self.name)
        self.nav_ = self.nav0 * (1.0 + self.returns_).cumprod()
        self.contributions_ = pd.DataFrame(contrib[self.in_sample:], index=idx, columns=cols)
        self.final_weights_ = pd.Series(w_held, index=cols, name=self.name)
        return self.returns_

    def annual_cost(self, periods_per_year: int = 252) -> float:
        """Average cost per year (fraction of NAV), excluding the initial build-up."""
        years = len(self.returns_) / periods_per_year
        return float(self.costs_.iloc[1:].sum() / years)

    def orders_per_rebalancing(self) -> float:
        return float(self.orders_.iloc[1:].mean())
