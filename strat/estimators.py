"""Estimators plugged into portopt.

``SignalEstimator`` : score -> selection (hysteresis, capped rotations) -> risk budgeting by sleeve
                      -> weight bounds -> turnover rules (no-trade band, turnover cap).
``FixedWeights`` / ``BandedSubset`` : benchmarks that only trade when a line drifts out of a band.

Turnover rules need the weights currently held. ``TurnoverBacktester`` writes them into
``estimator.prev_weights`` before each fit; with a plain portopt ``Backtester`` the estimator
simply behaves as if it started from cash at every rebalancing.
"""

from __future__ import annotations

import math
from typing import Mapping

import numpy as np
import pandas as pd
from portopt import BaseEstimator, RiskParity, risk_contributions

from .config import StratConfig
from .signals import signal_table

EPS = 1e-6


def apply_bounds(w: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Project weights on {sum = 1, lo <= w <= hi}, keeping free coordinates proportional.

    Only one side is clamped per pass (the one with the larger violation): clamping caps and
    floors together can freeze every coordinate before the weights sum to 1.
    """
    w = np.asarray(w, dtype=float)
    w = w / w.sum()
    free = np.ones(len(w), dtype=bool)
    for _ in range(2 * len(w) + 2):
        if not free.any():
            break
        room = 1.0 - w[~free].sum()
        tot = w[free].sum()
        w[free] = w[free] / tot * room if tot > 0 else room / free.sum()
        over, under = free & (w > hi + 1e-12), free & (w < lo - 1e-12)
        if not (over.any() or under.any()):
            break
        if (w[over] - hi[over]).sum() >= (lo[under] - w[under]).sum():
            w[over] = hi[over]
            free &= ~over
        else:
            w[under] = lo[under]
            free &= ~under
    res = 1.0 - w.sum()                              # leftover goes where there is slack
    if abs(res) > 1e-10:
        slack = (hi - w) if res > 0 else (w - lo)
        if slack.sum() > 0:
            w = w + res * slack / slack.sum()
    return w


class SignalEstimator(BaseEstimator):
    def __init__(self, groups: Mapping[str, str], cfg: StratConfig | None = None, label: str = "Strat"):
        cfg = cfg or StratConfig()
        super().__init__(cfg.cov_method)
        self.groups = dict(groups)
        self.cfg = cfg
        self.label = label
        self.prev_weights: pd.Series | None = None
        self.signals_: pd.DataFrame | None = None
        self.ideal_weights_: pd.Series | None = None
        self.selected_: list[str] | None = None
        self.exits_: list[str] | None = None
        self.entries_: list[str] | None = None

    @property
    def name(self) -> str:
        return self.label

    # ------------------------------------------------------------------ selection
    def _select(self, sig: pd.DataFrame, held: set[str], max_exits: float):
        """Return (selected, executed exits, entries, fills). ``max_exits`` caps the rotations."""
        cfg = self.cfg
        per_group, wanted_out = {}, []
        for g, quota in cfg.quotas.items():
            s = sig[sig["group"] == g]
            keep_rank = max(quota, math.ceil(cfg.exit_quantile * len(s)))
            held_g = [a for a in s.index if a in held]
            for a in held_g:
                if not (s.at[a, "trend"] and s.at[a, "rank"] <= keep_rank):
                    # broken trend first, then the worst scores
                    wanted_out.append((bool(s.at[a, "trend"]), float(s.at[a, "score"]), a))
            cands = [a for a in s.sort_values("score", ascending=False).index
                     if s.at[a, "trend"] and a not in held]
            per_group[g] = (quota, held_g, cands)

        wanted_out.sort()
        n_out = len(wanted_out) if math.isinf(max_exits) else int(max_exits)
        exits = [a for _, _, a in wanted_out[:n_out]]

        selected, entries, fills = [], [], []
        for g, (quota, held_g, cands) in per_group.items():
            stay = [a for a in held_g if a not in exits][:quota]
            new = cands[: quota - len(stay)]
            n_replaced = sum(1 for a in exits if a in held_g)
            fills += new[n_replaced:]            # slots that were empty (cash) and get invested
            entries += new
            selected += stay + new
        return selected, exits, entries, fills

    # ------------------------------------------------------------------ allocation
    def _ideal_weights(self, returns: pd.DataFrame, selected: list[str]) -> pd.Series:
        """Risk budgeting by sleeve, weight bounds, empty slots parked in cash."""
        cfg = self.cfg
        w = pd.Series(0.0, index=returns.columns)
        share = len(selected) / cfg.n_lines          # invested fraction, the rest goes to cash
        if selected:
            grp = pd.Series({a: self.groups[a] for a in selected})
            count = grp.value_counts()
            budgets = {a: cfg.risk_budgets[grp[a]] / count[grp[a]] for a in selected}
            hi = np.array([cfg.w_max_by_group.get(grp[a], cfg.w_max) for a in selected]) / share
            lo = np.array([cfg.w_min_by_group.get(grp[a], cfg.w_min) for a in selected]) / share
            hi = np.maximum(hi, lo)
            if len(selected) == 1:
                w[selected] = share
            else:
                rp = RiskParity(cov_method=cfg.cov_method, budgets=budgets).fit(returns[selected])
                raw, cov = rp.weights_.to_numpy(), rp.cov_
                capped = (grp == cfg.capped_group).to_numpy()
                present = sum(cfg.risk_budgets[g] for g in count.index)
                cap = cfg.risk_budgets.get(cfg.capped_group, 1.0) / present

                def bounded(theta):                  # shrink the capped sleeve by theta, then bound
                    v = np.where(capped, theta * raw, raw)
                    return apply_bounds(v / v.sum(), lo, hi)

                def capped_share(v):
                    return float(risk_contributions(v, cov)[capped].sum())

                best = bounded(1.0)
                # weight bounds distort the budgets: re-impose the sleeve's risk budget as a ceiling
                if capped.any() and not capped.all() and capped_share(best) > cap + 0.005:
                    a, b = 0.0, 1.0
                    for _ in range(40):
                        mid = 0.5 * (a + b)
                        a, b = (a, mid) if capped_share(bounded(mid)) > cap else (mid, b)
                    best = bounded(a)
                w[selected] = best * share
        if share < 1 - EPS:
            if cfg.cash not in w.index:
                w[selected] /= w[selected].sum() if selected else 1.0
            else:
                w[cfg.cash] += 1.0 - share
        return w

    # ------------------------------------------------------------------ turnover rules
    def _execute(self, ideal: pd.Series, prev: pd.Series, exits, entries, fills):
        """Turn ideal weights into traded targets. Returns (target, turnover counted against the cap)."""
        cfg = self.cfg
        cash = cfg.cash
        kept = [a for a in prev.index[prev > EPS] if a not in exits and a != cash and ideal[a] > EPS]
        gaps = (ideal[kept] - prev[kept]).abs().sort_values()

        def build(resize):
            target = prev.copy()
            target[list(exits)] = 0.0
            frozen = [a for a in kept if a not in resize]
            active = list(entries) + list(resize)
            cash_w = ideal.get(cash, 0.0)
            if cash in prev.index and abs(cash_w - prev[cash]) < cfg.no_trade_band and not fills:
                cash_w = prev[cash]
            room = 1.0 - prev[frozen].sum() - cash_w
            if active and room > EPS:
                target[active] = ideal[active] / ideal[active].sum() * room
                if cash in target.index:
                    target[cash] = cash_w
            elif cash in target.index:               # nothing to buy: freed capital waits in cash
                target[cash] = max(1.0 - prev[frozen].sum(), 0.0)
            else:
                return ideal.copy()
            return target

        def turnover(target):
            # deploying cash into empty slots is not churn: it is left out of the cap
            return max((target - prev).abs().sum() - 2.0 * target[list(fills)].sum(), 0.0)

        def within_bounds(target, resize):
            active = set(entries) | set(resize)
            for a in list(entries) + kept:           # frozen lines may sit one band outside
                tol = 1e-6 if a in active else cfg.no_trade_band
                hi = cfg.w_max_by_group.get(self.groups[a], cfg.w_max)
                lo = cfg.w_min_by_group.get(self.groups[a], cfg.w_min)
                if not (lo - tol <= target[a] <= hi + tol):
                    return False
                # a traded line must land on its ideal weight: freed capital is never
                # dumped on the few lines that happen to be traded
                if a in active and abs(target[a] - ideal[a]) > cfg.no_trade_band:
                    return False
            return True

        order = list(gaps.index)                     # kept lines, smallest gap first
        j = sum(gaps >= cfg.no_trade_band)           # resize the j largest gaps
        target = build(order[len(order) - j:])
        while j > 0 and turnover(target) > cfg.max_turnover + 1e-9:
            j -= 1                                   # give up the smallest resize first
            target = build(order[len(order) - j:])
        while j < len(order) and not within_bounds(target, order[len(order) - j:]):
            j += 1                                   # weight bounds win over the turnover cap
            target = build(order[len(order) - j:])
        return target, turnover(target)

    # ------------------------------------------------------------------ portopt hook
    def _compute_weights(self, returns: pd.DataFrame) -> np.ndarray:
        cfg = self.cfg
        cols = returns.columns
        sig = signal_table(returns, self.groups, cfg)
        self.signals_ = sig
        prev = pd.Series(0.0, index=cols)
        if self.prev_weights is not None:
            prev = self.prev_weights.reindex(cols).fillna(0.0).clip(lower=0.0)
            if prev.sum() > EPS:
                prev = prev / prev.sum()             # weight of lines no longer eligible is released
        held = set(prev.index[prev > EPS]) - {cfg.cash}

        if prev.sum() <= EPS:                        # initial build from cash: no turnover rule
            selected, exits, entries, _ = self._select(sig, set(), math.inf)
            target = self._ideal_weights(returns, selected)
            ideal = target
        else:
            for k in range(cfg.max_swaps, -1, -1):
                selected, exits, entries, fills = self._select(sig, held, k)
                ideal = self._ideal_weights(returns, selected)
                target, tau = self._execute(ideal, prev, exits, entries, fills)
                if tau <= cfg.max_turnover + 1e-9 or len(exits) <= 1:
                    break                            # the cap is soft for the last rotation
        self.selected_, self.exits_, self.entries_ = selected, list(exits), list(entries)
        self.ideal_weights_ = ideal
        target = target.clip(lower=0.0)
        target[target < EPS] = 0.0
        return target.to_numpy()


class FixedWeights(BaseEstimator):
    """Constant-mix benchmark. Trades only when a line drifts more than ``band`` from its target."""

    def __init__(self, weights: Mapping[str, float], band: float = 0.05, label: str = "Fixed"):
        super().__init__("sample")
        self.target = dict(weights)
        self.band = band
        self.label = label
        self.prev_weights: pd.Series | None = None

    @property
    def name(self) -> str:
        return self.label

    def _compute_weights(self, returns: pd.DataFrame) -> np.ndarray:
        w = pd.Series(self.target, dtype=float).reindex(returns.columns).fillna(0.0)
        if w.sum() <= 0:
            return np.full(len(w), 1.0 / len(w))
        w /= w.sum()
        return _banded(w, self.prev_weights, self.band)


class BandedSubset(BaseEstimator):
    """Any portopt estimator restricted to a list of assets, with the same drift band."""

    def __init__(self, inner: BaseEstimator, assets: list[str], band: float = 0.05, label: str | None = None):
        super().__init__("sample")
        self.inner, self.assets, self.band = inner, list(assets), band
        self.label = label or inner.name
        self.prev_weights: pd.Series | None = None

    @property
    def name(self) -> str:
        return self.label

    def _compute_weights(self, returns: pd.DataFrame) -> np.ndarray:
        live = [a for a in self.assets if a in returns.columns]
        w = pd.Series(0.0, index=returns.columns)
        if not live:
            return np.full(len(w), 1.0 / len(w))
        w[live] = self.inner.fit(returns[live]).weights_
        return _banded(w, self.prev_weights, self.band)


def _banded(w: pd.Series, prev: pd.Series | None, band: float) -> np.ndarray:
    if prev is not None:
        prev = prev.reindex(w.index).fillna(0.0)
        if prev.sum() > EPS and (w - prev / prev.sum()).abs().max() < band:
            return (prev / prev.sum()).to_numpy()
    return w.to_numpy()
