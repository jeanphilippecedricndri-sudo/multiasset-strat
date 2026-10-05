import dataclasses

import numpy as np
import pandas as pd
import pytest

from strat import GROUPS, SignalEstimator, StratConfig, TurnoverBacktester, signal_table
from strat.data import simulate_universe
from strat.estimators import apply_bounds
from strat.risk import group_risk_contributions


@pytest.fixture(scope="module")
def R():
    return simulate_universe(GROUPS, n_obs=252 * 5, seed=3)


@pytest.fixture(scope="module")
def bt(R):
    cfg = StratConfig()
    b = TurnoverBacktester(R, SignalEstimator(GROUPS, cfg), cfg.in_sample, cfg.hold,
                           transaction_cost_bps=cfg.spread_bps, fixed_fee=cfg.fixed_fee, nav0=cfg.nav0)
    b.run()
    return b


def test_apply_bounds():
    w = apply_bounds(np.array([0.7, 0.2, 0.05, 0.05]), np.full(4, 0.1), np.full(4, 0.4))
    assert w.sum() == pytest.approx(1.0) and w.max() <= 0.4 + 1e-9 and w.min() >= 0.1 - 1e-9


def test_apply_bounds_caps_and_floors_together():
    # three lines above the cap, five below the floor, one in between: must still sum to 1
    raw = np.array([0.30, 0.30, 0.30, 0.04, 0.012, 0.012, 0.012, 0.012, 0.012])
    lo = np.array([0.0556] * 4 + [0.0333] * 5)
    hi = np.full(9, 0.2222)
    w = apply_bounds(raw, lo, hi)
    assert w.sum() == pytest.approx(1.0) and (w <= hi + 1e-9).all() and (w >= lo - 1e-9).all()


def test_momentum_is_12_1(R):
    cfg = StratConfig()
    win = R.iloc[:252].dropna(axis=1)
    sig = signal_table(win, GROUPS, cfg)
    a = win.columns[0]
    assert sig.at[a, "mom"] == pytest.approx((1 + win[a].iloc[: 252 - 21]).prod() - 1)


def test_fully_invested_long_only_ten_lines(bt):
    w = bt.weights_
    assert np.allclose(w.sum(axis=1), 1.0) and (w >= 0).all().all()
    assert (w.drop(columns=["SHY"]) > 0).sum(axis=1).max() <= 10


def test_initial_build_respects_bounds(bt):
    cfg = StratConfig()
    w0 = bt.weights_.iloc[0].drop("SHY")
    w0 = w0[w0 > 0]
    lo = np.array([cfg.w_min_by_group.get(GROUPS[a], cfg.w_min) for a in w0.index])
    assert (w0 <= cfg.w_max + 1e-6).all() and (w0.to_numpy() >= lo - 1e-6).all()


def test_weight_cap_holds_at_every_rebalancing(bt):
    cfg = StratConfig()
    assert bt.weights_.drop(columns=["SHY"]).max().max() <= cfg.w_max + cfg.no_trade_band + 1e-6


def test_rotations_capped(bt):
    cfg = StratConfig()
    held = bt.weights_.drop(columns=["SHY"]) > 0
    exits = (held.shift(1, fill_value=False) & ~held).sum(axis=1).iloc[1:]
    assert exits.max() <= cfg.max_swaps


def test_stock_risk_budget(bt):
    cfg = StratConfig()
    rc = group_risk_contributions(bt, GROUPS, cfg.cov_method)["stocks"]
    assert rc.iloc[0] <= cfg.risk_budgets["stocks"] + 0.01      # exact at the initial build
    assert rc.mean() <= cfg.risk_budgets["stocks"] + 0.05       # drift inside the no-trade band


def test_no_look_ahead(R):
    cfg = StratConfig()
    cut = 252 * 3
    R2 = R.copy()
    R2.iloc[cut:] = R2.iloc[cut:] * -1.0                         # corrupt the future
    kw = dict(transaction_cost_bps=cfg.spread_bps, fixed_fee=cfg.fixed_fee, nav0=cfg.nav0)
    a = TurnoverBacktester(R, SignalEstimator(GROUPS, cfg), cfg.in_sample, cfg.hold, **kw)
    b = TurnoverBacktester(R2, SignalEstimator(GROUPS, cfg), cfg.in_sample, cfg.hold, **kw)
    a.run(); b.run()
    before = a.weights_.index <= R.index[cut]
    pd.testing.assert_frame_equal(a.weights_[before], b.weights_[before])


def test_fixed_fee_accounting(R):
    cfg = dataclasses.replace(StratConfig(), spread_bps=0.0)
    b = TurnoverBacktester(R, SignalEstimator(GROUPS, cfg), cfg.in_sample, cfg.hold, fixed_fee=1.0, nav0=1000.0)
    b.run()
    assert b.costs_.iloc[0] == pytest.approx(b.orders_.iloc[0] / 1000.0)
    net, gross = (1 + b.returns_).prod(), (1 + b.gross_returns_).prod()
    assert net == pytest.approx(gross * np.prod(1 - b.costs_))


def test_alphas_use_only_the_window(R):
    from strat.alphas import ALPHAS, compute_alphas
    win = R.iloc[300:552].dropna(axis=1)
    a = compute_alphas(win)
    assert list(a.columns) == list(ALPHAS) and np.isfinite(a.to_numpy()).mean() > 0.99
    x = win.iloc[:, 0]
    assert a["rev_5d"].iloc[0] == pytest.approx(-((1 + x.iloc[-5:]).prod() - 1))
    assert a["ma_ratio_200"].iloc[0] == pytest.approx((1 + x).cumprod().iloc[-1] / (1 + x).cumprod().iloc[-200:].mean() - 1)
