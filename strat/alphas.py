"""Alpha library: price-only signals taken from the reference books, one function per idea.

Every alpha maps the in-sample return window (T x N) to one value per asset, HIGHER = MORE
ATTRACTIVE. Nothing outside the window is used, so there is no look-ahead.

Sources
    [V]  Velu, Hardy, Nehren - Algorithmic Trading and Quantitative Strategies, ch. 5
    [FA] Tulchinsky et al.   - Finding Alphas, ch. 5, 12, 18, 21, 29
    [I]  Isichenko           - Quantitative Portfolio Management, sec. 2.2
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TD = 21  # trading days per month


def _prep(window: pd.DataFrame, market: str | None):
    r = window.to_numpy(dtype=float)
    logp = np.cumsum(np.log1p(r), axis=0)
    p = np.exp(logp)
    if market is not None and market in window.columns:
        m = window[market].to_numpy(dtype=float)
    else:
        m = r.mean(axis=1)                                   # equal-weighted market, [I] eq. 2.1
    beta = (r * m[:, None]).sum(axis=0) / (m**2).sum()       # no-intercept beta, [I] eq. 2.2
    resid = r - m[:, None] * beta                            # net-of-market daily returns
    return r, logp, p, resid


def _ret(logp, a: int, b: int | None = None):
    """Simple return between row a and row b of the log price index (b=None: last row)."""
    end = logp[-1] if b is None else logp[b]
    return np.expm1(end - logp[a])


# ------------------------------------------------------------------ momentum / trend
def mom_12_1(r, logp, p, e):
    """[V 5.9, FA 21] Jegadeesh-Titman: 12-month return skipping the last month."""
    return np.expm1(logp[len(r) - 1 - TD])


def mom_6_1(r, logp, p, e):
    """[FA 21] 6-month momentum (Moskowitz-Grinblatt horizon), skipping the last month."""
    return _ret(logp, len(r) - 1 - 6 * TD, len(r) - 1 - TD)


def tsmom_volscaled(r, logp, p, e):
    """[V eq. 5.39-5.41] Moskowitz-Ooi-Pedersen: 12-month return over ex-ante EWMA volatility."""
    lam = 1 - 1 / 61.0                                       # centre of mass 60 days
    w = lam ** np.arange(len(r) - 1, -1, -1)
    w /= w.sum()
    mu = w @ r
    vol = np.sqrt(w @ (r - mu) ** 2 * 261)
    return np.expm1(logp[-1]) / np.where(vol > 0, vol, np.nan)


def ma_ratio_200(r, logp, p, e):
    """[V 5.3.2 SMA rule, FA 18 strategy 1, FA 29] price over its 200-day simple moving average."""
    return p[-1] / p[-200:].mean(axis=0) - 1


def ma_cross_50_200(r, logp, p, e):
    """[FA 18 strategy 2, V moving-average oscillator] short average over long average."""
    return p[-50:].mean(axis=0) / p[-200:].mean(axis=0) - 1


def breakout_252(r, logp, p, e):
    """[FA 18 strategy 3, V running maximum] distance to the 1-year high (0 = at the high)."""
    return p[-1] / p.max(axis=0) - 1


def resid_mom_12_1(r, logp, p, e):
    """[I 2.2.1-2.2.2, V 5.9] momentum of the stock-specific (net-of-market) return, standardised."""
    x = e[: len(r) - TD]
    return x.sum(axis=0) / (x.std(axis=0, ddof=1) * np.sqrt(len(x)))


# ------------------------------------------------------------------ reversal
def rev_5d(r, logp, p, e):
    """[FA 5, Alpha1] minus the 5-day return."""
    return -_ret(logp, -6)


def rev_21d(r, logp, p, e):
    """[V 5.9 contrarian, FA 18 momentum-reversion] minus the 1-month return."""
    return -_ret(logp, -TD - 1)


def resid_rev_21d(r, logp, p, e):
    """[I eq. 2.3] mean reversion of the net-of-market return over one month."""
    return -e[-TD:].sum(axis=0)


def bollinger_20(r, logp, p, e):
    """[V 5.3.2 BWMA rule] minus the distance to the 20-day mean, in standard deviations."""
    sd = p[-20:].std(axis=0, ddof=1)
    return -(p[-1] - p[-20:].mean(axis=0)) / np.where(sd > 0, sd, np.nan)


# ------------------------------------------------------------------ statistical predictors
def lowvol_63(r, logp, p, e):
    """[I 2.2.4] minus the 3-month volatility ("used to be a reasonable predictor")."""
    return -r[-63:].std(axis=0, ddof=1)


def persistence(r, logp, p, e):
    """[I 2.2.4 Hurst, 2.2.2 signature plot] variance ratio of monthly to daily returns (>1 = trending)."""
    k = TD
    n = (len(r) // k) * k
    lr = np.log1p(r[-n:])
    block = lr.reshape(n // k, k, -1).sum(axis=1)
    return block.var(axis=0, ddof=1) / (k * lr.var(axis=0, ddof=1))


ALPHAS = {f.__name__: f for f in (
    mom_12_1, mom_6_1, tsmom_volscaled, ma_ratio_200, ma_cross_50_200, breakout_252, resid_mom_12_1,
    rev_5d, rev_21d, resid_rev_21d, bollinger_20, lowvol_63, persistence,
)}


def compute_alphas(window: pd.DataFrame, market: str | None = "SPY", names=None) -> pd.DataFrame:
    """Assets x alphas table for one in-sample window."""
    args = _prep(window, market)
    names = names or list(ALPHAS)
    with np.errstate(divide="ignore", invalid="ignore"):
        return pd.DataFrame({n: ALPHAS[n](*args) for n in names}, index=window.columns)
