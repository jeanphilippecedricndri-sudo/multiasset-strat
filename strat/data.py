"""Data access: Yahoo Finance through portopt (with a per-ticker fallback), or a synthetic universe."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
from portopt import clean_prices, compute_returns, download_prices


def load_universe_returns(tickers: list[str], start: str, end: str | None = None,
                          cache_dir: str | Path | None = "data_cache") -> pd.DataFrame:
    """Daily simple total returns (T x N). Tickers Yahoo cannot serve are dropped with a warning."""
    try:
        prices = download_prices(tickers, start=start, end=end, cache_dir=cache_dir)
    except ValueError:
        frames, missing = [], []
        for t in tickers:  # one bad ticker must not kill the run
            try:
                frames.append(download_prices(t, start=start, end=end, cache_dir=cache_dir))
            except Exception:
                missing.append(t)
        if not frames:
            raise
        if missing:
            warnings.warn(f"No data for {missing}: dropped from the universe.")
        prices = pd.concat(frames, axis=1).sort_index()
    prices = clean_prices(prices, ffill_limit=5)
    return compute_returns(prices, method="simple")


def simulate_universe(groups: Mapping[str, str], n_obs: int = 252 * 12, start: str = "2012-01-02",
                      seed: int = 7) -> pd.DataFrame:
    """Synthetic returns with the universe's tickers (offline runs and tests).

    One market factor, one factor per sleeve, fat-tailed idiosyncratic noise, slowly moving
    asset drifts (so that momentum carries some information) and staggered listings.
    """
    rng = np.random.default_rng(seed)
    names = list(groups)
    n = len(names)
    spec = {  # beta range, sleeve-factor vol, idio vol range, annual drift
        "stocks": ((0.8, 1.5), 0.05, (0.18, 0.35), 0.09),
        "etf": ((0.6, 1.2), 0.04, (0.03, 0.10), 0.07),
        "bonds": ((-0.25, 0.15), 0.05, (0.01, 0.04), 0.03),
        "commodities": ((0.0, 0.4), 0.14, (0.06, 0.15), 0.03),
        "cash": ((0.0, 0.0), 0.0, (0.008, 0.012), 0.015),
    }
    mkt = rng.standard_t(5, n_obs) * 0.16 / np.sqrt(252) / np.sqrt(5 / 3) + 0.06 / 252
    sleeves = sorted(set(groups.values()))
    fac = {g: rng.standard_normal(n_obs) * spec[g][1] / np.sqrt(252) for g in sleeves}
    block = 126
    x = np.empty((n_obs, n))
    for j, name in enumerate(names):
        g = groups[name]
        (b0, b1), _, (s0, s1), mu = spec[g]
        beta = 1.0 if name == "SPY" else rng.uniform(b0, b1)
        idio = 0.0 if name == "SPY" else rng.uniform(s0, s1)
        drift = np.repeat(rng.normal(mu, idio * 0.6, n_obs // block + 1), block)[:n_obs] / 252
        eps = rng.standard_t(4, n_obs) * idio / np.sqrt(252) / np.sqrt(2.0)
        x[:, j] = drift + beta * mkt + (0.0 if name == "SPY" else fac[g]) + eps
    out = pd.DataFrame(x, index=pd.bdate_range(start, periods=n_obs), columns=names)
    late = rng.choice([k for k in names if groups[k] == "stocks"], size=min(8, n // 10), replace=False)
    for name in late:  # late listings
        out.loc[out.index[: int(rng.integers(100, 700))], name] = np.nan
    return out
