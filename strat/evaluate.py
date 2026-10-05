"""Signal quality, judged separately from the portfolio: rank IC of the score against forward returns.

    IC_t = Spearman( S_t , r_{t -> t+h} )   computed inside each sleeve,   ICIR = mean(IC) / std(IC)
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .config import StratConfig
from .signals import signal_table


def information_coefficients(returns: pd.DataFrame, groups: Mapping[str, str], cfg: StratConfig,
                             horizon: int | None = None, min_names: int = 5) -> pd.DataFrame:
    """IC time series, one column per sleeve, on the backtest's rebalancing grid."""
    h = horizon or cfg.hold
    r = returns.sort_index()
    rows = {}
    for t in range(cfg.in_sample, len(r) - h, cfg.hold):
        win = r.iloc[t - cfg.in_sample:t]
        win = win.loc[:, win.notna().all().to_numpy()]
        sig = signal_table(win, groups, cfg)
        fwd = (1.0 + r.iloc[t:t + h][win.columns].fillna(0.0)).prod() - 1.0
        ic = {}
        for g, s in sig.groupby("group"):
            if len(s) >= min_names:
                ic[g] = spearmanr(s["score"], fwd[s.index]).statistic
        rows[r.index[t]] = ic
    return pd.DataFrame(rows).T


def ic_summary(ic: pd.DataFrame, periods_per_year: float) -> pd.DataFrame:
    n = ic.notna().sum()
    mean, sd = ic.mean(), ic.std(ddof=1)
    return pd.DataFrame({
        "IC moyen": mean,
        "ICIR": mean / sd,
        "ICIR annualisé": mean / sd * np.sqrt(periods_per_year),
        "t-stat": mean / sd * np.sqrt(n),
        "IC > 0": (ic > 0).mean(),
        "n": n,
    })
