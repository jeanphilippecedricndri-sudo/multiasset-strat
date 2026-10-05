"""Target weights for the next holding period.

These are not return forecasts: they are the weights the strategy would set today, using every
observation available, for the period that starts at the next session.
"""

from __future__ import annotations

import copy
from typing import Mapping

import pandas as pd

from .config import StratConfig
from .universe import NAMES, UCITS


def next_weights(returns: pd.DataFrame, estimator, groups: Mapping[str, str], cfg: StratConfig,
                 holdings: pd.Series | None = None, nav: float | None = None) -> pd.DataFrame:
    """Fit on the last ``in_sample`` observations and compare with the current holdings.

    holdings : weights currently held (index = tickers). None = start from cash.
    """
    nav = nav or cfg.nav0
    win = returns.sort_index().iloc[-cfg.in_sample:]
    win = win.loc[:, win.notna().all().to_numpy()]
    est = copy.deepcopy(estimator)
    held = pd.Series(0.0, index=win.columns)
    if holdings is not None:
        held = holdings.reindex(win.columns).fillna(0.0)
        est.prev_weights = held
    est.fit(win)
    target = est.weights_
    sig = est.signals_
    rows = target.index[(target > 0) | (held > 0)]
    out = pd.DataFrame({
        "name": [NAMES.get(a, a) for a in rows],
        "group": [groups.get(a, "other") for a in rows],
        "score": sig.loc[rows, "score"].round(2),
        "rank": sig.loc[rows, "rank"],
        "trend": sig.loc[rows, "trend"],
        "w_current": held[rows],
        "w_target": target[rows],
    }, index=rows)
    out["delta"] = (out["w_target"] - out["w_current"]).round(6) + 0.0
    out["order_eur"] = (out["delta"] * nav).round(0)
    out["action"] = out["delta"].map(lambda d: "BUY" if d > 1e-4 else ("SELL" if d < -1e-4 else "HOLD"))
    out["ucits_isin"] = [UCITS.get(a, "") for a in rows]
    out.index.name = "asset"
    out.attrs["as_of"] = win.index[-1]
    return out.sort_values(["w_target", "w_current"], ascending=False)
