"""Composite score per sleeve, built from the alpha library (see alphas.py).

Each alpha is winsorised and z-scored inside its sleeve, then combined with the sleeve's own
weights (``cfg.signal_weights[sleeve]``). The trend filter P_t > SMA_200 stays an absolute
entry condition, not a score.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

from .alphas import compute_alphas
from .config import StratConfig


def _zscore(x: pd.Series, winsor: float) -> pd.Series:
    x = x.replace([np.inf, -np.inf], np.nan)
    if x.notna().sum() < 2:
        return pd.Series(0.0, index=x.index)
    x = x.clip(x.quantile(winsor), x.quantile(1 - winsor))
    sd = x.std(ddof=0)
    z = (x - x.mean()) / sd if sd > 0 else x * 0.0
    return z.fillna(0.0)                                   # missing alpha = neutral


def signal_table(window: pd.DataFrame, groups: Mapping[str, str], cfg: StratConfig) -> pd.DataFrame:
    """One row per asset: group, mom, vol, trend, composite score, rank inside the sleeve."""
    if window.shape[0] < 200:
        raise ValueError("window too short for the signals.")
    used = sorted({a for w in cfg.signal_weights.values() for a in w} | {"mom_12_1", "ma_ratio_200", "lowvol_63"})
    a = compute_alphas(window, cfg.market, names=used)
    t = pd.DataFrame({
        "group": [groups.get(c, "other") for c in a.index],
        "mom": a["mom_12_1"],
        "vol": -a["lowvol_63"] * np.sqrt(252),
        "trend": a["ma_ratio_200"] > 0,
    }, index=window.columns)
    score = pd.Series(0.0, index=t.index)
    for g, members in t.groupby("group").groups.items():
        weights = cfg.signal_weights.get(g, {})
        tot = sum(abs(w) for w in weights.values()) or 1.0
        for name, w in weights.items():
            score[members] += w / tot * _zscore(a.loc[members, name], cfg.winsor)
    t["score"] = score
    t["rank"] = t.groupby("group")["score"].rank(ascending=False, method="first").astype(int)
    return t
