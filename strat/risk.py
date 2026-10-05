"""Risk checks on a finished backtest.

Single-factor decomposition of the ex-ante variance at each rebalancing (market = cfg.market):

    r_i = alpha_i + beta_i r_m + eps_i
    sigma_p^2 = (sum_i w_i beta_i)^2 sigma_m^2  +  sum_i w_i^2 sigma_eps_i^2  +  cross terms
                \\______ market ______/           \\______ specific ______/     \\_ other factors _/

"specific" is the diversifiable part carried line by line; "other" is the residual covariance
between lines (rates, commodities, sectors), i.e. systematic risk that is not equity beta.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd
from portopt import estimate_cov, risk_contributions


def variance_decomposition(bt, groups: Mapping[str, str], market: str, stock_group: str = "stocks") -> pd.DataFrame:
    r = bt.returns
    pos = r.index.get_indexer(bt.weights_.index)
    rows = []
    for p, (date, w) in zip(pos, bt.weights_.iterrows()):
        live = list(w[w > 0].index)
        win = r.iloc[p - bt.in_sample:p]
        if not live or market not in win or win[market].isna().any():
            continue
        x = win[live].to_numpy()
        m = win[market].to_numpy()
        wv = w[live].to_numpy()
        cov = np.cov(x, rowvar=False, ddof=1).reshape(len(live), len(live))
        var_m = m.var(ddof=1)
        beta = np.array([np.cov(x[:, i], m, ddof=1)[0, 1] for i in range(len(live))]) / var_m
        resid = np.clip(np.diag(cov) - beta**2 * var_m, 0.0, None)
        var_p = float(wv @ cov @ wv)
        mkt = float((wv @ beta) ** 2 * var_m)
        spec = wv**2 * resid
        is_stock = np.array([groups.get(a) == stock_group for a in live])
        rows.append({
            "date": date,
            "vol_ex_ante": np.sqrt(var_p * 252),
            "beta": float(wv @ beta),
            "market": mkt / var_p,
            "specific": float(spec.sum()) / var_p,
            "specific_stocks": float(spec[is_stock].sum()) / var_p,
            "other_factors": 1.0 - (mkt + float(spec.sum())) / var_p,
        })
    return pd.DataFrame(rows).set_index("date")


def group_risk_contributions(bt, groups: Mapping[str, str], cov_method="ledoit_wolf") -> pd.DataFrame:
    """Ex-ante relative risk contribution of each sleeve at each rebalancing (rows sum to 1)."""
    r = bt.returns
    pos = r.index.get_indexer(bt.weights_.index)
    rows = {}
    for p, (date, w) in zip(pos, bt.weights_.iterrows()):
        live = list(w[w > 0].index)
        if len(live) < 2:
            continue
        rc = risk_contributions(w[live].to_numpy(), estimate_cov(r.iloc[p - bt.in_sample:p][live], cov_method))
        rows[date] = pd.Series(rc, index=live).groupby(lambda a: groups.get(a, "other")).sum()
    return pd.DataFrame(rows).T.fillna(0.0)


def worst_line_losses(bt, top: int = 15) -> pd.DataFrame:
    """Largest single-day losses caused by one line, in NAV points (contribution = weight x return)."""
    c = bt.contributions_
    s = c.stack()
    s = s[s < 0].nsmallest(top)
    out = pd.DataFrame({"contribution": s})
    out["asset_return"] = [bt.returns.at[d, a] for d, a in s.index]
    out["weight"] = out["contribution"] / out["asset_return"]
    out.index.names = ["date", "asset"]
    out = out.reset_index()
    from .universe import NAMES
    out.insert(2, "name", [NAMES.get(a, a) for a in out["asset"]])
    return out


def worst_loss_by_line(bt) -> pd.DataFrame:
    c = bt.contributions_
    held = c.columns[(c != 0).any()]
    return pd.DataFrame({"worst_day": c[held].min(), "date": c[held].idxmin(),
                         "days_held": (c[held] != 0).sum()}).sort_values("worst_day")


def risk_checks(bt, groups: Mapping[str, str], cfg) -> pd.DataFrame:
    """Do the bounds hold? One line per check: observed value, limit, verdict."""
    w = bt.weights_.iloc[1:]
    held = bt.weights_.copy()
    vd = variance_decomposition(bt, groups, cfg.market)
    grc = group_risk_contributions(bt, groups, cfg.cov_method)
    n_lines = (w.drop(columns=[cfg.cash], errors="ignore") > 0).sum(axis=1)
    worst = float(bt.contributions_.min().min())
    stock_cap = cfg.w_max_by_group.get("stocks", cfg.w_max)
    rows = [
        ("Lignes risquées, max", n_lines.max(), cfg.n_lines, n_lines.max() <= cfg.n_lines),
        ("Poids cible max", held.drop(columns=[cfg.cash], errors="ignore").max().max(),
         cfg.w_max + cfg.no_trade_band, None),
        ("Budget de risque actions, moyenne", grc.get("stocks", pd.Series(dtype=float)).mean(),
         cfg.risk_budgets["stocks"], None),
        ("Budget de risque actions, max", grc.get("stocks", pd.Series(dtype=float)).max(), np.nan, None),
        ("Part spécifique de la variance, moyenne", vd["specific"].mean(), np.nan, None),
        ("dont actions", vd["specific_stocks"].mean(), np.nan, None),
        ("Part marché de la variance, moyenne", vd["market"].mean(), np.nan, None),
        ("Pire perte journalière d'une ligne (pts de NAV)", worst, -stock_cap * 0.30, worst >= -stock_cap * 0.30),
        ("Turnover par rebalancement, médiane", bt.turnover_.iloc[1:].median(), cfg.max_turnover, None),
        ("Rebalancements au-dessus du plafond", (bt.turnover_.iloc[1:] > cfg.max_turnover + 1e-9).mean(), np.nan, None),
        ("Ordres par rebalancement, moyenne", bt.orders_.iloc[1:].mean(), np.nan, None),
        ("Coût annuel (fraction de NAV)", bt.annual_cost(), np.nan, None),
    ]
    out = pd.DataFrame(rows, columns=["check", "observed", "limit", "ok"]).set_index("check")
    out.loc["Poids cible max", "ok"] = out.loc["Poids cible max", "observed"] <= cfg.w_max + cfg.no_trade_band + 1e-6
    out["ok"] = out["ok"].map({True: "oui", False: "NON"}).fillna("")
    return out
