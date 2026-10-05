"""Alpha research: test each alpha alone, per sleeve, with a train / test split.

    python -m strat.research [--split 2017-12-31] [--out outputs/research]

Two measures per (alpha, sleeve), on the backtest's rebalancing grid and holding horizon:
  * rank IC          Spearman(alpha_t, r_{t -> t+h}) inside the sleeve
  * top-n excess     mean forward return of the n best names minus the sleeve average,
                     n = the sleeve's quota. This is what a long-only top-n portfolio earns.
Selection is made on the train period only; the test period is read once, afterwards.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .alphas import ALPHAS, compute_alphas
from .config import StratConfig
from .universe import GROUPS


def build_panel(returns: pd.DataFrame, cfg: StratConfig, groups=GROUPS) -> pd.DataFrame:
    """Long table: one row per (date, asset) with every alpha and the forward return."""
    r = returns.sort_index()
    h = cfg.hold
    frames = []
    for t in range(cfg.in_sample, len(r) - h + 1, h):
        win = r.iloc[t - cfg.in_sample:t]
        win = win.loc[:, win.notna().all().to_numpy()]
        a = compute_alphas(win, cfg.market)
        a["fwd"] = (1.0 + r.iloc[t:t + h][win.columns].fillna(0.0)).prod() - 1.0
        a["group"] = [groups.get(c, "other") for c in a.index]
        a["date"] = r.index[t]
        frames.append(a.rename_axis("asset").reset_index())
    return pd.concat(frames, ignore_index=True)


def _tstat(x: pd.Series) -> float:
    x = x.dropna()
    return float(x.mean() / x.std(ddof=1) * np.sqrt(len(x))) if len(x) > 2 and x.std(ddof=1) > 0 else np.nan


def per_date_stats(panel: pd.DataFrame, quotas: dict, min_names: int = 5) -> pd.DataFrame:
    """IC and top-n excess return for each (date, sleeve, alpha)."""
    rows = []
    for (date, g), d in panel.groupby(["date", "group"]):
        if g not in quotas or len(d) < min_names:
            continue
        n = quotas[g]
        for name in ALPHAS:
            x = d[[name, "fwd"]].dropna()
            if len(x) < min_names:
                continue
            ic = stats.spearmanr(x[name], x["fwd"]).statistic
            top = x.nlargest(n, name)["fwd"].mean() - x["fwd"].mean()
            rows.append((date, g, name, ic, top))
    return pd.DataFrame(rows, columns=["date", "group", "alpha", "ic", "top_excess"])


def summarize(pds: pd.DataFrame, split: str, periods_per_year: float) -> pd.DataFrame:
    split = pd.Timestamp(split)
    out = []
    for (g, name), d in pds.groupby(["group", "alpha"]):
        tr, te = d[d["date"] <= split], d[d["date"] > split]
        out.append({
            "group": g, "alpha": name,
            "ic_train": tr["ic"].mean(), "t_ic_train": _tstat(tr["ic"]),
            "ic_test": te["ic"].mean(), "t_ic_test": _tstat(te["ic"]),
            "top_train": tr["top_excess"].mean() * periods_per_year, "t_top_train": _tstat(tr["top_excess"]),
            "top_test": te["top_excess"].mean() * periods_per_year, "t_top_test": _tstat(te["top_excess"]),
            "n_train": len(tr), "n_test": len(te),
        })
    s = pd.DataFrame(out)
    # Benjamini-Hochberg on the train IC p-values, across every (alpha, sleeve) tested
    p = 2 * stats.t.sf(s["t_ic_train"].abs(), s["n_train"] - 1)
    order = np.argsort(p)
    q = np.empty(len(p))
    q[order] = np.minimum.accumulate((p[order] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    s["p_train"], s["q_bh_train"] = p, np.minimum(q, 1.0)
    return s.sort_values(["group", "t_ic_train"], ascending=[True, False]).reset_index(drop=True)


def alpha_correlations(panel: pd.DataFrame, group: str) -> pd.DataFrame:
    """Average cross-sectional rank correlation between alphas inside one sleeve."""
    d = panel[panel["group"] == group]
    mats = [x[list(ALPHAS)].rank().corr() for _, x in d.groupby("date") if len(x) >= 5]
    return sum(mats) / len(mats)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prices", default=None, help="CSV of prices (default: newest file in data_cache/)")
    ap.add_argument("--split", default="2017-12-31")
    ap.add_argument("--out", default="outputs/research")
    a = ap.parse_args(argv)
    from portopt import clean_prices, compute_returns
    path = Path(a.prices) if a.prices else max(Path("data_cache").glob("prices_*.csv"), key=lambda f: f.stat().st_size)
    returns = compute_returns(clean_prices(pd.read_csv(path, index_col=0, parse_dates=True)), "simple")
    cfg = StratConfig()
    panel = build_panel(returns, cfg)
    pds = per_date_stats(panel, cfg.quotas)
    summ = summarize(pds, a.split, 252 / cfg.hold)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    summ.to_csv(out / "alpha_summary.csv", index=False)
    pds.to_csv(out / "alpha_by_date.csv", index=False)
    alpha_correlations(panel, "stocks").to_csv(out / "alpha_corr_stocks.csv")
    with pd.option_context("display.width", 220, "display.max_columns", 30, "display.max_rows", 200,
                           "display.float_format", "{:,.3f}".format):
        print(summ.drop(columns=["p_train", "n_train", "n_test"]))
    return summ


if __name__ == "__main__":
    main()
