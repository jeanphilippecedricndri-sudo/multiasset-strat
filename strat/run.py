"""End-to-end run:  python -m strat.run  [--synthetic] [--hold 21] [--init | --holdings holdings.csv]

Every run writes, in --out (default ./outputs):
    report.html            performance, risk checks, signal quality, charts
    next_weights.csv       target weights for the next holding period, with orders in EUR
    with --save: portfolio/holdings.csv (reloaded by default next time) and portfolio/history/
    summary.csv, risk_checks.csv, worst_losses.csv, ic.csv, weights_history.csv, charts/*.png
"""

from __future__ import annotations

import argparse
import dataclasses
import warnings
from pathlib import Path

import pandas as pd
from portopt import ERC, summary

from . import evaluate, forecast, report, risk
from .backtest import TurnoverBacktester
from .config import StratConfig
from .data import load_universe_returns, simulate_universe
from .estimators import BandedSubset, FixedWeights, SignalEstimator
from .universe import GROUPS, NAMES, all_tickers

CORE_ETF = ["SPY", "EFA", "EEM", "IEF", "TLT", "LQD", "GLD", "DBC"]


def build_estimators(cfg: StratConfig) -> dict:
    return {
        "Strat": SignalEstimator(GROUPS, cfg, label="Strat"),
        "60/40": FixedWeights({"SPY": 0.6, "IEF": 0.4}, band=0.05, label="60/40"),
        "ERC 8 ETF": BandedSubset(ERC(cov_method=cfg.cov_method), CORE_ETF, band=0.05, label="ERC 8 ETF"),
    }


def run(cfg: StratConfig, returns: pd.DataFrame, out_dir: Path, holdings: pd.Series | None = None,
        nav: float | None = None, verbose: bool = True, init: bool = False,
        save: bool = False, portfolio: str = "portfolio/holdings.csv", source: str | None = None) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    say = print if verbose else (lambda *a, **k: None)
    groups = {a: g for a, g in GROUPS.items() if a in returns.columns}
    ests = build_estimators(cfg)

    bts = {k: TurnoverBacktester(returns, e, cfg.in_sample, cfg.hold, transaction_cost_bps=cfg.spread_bps,
                                 fixed_fee=cfg.fixed_fee, nav0=cfg.nav0) for k, e in ests.items()}
    rets = pd.DataFrame({k: b.run() for k, b in bts.items()})
    bt = bts["Strat"]

    ppy = 252
    summ = summary(rets, turnover={k: b.annualized_turnover() for k, b in bts.items()})
    summ.loc["Orders / rebal"] = {k: b.orders_per_rebalancing() for k, b in bts.items()}
    summ.loc["Ann. cost"] = {k: b.annual_cost() for k, b in bts.items()}
    checks = risk.risk_checks(bt, groups, cfg)
    worst = risk.worst_line_losses(bt, 15)
    ic = evaluate.information_coefficients(returns, groups, cfg)
    ic_sum = evaluate.ic_summary(ic, ppy / cfg.hold)

    # weights for the next period: real holdings if given, otherwise the backtest's simulated book
    # weights for the next period, from: cash (--init), real holdings (--holdings), or by default
    # the backtest's simulated book
    simulated = holdings is None and not init
    held = None if init else (bt.final_weights_ if simulated else holdings)
    nxt = forecast.next_weights(returns, ests["Strat"], groups, cfg, held, nav)
    as_of = nxt.attrs["as_of"]
    source = source or ("construction initiale depuis le cash" if init else "portefeuille simulé du backtest" if simulated else "portefeuille réel fourni")

    summ.to_csv(out_dir / "summary.csv")
    checks.to_csv(out_dir / "risk_checks.csv")
    worst.to_csv(out_dir / "worst_losses.csv", index=False)
    ic.to_csv(out_dir / "ic.csv")
    nxt.to_csv(out_dir / "next_weights.csv")
    target = nxt.loc[nxt["w_target"] > 0, ["name", "w_target"]].rename(columns={"w_target": "weight"})
    if save:  # the portfolio file is only ever written on request, and every version is archived
        pf = Path(portfolio)
        (pf.parent / "history").mkdir(parents=True, exist_ok=True)
        target.to_csv(pf)
        nxt.to_csv(pf.parent / "history" / f"rebalancing_{as_of:%Y-%m-%d}.csv")
    w = bt.weights_
    w.loc[:, (w > 0).any()].to_csv(out_dir / "weights_history.csv")

    charts = report.save_charts(rets, bts, "Strat", groups, cfg, out_dir / "charts")
    pct = lambda x: f"{x:.2%}"
    show = nxt.drop(columns=["ucits_isin"]).copy()
    show["name"] = show["name"].str.slice(0, 38)
    for c in ("w_current", "w_target", "delta"):
        show[c] = show[c].map(pct)
    basis = source
    sections = [
        (f"Poids cibles pour la prochaine période (données au {as_of:%Y-%m-%d})",
         f"<p>Poids que la stratégie fixe aujourd'hui, à appliquer à la prochaine séance. Point de départ : {basis}. "
         f"{int((nxt['action'] != 'HOLD').sum())} ordres.</p>" + report._table(show, lambda x: f"{x:,.2f}")),
        ("Performance hors échantillon, nette de frais", report._table(summ)),
        ("Contrôles de risque", report._table(checks)),
        ("Pires pertes journalières par ligne", report._table(worst, lambda x: f"{x:.2%}")),
        ("Qualité du signal (IC de rang par poche)", report._table(ic_sum)),
        ("Paramètres", "<div class='scroll'><table>" + "".join(
            f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in dataclasses.asdict(cfg).items()) + "</table></div>"),
    ]
    meta = (f"{returns.index[0]:%Y-%m-%d} → {returns.index[-1]:%Y-%m-%d} · {returns.shape[1]} actifs · "
            f"rebalancement tous les {cfg.hold} jours · frais {cfg.fixed_fee:g} € par ordre + {cfg.spread_bps:g} bps")
    path = report.write_html(out_dir, "Strat multi-asset", meta, sections, charts)

    with pd.option_context("display.width", 200, "display.max_columns", 20, "display.float_format", "{:,.3f}".format):
        say(f"\n=== Performance OOS nette ({meta}) ===\n{summ}")
        say(f"\n=== Contrôles de risque ===\n{checks}")
        say(f"\n=== Qualité du signal ===\n{ic_sum}")
        say(f"\n=== Poids cibles prochaine période (au {as_of:%Y-%m-%d}, base : {basis}) ===\n{show}")
    say(f"\nRapport : {path.resolve()}")
    return {"returns": rets, "backtesters": bts, "summary": summ, "checks": checks, "ic": ic,
            "next_weights": nxt, "report": path}


def read_holdings(path: str) -> pd.Series:
    """CSV indexed by ticker with a ``weight`` or ``value`` column (EUR amounts are fine):
    normalised to sum to 1. Other columns, such as ``name``, are ignored."""
    df = pd.read_csv(path, index_col=0)
    col = next((c for c in ("value", "weight") if c in df.columns), None)
    s = (df[col] if col else df.select_dtypes("number").iloc[:, 0]).astype(float)
    unknown = [a for a in s.index if a not in GROUPS]
    if unknown:
        raise ValueError(f"{path}: tickers outside the universe {unknown}. Use the proxy tickers (USO, CIBR, ...).")
    return s / s.sum()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--synthetic", action="store_true", help="simulated data, no network")
    ap.add_argument("--start", default=None)
    ap.add_argument("--hold", type=int, default=None, help="holding period in days (21 monthly, 5 weekly)")
    ap.add_argument("--nav", type=float, default=None, help="capital in EUR")
    ap.add_argument("--holdings", default=None,
                    help="CSV of the lines currently held (default: portfolio/holdings.csv when it exists)")
    ap.add_argument("--save", action="store_true",
                    help="write the targets to portfolio/holdings.csv and archive the rebalancing in portfolio/history/")
    ap.add_argument("--init", action="store_true", help="initial build: target weights starting from cash")
    ap.add_argument("--max-stock-weight", type=float, default=None)
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--cache", default="data_cache")
    a = ap.parse_args(argv)

    over = {}
    if a.start: over["start"] = a.start
    if a.hold: over["hold"] = a.hold
    if a.nav: over["nav0"] = a.nav
    if a.max_stock_weight: over["w_max_by_group"] = {"stocks": a.max_stock_weight}
    cfg = dataclasses.replace(StratConfig(), **over)

    warnings.simplefilter("ignore", FutureWarning)
    if a.synthetic:
        returns = simulate_universe(GROUPS)
    else:
        # explicit end date: the price cache is keyed by it, so each day downloads fresh data once
        end = (pd.Timestamp.today().normalize() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        returns = load_universe_returns(all_tickers(), cfg.start, end=end, cache_dir=a.cache)
    default = Path("portfolio/holdings.csv")
    path = a.holdings or (str(default) if default.exists() and not a.init else None)
    holdings = read_holdings(path) if path and not a.init else None
    source = ("construction initiale depuis le cash" if a.init else
              f"portefeuille réel ({path})" if path else "portefeuille simulé du backtest")
    hist = sorted(Path("portfolio/history").glob("rebalancing_*.csv"))
    if hist and not a.init:  # where are we in the monthly cycle?
        last = pd.Timestamp(hist[-1].stem.split("_")[1])
        n = int((returns.index > last).sum())
        print(f"Dernier rebalancement enregistré : {last:%Y-%m-%d}, {n} séances écoulées "
              f"(cycle de {cfg.hold}).")
        if n < cfg.hold:
            print(f"ATTENTION : rebalancement prévu dans {cfg.hold - n} séances. "
                  "Les poids ci-dessous sont indicatifs, ne les enregistre pas avec --save.")
    return run(cfg, returns, Path(a.out), holdings, a.nav, init=a.init, save=a.save, source=source)


if __name__ == "__main__":
    main()
