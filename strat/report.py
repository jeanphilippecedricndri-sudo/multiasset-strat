"""Run report: portopt charts + strategy-specific tables, gathered in one HTML page."""

from __future__ import annotations

import html
from pathlib import Path
from typing import Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from portopt import plot, summary

from .risk import group_risk_contributions, variance_decomposition


def save_charts(rets: pd.DataFrame, bts: Mapping, strat: str, groups: Mapping[str, str], cfg,
                out_dir: Path) -> list[Path]:
    plot.use_style()
    out_dir.mkdir(parents=True, exist_ok=True)
    colors = plot.colors_for(rets.columns)
    summ = summary(rets, turnover={k: b.annualized_turnover() for k, b in bts.items()})
    paths = []

    def save(fig, name):
        p = out_dir / name
        fig.savefig(p)
        plt.close(fig)
        paths.append(p)

    save(plot.plot_wealth(rets, colors=colors).figure, "01_wealth.png")
    save(plot.plot_drawdowns(rets, colors=colors), "02_drawdowns.png")
    save(plot.plot_rolling(rets, "vol", 252, colors=colors).figure, "03_rolling_vol.png")
    save(plot.plot_rolling(rets, "sharpe", 252, colors=colors).figure, "04_rolling_sharpe.png")
    save(plot.plot_metrics(summ, colors=colors), "05_metrics.png")

    by_group = lambda df: df.T.groupby(lambda a: groups.get(a, "other")).sum().T
    avg_w = pd.DataFrame({k: by_group(b.weights_.iloc[1:]).mean() for k, b in bts.items()}).T.fillna(0.0)
    save(plot.plot_heatmap(avg_w, "Average capital allocation by sleeve",
                           "Mean target weight across rebalancings").figure, "06_capital_by_sleeve.png")
    avg_rc = pd.DataFrame({k: group_risk_contributions(b, groups, cfg.cov_method).iloc[1:].mean()
                           for k, b in bts.items()}).T.fillna(0.0)
    save(plot.plot_heatmap(avg_rc, "Average risk allocation by sleeve",
                           "Mean ex-ante risk contribution, in-sample covariance").figure, "07_risk_by_sleeve.png")

    bt = bts[strat]
    w = bt.weights_.loc[:, (bt.weights_ > 0).any()]
    w = w[w.mean().sort_values(ascending=False).index[:40]]
    save(plot.plot_weights_through_time(w, f"{strat}: weights through time (40 most held lines)").figure,
         "08_weights.png")

    vd = variance_decomposition(bt, groups, cfg.market)
    fig, ax = plt.subplots(figsize=(11, 3.6))
    parts = vd[["market", "other_factors", "specific"]].clip(lower=0)
    ax.stackplot(parts.index, parts.T.to_numpy(), labels=["Market beta", "Other factors", "Specific"],
                 colors=[plot.colors_for(["a", "b", "c"])[k] for k in ("a", "b", "c")], alpha=0.9)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_title(f"{strat}: ex-ante variance decomposition", loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False)
    save(fig, "09_variance_decomposition.png")

    fig, ax = plt.subplots(figsize=(11, 3.2))
    ax.bar(bt.orders_.index[1:], bt.orders_.iloc[1:], width=cfg.hold * 0.9, color=colors[strat])
    ax.set_title(f"{strat}: orders per rebalancing", loc="left")
    save(fig, "10_orders.png")
    return paths


CSS = """
:root{--bg:#0b0f14;--panel:#111821;--line:#1f2a37;--fg:#dbe4ee;--muted:#8696a8;--accent:#f2b544}
body{background:var(--bg);color:var(--fg);font-family:'Space Grotesk',system-ui,sans-serif;margin:0}
main{max-width:1100px;margin:0 auto;padding:32px 20px 64px}
h1{font-size:26px;margin:0 0 4px} h2{font-family:'Space Mono',monospace;font-size:12px;letter-spacing:.14em;
text-transform:uppercase;color:var(--accent);margin:36px 0 12px}
p,.meta{color:var(--muted)} .scroll{overflow-x:auto}
table{border-collapse:collapse;font-family:'Space Mono',monospace;font-size:12px;font-variant-numeric:tabular-nums}
th,td{padding:5px 12px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}
th:first-child,td:first-child{text-align:left} th{color:var(--muted);font-weight:400}
img{max-width:100%;display:block;margin:12px 0;border:1px solid var(--line)}
"""


def _table(df: pd.DataFrame, fmt=None) -> str:
    return '<div class="scroll">' + df.to_html(float_format=fmt or (lambda x: f"{x:,.3f}"), border=0, na_rep="") + "</div>"


def write_html(out_dir: Path, title: str, meta: str, sections: list[tuple[str, str]], charts: list[Path]) -> Path:
    body = [f"<h1>{html.escape(title)}</h1><div class='meta'>{html.escape(meta)}</div>"]
    for name, content in sections:
        body.append(f"<h2>{html.escape(name)}</h2>{content}")
    body.append("<h2>Graphiques</h2>" + "".join(f'<img src="charts/{p.name}" alt="{p.stem}">' for p in charts))
    page = ("<!doctype html><html lang='fr'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
            "<link rel='stylesheet' href='https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600&family=Space+Mono&display=swap'>"
            f"<title>{html.escape(title)}</title><style>{CSS}</style><main>{''.join(body)}</main></html>")
    path = out_dir / "report.html"
    path.write_text(page, encoding="utf-8")
    return path
