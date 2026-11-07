"""End-to-end demo on real data: multi-asset ETFs, 2007-today.

Run:  python examples/demo.py
Outputs docs/tearsheet.png, docs/walkforward.png and docs/results.md
"""
from pathlib import Path

import quantcore as qc

DOCS = Path(__file__).resolve().parents[1] / "docs"
UNIVERSE = ["SPY", "EFA", "EEM", "TLT", "IEF", "GLD", "DBC", "VNQ"]

px = qc.data.prices(UNIVERSE + ["^GSPC"], start="2006-01-01").ffill()
bench_px = px[["SPY"]]
px = px[UNIVERSE].dropna()
costs = qc.CostModel(commission_bps=1, slippage_bps=2)

ew = qc.backtest(qc.sizing.rebalance(qc.strategies.buy_and_hold(px), "ME"), px, costs, name="Equal Weight")
tsmom_raw = qc.strategies.time_series_momentum(px, lookback=252)
tsmom = qc.backtest(qc.sizing.vol_target(tsmom_raw, px, target=0.10), px, costs, name="TSMOM (10% vol)")
sma = qc.backtest(qc.strategies.sma_crossover(px, 50, 200), px, costs, name="SMA 50/200")
spy = qc.backtest(qc.strategies.buy_and_hold(bench_px), bench_px, costs, name="SPY")

grid = qc.walkforward.param_grid(lookback=[63, 126, 189, 252])
wf_strat = lambda p, lookback: qc.sizing.vol_target(qc.strategies.time_series_momentum(p, lookback), p, 0.10)
wf, log = qc.walkforward.walk_forward(px, wf_strat, grid, train=756, test=252, costs=costs, name="TSMOM walk-forward")

start = wf.returns.index[0]
results = [tsmom, sma, ew, spy]
for r in results:
    r.returns = r.returns.loc[start:]
stats = qc.compare([wf] + results)
for r in [wf] + results:
    stats.loc["Avg Daily Turnover", r.name] = r.turnover.loc[start:].mean()

qc.report.tearsheet({r.name: r.returns for r in [tsmom, sma, ew, spy]}, DOCS / "tearsheet.png",
                    "QuantCore demo — multi-asset strategies vs SPY")
qc.report.tearsheet({wf.name: wf.returns, "SPY": spy.returns}, DOCS / "walkforward.png",
                    "Walk-forward TSMOM (parameters re-chosen yearly, out-of-sample only)")
md = f"Out-of-sample period: {start:%Y-%m-%d} to {wf.returns.index[-1]:%Y-%m-%d}\n\n" + qc.report.to_markdown(stats)
md += "\n\nWalk-forward parameter log (chosen in-sample, traded out-of-sample):\n\n| Test start | Test end | Lookback | In-sample Sharpe |\n|---|---|---:|---:|\n"
md += "\n".join(f"| {r.test_start:%Y-%m-%d} | {r.test_end:%Y-%m-%d} | {r.lookback} | {r.is_score:.2f} |" for r in log.itertuples())
(DOCS / "results.md").write_text(md)
print(md)
