import numpy as np
import pandas as pd
import pytest

import quantcore as qc
from quantcore import metrics as m


@pytest.fixture
def prices():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2015-01-01", periods=1500)
    r = rng.normal(0.0004, 0.01, size=(len(idx), 3))
    return pd.DataFrame(100 * np.cumprod(1 + r, axis=0), idx, ["A", "B", "C"])


def test_buy_and_hold_matches_asset(prices):
    w = pd.DataFrame({"A": 1.0}, index=prices.index)
    res = qc.backtest(w, prices, qc.CostModel(0, 0), lag=0)
    expected = prices["A"].pct_change().fillna(0)
    assert np.allclose(res.returns.values, expected.values)
    assert res.turnover.iloc[1:].abs().max() < 1e-12  # no rebalancing needed


def test_lag_prevents_lookahead(prices):
    # A signal built from day-t's own return is only exploitable with lag=0 (look-ahead).
    same_day = prices.pct_change()
    w = np.sign(same_day).fillna(0) / 3
    cheat = qc.backtest(w, prices, qc.CostModel(0, 0), lag=0).returns.sum()
    honest = qc.backtest(w, prices, qc.CostModel(0, 0), lag=1).returns.sum()
    assert cheat > 5 and abs(honest) < cheat / 5


def test_costs_reduce_returns(prices):
    w = qc.strategies.sma_crossover(prices, 10, 50)
    free = qc.backtest(w, prices, qc.CostModel(0, 0))
    paid = qc.backtest(w, prices, qc.CostModel(5, 5))
    assert paid.returns.sum() < free.returns.sum()
    assert np.isclose((free.returns - paid.returns).sum(), paid.costs.sum())


def test_metrics_known_values():
    r = pd.Series([0.01, -0.02, 0.03, -0.01], index=pd.bdate_range("2020-01-01", periods=4))
    assert np.isclose(m.max_drawdown(r), -0.02)
    assert m.hit_rate(r) == 0.5
    v, cv = m.var_cvar(pd.Series(np.linspace(-0.1, 0.1, 101)), 0.95)
    assert v == pytest.approx(-0.09) and cv <= v


def test_vol_target_hits_target(prices):
    w = qc.strategies.buy_and_hold(prices)
    vt = qc.sizing.vol_target(w, prices, target=0.15, lookback=60)
    res = qc.backtest(vt, prices, qc.CostModel(0, 0))
    assert m.ann_vol(res.returns.iloc[300:]) == pytest.approx(0.15, rel=0.2)


def test_walk_forward_is_out_of_sample(prices):
    grid = qc.walkforward.param_grid(fast=[10, 20], slow=[50, 100])
    res, log = qc.walkforward.walk_forward(prices, qc.strategies.sma_crossover, grid, train=500, test=250)
    assert len(log) == 4  # 1000 OOS bars / 250
    assert res.returns.index[-1] == prices.index[-1]
    assert res.returns.index[0] == prices.index[500]
    assert res.returns.index.is_unique


def test_report_outputs(tmp_path, prices):
    res = qc.backtest(qc.strategies.buy_and_hold(prices), prices)
    p = qc.report.tearsheet({"bh": res.returns}, tmp_path / "t.png")
    assert p.exists() and p.stat().st_size > 10_000
    md = qc.report.to_markdown(qc.compare([res]))
    assert "| Sharpe |" in md
