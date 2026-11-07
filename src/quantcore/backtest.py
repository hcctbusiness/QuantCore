"""Vectorized portfolio backtester.

Convention: ``weights`` are *target* weights decided using information available at
the close of day t. They are executed at the close of day t (or t+lag) and earn the
return from t+lag to t+lag+1. The default ``lag=1`` avoids look-ahead bias.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import metrics as _m


@dataclass
class CostModel:
    """Linear transaction-cost model expressed in basis points of traded notional."""

    commission_bps: float = 1.0
    slippage_bps: float = 2.0
    borrow_bps_annual: float = 0.0  # charged on gross short exposure

    @property
    def per_turn(self) -> float:
        return (self.commission_bps + self.slippage_bps) / 1e4


@dataclass
class BacktestResult:
    returns: pd.Series
    gross_returns: pd.Series
    weights: pd.DataFrame
    turnover: pd.Series
    costs: pd.Series
    name: str = "strategy"
    meta: dict = field(default_factory=dict)

    @property
    def equity(self) -> pd.Series:
        return (1.0 + self.returns).cumprod()

    def stats(self, periods: int = 252, benchmark: pd.Series | None = None) -> pd.Series:
        s = _m.summary(self.returns, periods=periods, benchmark=benchmark)
        s["Avg Daily Turnover"] = float(self.turnover.mean())
        s["Annual Cost Drag"] = float(self.costs.mean() * periods)
        return s.rename(self.name)


def backtest(
    weights: pd.DataFrame,
    prices: pd.DataFrame,
    costs: CostModel | None = None,
    lag: int = 1,
    name: str = "strategy",
) -> BacktestResult:
    """Run a vectorized backtest.

    Parameters
    ----------
    weights : target portfolio weights (index = dates, columns = assets).
    prices  : adjusted close prices with the same columns.
    costs   : :class:`CostModel`; defaults to 1bp commission + 2bp slippage.
    lag     : bars between signal and execution (>=1 recommended).
    """
    if lag < 0:
        raise ValueError("lag must be >= 0")
    costs = costs or CostModel()
    prices = prices.sort_index()
    cols = [c for c in weights.columns if c in prices.columns]
    if not cols:
        raise ValueError("weights and prices share no columns")
    rets = prices[cols].pct_change(fill_method=None).fillna(0.0)
    w = weights[cols].reindex(rets.index).ffill().fillna(0.0)
    held = w.shift(lag).fillna(0.0)

    gross = (held * rets).sum(axis=1)
    # held[t] earns rets[t]. By the time held[t] is set, the previous book held[t-1]
    # has drifted with rets[t-1]; turnover is measured against that drifted book.
    prev = held.shift(1).fillna(0.0)
    r_prev = rets.shift(1).fillna(0.0)
    denom = (1 + (prev * r_prev).sum(axis=1)).replace(0, np.nan)
    drifted = (prev * (1 + r_prev)).div(denom, axis=0).fillna(0.0)
    turnover = (held - drifted).abs().sum(axis=1)
    short = held.clip(upper=0).abs().sum(axis=1)
    cost = turnover * costs.per_turn + short * costs.borrow_bps_annual / 1e4 / 252
    net = gross - cost
    return BacktestResult(net, gross, held, turnover, cost, name=name)


def compare(results: list[BacktestResult], periods: int = 252) -> pd.DataFrame:
    """Side-by-side statistics table for several results."""
    return pd.concat([r.stats(periods) for r in results], axis=1)
