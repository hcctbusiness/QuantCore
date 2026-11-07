"""Walk-forward (rolling out-of-sample) optimisation."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Callable, Iterable

import pandas as pd

from .backtest import BacktestResult, CostModel, backtest
from . import metrics as _m


@dataclass
class Fold:
    train: pd.DatetimeIndex
    test: pd.DatetimeIndex


def splits(index: pd.DatetimeIndex, train: int = 756, test: int = 126, anchored: bool = False) -> list[Fold]:
    """Generate sequential train/test folds measured in bars."""
    folds, start = [], 0
    while start + train < len(index):
        tr_start = 0 if anchored else start
        folds.append(Fold(index[tr_start : start + train], index[start + train : start + train + test]))
        start += test  # final fold may be shorter than ``test`` so the OOS run reaches the last bar
    return folds


def param_grid(**grid: Iterable) -> list[dict]:
    keys = list(grid)
    return [dict(zip(keys, v)) for v in product(*grid.values())]


def walk_forward(
    prices: pd.DataFrame,
    strategy: Callable[..., pd.DataFrame],
    grid: list[dict],
    train: int = 756,
    test: int = 126,
    objective: Callable[[pd.Series], float] = _m.sharpe,
    costs: CostModel | None = None,
    anchored: bool = False,
    name: str = "walk-forward",
) -> tuple[BacktestResult, pd.DataFrame]:
    """Pick the best parameters on each train window and trade them on the next test window.

    ``strategy(prices, **params)`` must return a weight DataFrame computed only from
    past data (it is evaluated on the full history then sliced, so it must be causal).
    Returns the stitched out-of-sample result and a log of chosen parameters per fold.
    """
    cache = {}

    def run(p: dict) -> BacktestResult:
        k = tuple(sorted(p.items()))
        if k not in cache:
            cache[k] = backtest(strategy(prices, **p), prices, costs)
        return cache[k]

    oos, log = [], []
    for f in splits(prices.index, train, test, anchored):
        scores = {i: objective(run(p).returns.loc[f.train]) for i, p in enumerate(grid)}
        best = max(scores, key=lambda i: (scores[i] if scores[i] == scores[i] else -1e9))
        res = run(grid[best])
        oos.append((res, f.test))
        log.append({"test_start": f.test[0], "test_end": f.test[-1], "is_score": scores[best], **grid[best]})

    if not oos:
        raise ValueError("not enough data for a single fold")
    pick = lambda attr: pd.concat([getattr(r, attr).loc[t] for r, t in oos])
    result = BacktestResult(
        pick("returns"), pick("gross_returns"), pick("weights"), pick("turnover"), pick("costs"), name=name
    )
    return result, pd.DataFrame(log)
