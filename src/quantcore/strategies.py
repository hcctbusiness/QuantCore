"""Reference strategies. Each returns a causal target-weight DataFrame."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .sizing import inverse_vol, rebalance


def buy_and_hold(prices: pd.DataFrame, weights: dict[str, float] | None = None) -> pd.DataFrame:
    w = weights or {c: 1 / prices.shape[1] for c in prices.columns}
    return pd.DataFrame({c: w.get(c, 0.0) for c in prices.columns}, index=prices.index)


def sma_crossover(prices: pd.DataFrame, fast: int = 50, slow: int = 200, long_only: bool = True) -> pd.DataFrame:
    sig = np.sign(prices.rolling(fast).mean() - prices.rolling(slow).mean())
    if long_only:
        sig = sig.clip(lower=0)
    sig = sig.where(prices.rolling(slow).count() >= slow, 0.0)
    n = prices.notna().sum(axis=1).replace(0, np.nan)
    return sig.div(n, axis=0).fillna(0.0)


def time_series_momentum(prices: pd.DataFrame, lookback: int = 252, vol_lookback: int = 60, freq: str = "ME") -> pd.DataFrame:
    """Moskowitz-Ooi-Pedersen style TSMOM: sign of trailing return, inverse-vol sized, monthly rebalanced."""
    sig = np.sign(prices / prices.shift(lookback) - 1)
    w = inverse_vol(prices, vol_lookback, signal=sig)
    return rebalance(w, freq)


def cross_sectional_momentum(prices: pd.DataFrame, lookback: int = 252, skip: int = 21, top: float = 0.3, freq: str = "ME") -> pd.DataFrame:
    """Long the top ``top`` fraction by 12-1 momentum, equal weighted."""
    mom = prices.shift(skip) / prices.shift(lookback) - 1
    rank = mom.rank(axis=1, pct=True)
    long = (rank >= 1 - top).astype(float)
    w = long.div(long.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    return rebalance(w, freq)
