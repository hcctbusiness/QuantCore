"""Position-sizing and portfolio-construction helpers."""
from __future__ import annotations

import numpy as np
import pandas as pd


def equal_weight(signal: pd.DataFrame) -> pd.DataFrame:
    """Equal weight across assets with a non-zero signal, preserving sign."""
    s = np.sign(signal).fillna(0.0)
    n = s.abs().sum(axis=1).replace(0, np.nan)
    return s.div(n, axis=0).fillna(0.0)


def inverse_vol(prices: pd.DataFrame, lookback: int = 60, signal: pd.DataFrame | None = None) -> pd.DataFrame:
    """Inverse-volatility weights (sum of |w| = 1), optionally signed by ``signal``."""
    vol = prices.pct_change(fill_method=None).rolling(lookback).std()
    iv = 1.0 / vol.replace(0, np.nan)
    if signal is not None:
        iv = iv * np.sign(signal.reindex_like(iv)).fillna(0.0)
    return iv.div(iv.abs().sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)


def vol_target(
    weights: pd.DataFrame,
    prices: pd.DataFrame,
    target: float = 0.10,
    lookback: int = 60,
    max_leverage: float = 3.0,
    periods: int = 252,
) -> pd.DataFrame:
    """Scale a weight matrix so the ex-ante portfolio volatility hits ``target``.

    Uses only trailing data (the realised vol of the un-scaled book), so no look-ahead.
    """
    rets = prices[weights.columns].pct_change(fill_method=None).fillna(0.0)
    raw = (weights.shift(1).fillna(0.0) * rets).sum(axis=1)
    realised = raw.rolling(lookback, min_periods=lookback // 2).std() * np.sqrt(periods)
    scale = (target / realised.replace(0, np.nan)).clip(upper=max_leverage).fillna(0.0)
    return weights.mul(scale, axis=0)


def rebalance(weights: pd.DataFrame, freq: str = "ME") -> pd.DataFrame:
    """Hold weights constant between rebalance dates (e.g. 'W-FRI', 'ME', 'QE')."""
    idx = weights.index
    marks = weights.groupby(pd.Grouper(freq=freq)).tail(1).index
    return weights.loc[marks].reindex(idx).ffill().fillna(0.0)
