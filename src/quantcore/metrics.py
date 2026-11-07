"""Performance and risk metrics for periodic return series."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _clean(r: pd.Series) -> pd.Series:
    return pd.Series(r).dropna().astype(float)


def cagr(r: pd.Series, periods: int = 252) -> float:
    r = _clean(r)
    if r.empty:
        return np.nan
    total = float((1 + r).prod())
    years = len(r) / periods
    return total ** (1 / years) - 1 if total > 0 else -1.0


def ann_vol(r: pd.Series, periods: int = 252) -> float:
    return float(_clean(r).std(ddof=1) * np.sqrt(periods))


def sharpe(r: pd.Series, periods: int = 252, rf: float = 0.0) -> float:
    r = _clean(r) - rf / periods
    sd = r.std(ddof=1)
    return float(r.mean() / sd * np.sqrt(periods)) if sd > 0 else np.nan


def sortino(r: pd.Series, periods: int = 252, rf: float = 0.0) -> float:
    r = _clean(r) - rf / periods
    downside = np.sqrt((np.minimum(r, 0) ** 2).mean())
    return float(r.mean() / downside * np.sqrt(periods)) if downside > 0 else np.nan


def drawdown(r: pd.Series) -> pd.Series:
    eq = (1 + _clean(r)).cumprod()
    return eq / eq.cummax() - 1


def max_drawdown(r: pd.Series) -> float:
    dd = drawdown(r)
    return float(dd.min()) if not dd.empty else np.nan


def calmar(r: pd.Series, periods: int = 252) -> float:
    mdd = max_drawdown(r)
    return cagr(r, periods) / abs(mdd) if mdd < 0 else np.nan


def hit_rate(r: pd.Series) -> float:
    r = _clean(r)
    r = r[r != 0]
    return float((r > 0).mean()) if len(r) else np.nan


def var_cvar(r: pd.Series, level: float = 0.95) -> tuple[float, float]:
    """Historical one-period Value-at-Risk and Conditional VaR (returned as negatives)."""
    r = _clean(r)
    q = float(np.quantile(r, 1 - level))
    return q, float(r[r <= q].mean())


def beta_alpha(r: pd.Series, bench: pd.Series, periods: int = 252) -> tuple[float, float]:
    df = pd.concat([_clean(r), _clean(bench)], axis=1, join="inner").dropna()
    if len(df) < 3:
        return np.nan, np.nan
    b = np.cov(df.iloc[:, 0], df.iloc[:, 1])[0, 1] / df.iloc[:, 1].var()
    a = (df.iloc[:, 0].mean() - b * df.iloc[:, 1].mean()) * periods
    return float(b), float(a)


def rolling_sharpe(r: pd.Series, window: int = 252, periods: int = 252) -> pd.Series:
    r = _clean(r)
    return r.rolling(window).mean() / r.rolling(window).std() * np.sqrt(periods)


def summary(r: pd.Series, periods: int = 252, benchmark: pd.Series | None = None) -> pd.Series:
    v95, cv95 = var_cvar(r)
    out = {
        "CAGR": cagr(r, periods),
        "Ann. Vol": ann_vol(r, periods),
        "Sharpe": sharpe(r, periods),
        "Sortino": sortino(r, periods),
        "Max Drawdown": max_drawdown(r),
        "Calmar": calmar(r, periods),
        "Hit Rate": hit_rate(r),
        "Skew": float(_clean(r).skew()),
        "VaR 95%": v95,
        "CVaR 95%": cv95,
    }
    if benchmark is not None:
        b, a = beta_alpha(r, benchmark, periods)
        out["Beta"], out["Alpha (ann.)"] = b, a
    return pd.Series(out)
