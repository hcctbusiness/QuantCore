"""Free data loaders (Yahoo Finance, FRED) with a local parquet/csv cache."""
from __future__ import annotations

import io
import os
from pathlib import Path
from urllib.request import urlopen

import pandas as pd

CACHE = Path(os.environ.get("QUANTCORE_CACHE", Path.home() / ".quantcore_cache"))


def _cache_path(key: str) -> Path:
    CACHE.mkdir(parents=True, exist_ok=True)
    return CACHE / f"{key}.csv"


def prices(tickers: list[str] | str, start: str = "2005-01-01", end: str | None = None, refresh: bool = False) -> pd.DataFrame:
    """Adjusted close prices from Yahoo Finance (requires ``yfinance``)."""
    tickers = [tickers] if isinstance(tickers, str) else list(tickers)
    key = "px_" + "_".join(sorted(t.replace("^", "").replace("=", "") for t in tickers)) + f"_{start}_{end}"
    path = _cache_path(key[:150])
    if path.exists() and not refresh:
        return pd.read_csv(path, index_col=0, parse_dates=True)
    import yfinance as yf

    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False, threads=True)
    px = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]].rename(columns={"Close": tickers[0]})
    px = px[[t for t in tickers if t in px.columns]].dropna(how="all")
    px.to_csv(path)
    return px


def fred(series: list[str] | str, start: str = "1990-01-01", refresh: bool = False) -> pd.DataFrame:
    """FRED series via the public CSV endpoint (no API key needed)."""
    series = [series] if isinstance(series, str) else list(series)
    out = []
    for s in series:
        path = _cache_path(f"fred_{s}")
        if path.exists() and not refresh:
            df = pd.read_csv(path, index_col=0, parse_dates=True)
        else:
            url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={s}"
            txt = urlopen(url, timeout=30).read().decode()  # FRED stalls on custom UAs
            df = pd.read_csv(io.StringIO(txt), index_col=0, parse_dates=True, na_values=".")
            df.columns = [s]
            df.to_csv(path)
        out.append(df)
    return pd.concat(out, axis=1).loc[start:]
