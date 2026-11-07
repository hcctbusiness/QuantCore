"""Tearsheet charts and markdown tables."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import metrics as _m  # noqa: E402

PALETTE = ["#0B3C5D", "#D9822B", "#328CC1", "#7A7A7A", "#2E7D32", "#B23A48"]
_PCT = {"CAGR", "Ann. Vol", "Max Drawdown", "Hit Rate", "VaR 95%", "CVaR 95%", "Alpha (ann.)", "Avg Daily Turnover", "Annual Cost Drag"}


def monthly_table(r: pd.Series) -> pd.DataFrame:
    m = (1 + r).groupby([r.index.year, r.index.month]).prod() - 1
    t = m.unstack()
    t.columns = [pd.Timestamp(2000, c, 1).strftime("%b") for c in t.columns]
    t["Year"] = (1 + r).groupby(r.index.year).prod() - 1
    return t


def tearsheet(returns: dict[str, pd.Series] | pd.Series, path: str | Path, title: str = "Strategy Tearsheet") -> Path:
    """Save a 4-panel tearsheet: equity, drawdown, rolling Sharpe, monthly heatmap of the first series."""
    if isinstance(returns, pd.Series):
        returns = {returns.name or "strategy": returns}
    fig = plt.figure(figsize=(13, 13), dpi=110)
    gs = fig.add_gridspec(4, 1, height_ratios=[2.2, 1, 1, 2.8], hspace=0.45, top=0.93, bottom=0.05)
    ax1, ax2, ax3, ax4 = (fig.add_subplot(gs[i]) for i in range(4))
    for i, (n, r) in enumerate(returns.items()):
        c = PALETTE[i % len(PALETTE)]
        lw = 2.0 if i == 0 else 1.3
        ax1.plot((1 + r).cumprod(), color=c, lw=lw, label=f"{n}  (Sharpe {_m.sharpe(r):.2f})")
        ax2.plot(_m.drawdown(r), color=c, lw=lw)
        ax3.plot(_m.rolling_sharpe(r), color=c, lw=lw)
    ax1.set_yscale("log"); _f = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"); ax1.yaxis.set_major_formatter(_f); ax1.yaxis.set_minor_formatter(_f); ax1.set_title("Growth of $1 (log scale)", loc="left", fontweight="bold"); ax1.legend(frameon=False)
    ax2.set_title("Drawdown", loc="left", fontweight="bold"); ax2.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax3.set_title("Rolling 1-year Sharpe", loc="left", fontweight="bold"); ax3.axhline(0, color="k", lw=0.6)
    first = next(iter(returns.values()))
    mt = monthly_table(first).drop(columns="Year")
    v = np.nanmax(np.abs(mt.values)) if mt.size else 0.1
    ax4.imshow(mt.values, aspect="auto", cmap="RdYlGn", vmin=-v, vmax=v)
    ax4.set_xticks(range(mt.shape[1]), mt.columns); ax4.set_yticks(range(mt.shape[0]), mt.index)
    for (y, x), val in np.ndenumerate(mt.values):
        if not np.isnan(val):
            ax4.text(x, y, f"{val*100:.1f}", ha="center", va="center", fontsize=6.5)
    ax4.set_title(f"Monthly returns (%) — {next(iter(returns))}", loc="left", fontweight="bold")
    for a in (ax1, ax2, ax3):
        a.grid(alpha=0.25); a.spines[["top", "right"]].set_visible(False)
    fig.suptitle(title, fontsize=15, fontweight="bold", x=0.06, y=0.975, ha="left")
    fig.text(0.99, 0.02, "HCCT Group · QuantCore", ha="right", fontsize=8, color="#7A7A7A")
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    return path


def to_markdown(stats: pd.DataFrame | pd.Series) -> str:
    df = stats.to_frame() if isinstance(stats, pd.Series) else stats
    fmt = df.copy().astype(object)
    for idx in df.index:
        for c in df.columns:
            v = df.loc[idx, c]
            fmt.loc[idx, c] = "—" if pd.isna(v) else (f"{v*100:.2f}%" if idx in _PCT else f"{v:.2f}")
    head = "| Metric | " + " | ".join(map(str, df.columns)) + " |"
    sep = "|---|" + "---:|" * len(df.columns)
    rows = [f"| {i} | " + " | ".join(fmt.loc[i]) + " |" for i in df.index]
    return "\n".join([head, sep, *rows])
