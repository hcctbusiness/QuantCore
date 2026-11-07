"""QuantCore — a lightweight, research-grade backtesting engine by HCCT Group."""
from .backtest import BacktestResult, CostModel, backtest, compare
from . import data, metrics, report, sizing, strategies, walkforward

__version__ = "0.1.0"
__all__ = ["BacktestResult", "CostModel", "backtest", "compare", "data", "metrics", "report", "sizing", "strategies", "walkforward"]
