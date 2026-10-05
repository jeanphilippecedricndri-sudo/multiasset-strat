"""Systematic multi-asset strategy on top of portopt: signals -> selection -> risk budgeting -> backtest."""

from .backtest import TurnoverBacktester
from .config import StratConfig
from .estimators import BandedSubset, FixedWeights, SignalEstimator
from .signals import signal_table
from .universe import GROUPS, UNIVERSE

__all__ = [
    "StratConfig", "SignalEstimator", "FixedWeights", "BandedSubset", "TurnoverBacktester",
    "signal_table", "UNIVERSE", "GROUPS",
]
