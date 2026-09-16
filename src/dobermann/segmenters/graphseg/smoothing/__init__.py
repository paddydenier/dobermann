# src/dobermann/segmenters/graphseg/smoothing/__init__.py

from .base import Smoother
from .majority_vote import MajorityVoteSmoother

__all__ = [
    "Smoother",
    "MajorityVoteSmoother",
]
