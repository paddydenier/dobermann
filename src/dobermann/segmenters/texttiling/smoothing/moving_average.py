import numpy as np
from scipy.ndimage import uniform_filter1d

from .base import Smoother


class MovingAverageSmoother(Smoother):
    def __init__(self, window_size: int = 2):
        self.window_size = window_size

    def smooth(self, signal: np.ndarray) -> np.ndarray:
        return uniform_filter1d(signal, size=self.window_size)
