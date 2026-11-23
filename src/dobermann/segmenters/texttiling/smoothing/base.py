from abc import ABC, abstractmethod

import numpy as np


class SignalSmoother(ABC):
    @abstractmethod
    def smooth(self, signal: np.ndarray) -> np.ndarray: ...
