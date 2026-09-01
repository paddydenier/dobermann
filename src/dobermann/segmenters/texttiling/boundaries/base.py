from abc import ABC, abstractmethod

import numpy as np


class BoundaryDetector(ABC):
    @abstractmethod
    def boundaries(self, signal: np.ndarray) -> list[int]: ...
