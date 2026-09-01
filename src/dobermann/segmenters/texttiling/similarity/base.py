from abc import ABC, abstractmethod

import numpy as np


class Similarity(ABC):
    @abstractmethod
    def compute(self, embeddings: np.ndarray) -> list[float]: ...
