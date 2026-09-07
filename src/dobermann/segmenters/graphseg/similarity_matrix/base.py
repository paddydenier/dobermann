from abc import ABC, abstractmethod

import numpy as np


class SimilarityMatrix(ABC):
    @abstractmethod
    def compute(self, embeddings: np.ndarray) -> np.ndarray: ...
