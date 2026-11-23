import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from .base import SimilarityMatrix


class CosineSimilarityMatrix(SimilarityMatrix):
    def compute(self, embeddings: np.ndarray) -> np.ndarray:
        return cosine_similarity(embeddings)
