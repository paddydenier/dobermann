import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from .base import Similarity


class CosineSimilarity(Similarity):
    def compute(self, embeddings: np.ndarray) -> list[float]:
        similarities = []

        for i in range(len(embeddings) - 1):
            sim = cosine_similarity(
                [embeddings[i]],
                [embeddings[i + 1]],
            )[0][0]

            similarities.append(sim)

        return similarities
