import numpy as np
from sentence_transformers import SentenceTransformer

from .base import Embedder

# TODO: implement in current algorithms


class SentenceTransformerEmbedder(Embedder):
    def __init__(self, model: SentenceTransformer):
        self._model = model

    def embed(self, sentences: list[str]) -> np.ndarray:
        return self._model.encode(sentences)
