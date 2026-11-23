from typing import TYPE_CHECKING

import numpy as np

from .base import Embedder

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

# TODO: implement in current algorithms


class SentenceTransformerEmbedder(Embedder):
    def __init__(self, model: "SentenceTransformer"):
        self._model = model

    def embed(self, sentences: list[str]) -> np.ndarray:
        return self._model.encode(sentences)
