from typing import TYPE_CHECKING

import numpy as np

from .base import Embedder

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class SentenceTransformerEmbedder(Embedder):
    def __init__(self, model: "SentenceTransformer"):
        self._model = model

    @classmethod
    def default(cls) -> "SentenceTransformerEmbedder":
        """Create an embedder backed by the default sentence-transformer model.

        Every call loads a new model. To share one model between several
        segmenters, create the embedder once and pass it to each of them.
        """
        from sentence_transformers import SentenceTransformer

        return cls(SentenceTransformer(DEFAULT_MODEL_NAME))

    def embed(self, sentences: list[str]) -> np.ndarray:
        return self._model.encode(sentences)
