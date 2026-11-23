import logging
from typing import TYPE_CHECKING

import numpy as np

from .base import Embedder

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

logger = logging.getLogger(__name__)


class SentenceTransformerEmbedder(Embedder):
    def __init__(self, model: "SentenceTransformer"):
        self._model = model

    @classmethod
    def default(cls, device: str | None = None) -> "SentenceTransformerEmbedder":
        """Create an embedder backed by the default sentence-transformer model.

        Every call loads a new model. To share one model between several
        segmenters, create the embedder once and pass it to each of them.

        Args:
            device: The device to run the model on, such as ``"cpu"``,
                ``"cuda"`` or ``"cuda:0"``. ``None`` picks a device
                automatically (a GPU if one is available).

        Raises:
            ValueError: If a CUDA device is requested but CUDA is not
                available.
        """
        from sentence_transformers import SentenceTransformer
        from transformers.utils import is_torch_cuda_available

        if device is not None and device.startswith("cuda"):
            if not is_torch_cuda_available():
                raise ValueError(
                    f"Device {device!r} was requested, but CUDA is not "
                    "available. Install a CUDA build of torch or use "
                    "device='cpu'."
                )

        embedder = cls(SentenceTransformer(DEFAULT_MODEL_NAME, device=device))
        logger.info(
            "Loaded sentence-transformer model %s on %s",
            DEFAULT_MODEL_NAME,
            embedder.device,
        )
        return embedder

    @property
    def device(self) -> str:
        """The device the model runs on, such as ``"cpu"`` or ``"cuda:0"``."""
        return str(self._model.device)

    def embed(self, sentences: list[str]) -> np.ndarray:
        return self._model.encode(sentences)
