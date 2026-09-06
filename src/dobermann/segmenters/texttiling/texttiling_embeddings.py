import logging
import time

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks
from sklearn.metrics.pairwise import cosine_similarity
from transformers import logging as hf_logging

from ...embeddings import Embedder
from ..abstract import SegmentationResult, Segmenter
from .boundaries import BoundaryDetector
from .postprocessor import PostProcessor  # needs better naming
from .similarity import Similarity
from .smoothing import Smoother

# no need to know implementation, just the abstraction

# TODO list for refactoring into components
# TODO: move TextTilingEmbeddings into a texttiling/ folder
# TODO: deconstruct each pipeline function into components and implement here

# NOTE: don't set defaults here, use a factory/builder


class TextTilingEmbeddings(Segmenter):
    """Embedding-based TextTiling segmentation.

    This segmenter replaces lexical similarity with sentence
    embedding similarity computed from a transformer model.

    Pipeline:
        1. Encode sentences into embeddings
        2. Compute adjacent cosine similarities
        3. Smooth similarity signal
        4. Detect valley boundaries
        5. Convert boundaries into segment lengths

    Args:
        model:
            SentenceTransformer model name.
    """

    def __init__(
        self,
        embedder: Embedder,
        similarity: Similarity,
        smoother: Smoother,
        boundary: BoundaryDetector,
        post_procesor: PostProcessor,
    ):
        self.embedder = embedder
        self.similarity = similarity
        self.smoother = smoother
        self.boundary = boundary
        self.post_procesor = post_procesor
        logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
        hf_logging.set_verbosity_error()

    def _segment(self, sentences: list[str]) -> SegmentationResult:
        start = time.perf_counter()

        embeddings = self.embedder.embed(sentences)
        similarities = self.similarity.compute(embeddings)
        smoothed = self.smoother.smooth(similarities)
        boundaries = self.boundary.boundaries(smoothed)

        segment_lengths = self.post_procesor.process(boundaries, len(sentences))

        end = time.perf_counter()
        runtime = end - start

        metadata = {
            "embeddings": embeddings,
            "similarities": similarities,
            "smoothed": smoothed,
            "boundaries": boundaries,
        }

        return SegmentationResult(
            segment_lengths=segment_lengths, runtime=runtime, metadata=metadata
        )
