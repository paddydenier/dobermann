import logging
import time

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks
from sklearn.metrics.pairwise import cosine_similarity
from transformers import logging as hf_logging

from .abstract import SegmentationResult, Segmenter
from ..embeddings import Embedder


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

    def __init__(self, embedder: Embedder):
        self.embedder = embedder
        logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
        hf_logging.set_verbosity_error()

    def _segment(self, sentences: list[str]) -> SegmentationResult:
        start = time.perf_counter()

        embeddings = self.embedder.embed(sentences)
        similarities = self._similarity(embeddings)
        smoothed = self._smooth(similarities)
        boundaries = self._boundaries(signal=smoothed)
        segment_lengths = self._postprocess(
            boundaries=boundaries, n_sentences=len(sentences)
        )

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

    def _vectorize(self, model, sentences):
        embeddings = model.encode(sentences)
        return embeddings

    def _similarity(self, embeddings):
        sims = []

        for i in range(len(embeddings) - 1):
            sim = cosine_similarity([embeddings[i]], [embeddings[i + 1]])[0][0]

            sims.append(sim)

        return sims

    def _smooth(self, similarities):
        # TODO: make smoothing windows a class state
        smoothed = uniform_filter1d(similarities, size=2)
        return smoothed

    # TODO: fix thresholding
    # TODO: adaptive thresholding

    def _boundaries(self, signal, alpha=0.5, plimit=0.1):
        """
        signal: smoothed similarity curve
        alpha : threshold parameter
        plimit: minimum candidate score
        """

        signal = np.asarray(signal)

        valleys, _ = find_peaks(-signal)

        candidates = []

        for v in valleys:
            left_peak = np.max(signal[: v + 1])
            right_peak = np.max(signal[v:])

            score = 0.5 * (left_peak + right_peak - 2 * signal[v])

            if score >= plimit:
                candidates.append((v, score))

        if not candidates:
            return []

        vals = np.array([s for _, s in candidates])

        mu = np.mean(vals)
        sigma = np.std(vals)

        threshold = mu - alpha * sigma

        boundaries = [idx for idx, score in candidates if score >= threshold]

        return boundaries

    def _postprocess(self, boundaries, n_sentences):
        boundaries = sorted(int(b) for b in boundaries)

        lengths = []
        start = 0

        for b in boundaries:
            end = b + 1  # boundary after sentence b
            lengths.append(end - start)
            start = end

        lengths.append(n_sentences - start)

        return lengths
