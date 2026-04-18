import logging
import time

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import logging as hf_logging
from dataclasses import dataclass

# TODO: String representation

# TODO: convert to pure functions
# TODO: store result in dataclass


@dataclass
class SegmentationResult:
    segment_lengths: list[int]
    runtime: float


class TextTilingEmbeddings:
    def __init__(self, model: str):
        logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
        hf_logging.set_verbosity_error()
        self.model = SentenceTransformer(model)

    def segment(self, sentences: list[str]):
        start = time.perf_counter()
        self.sentences = sentences
        self.embeddings = self._vectorize()
        self.similarities = self._similarity()
        self.smoothed = self._smooth()
        self.boundaries = self._boundaries(signal=self.smoothed)
        self.segment_lengths = self._postprocess(
            boundaries=self.boundaries, n_sentences=len(self.sentences)
        )
        end = time.perf_counter()
        self.time = end - start

    def _vectorize(self):
        embeddings = self.model.encode(self.sentences)
        return embeddings

    def _similarity(self):
        sims = []

        for i in range(len(self.embeddings) - 1):
            sim = cosine_similarity([self.embeddings[i]], [self.embeddings[i + 1]])[0][
                0
            ]

            sims.append(sim)

        return sims

    def _smooth(self):
        smoothed = uniform_filter1d(self.similarities, size=2)
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
