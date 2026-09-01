import numpy as np
from scipy.signal import find_peaks

from .base import BoundaryDetector


class AdaptiveValleyBoundaryDetector(BoundaryDetector):
    def __init__(self, alpha: float = 0.5, plimit: float = 0.1):
        self.alpha = alpha
        self.plimit = plimit

    def boundaries(self, signal: np.ndarray) -> list[int]:
        valleys, _ = find_peaks(-signal)

        candidates = []

        for v in valleys:
            left_peak = np.max(signal[: v + 1])
            right_peak = np.max(signal[v:])

            score = 0.5 * (left_peak + right_peak - 2 * signal[v])

            if score >= self.plimit:
                candidates.append((v, score))

        if not candidates:
            return []

        scores = np.array([score for _, score in candidates])

        mu = np.mean(scores)
        sigma = np.std(scores)

        threshold = mu - self.alpha * sigma

        return [idx for idx, score in candidates if score >= threshold]
