import numpy as np

from .base import Smoother


class MajorityVoteSmoother(Smoother):
    def smooth(
        self,
        labels: list[int],
        window: int = 2,
    ) -> list[int]:
        """
        Majority-vote smoothing.

        Example:
            A A B A A -> A A A A A
        """

        smoothed = labels.copy()
        n = len(labels)

        for i in range(n):
            left = max(0, i - window)
            right = min(n, i + window + 1)

            neighborhood = labels[left:right]

            values, counts = np.unique(neighborhood, return_counts=True)
            majority = values[np.argmax(counts)]

            smoothed[i] = int(majority)

        return smoothed
