class SegmentationEvaluator:
    def __init__(self, window_size: int | None = None):
        self.k = window_size

    # ---------- Public API ----------

    def pk(self, ref: list[int], hyp: list[int]) -> float:
        self._validate(ref, hyp)

        ref_b = self._lengths_to_boundaries(ref)
        hyp_b = self._lengths_to_boundaries(hyp)

        n = len(ref_b)
        k = self.k or self._default_k(ref)

        if k <= 0 or k >= n:
            raise ValueError(f"Invalid window size k={k} for n={n}")

        errors = 0
        total = n - k

        for i in range(total):
            same_ref = self._same_segment(ref_b, i, i + k)
            same_hyp = self._same_segment(hyp_b, i, i + k)

            if same_ref != same_hyp:
                errors += 1

        return errors / total

    def windowdiff(self, ref: list[int], hyp: list[int]) -> float:
        self._validate(ref, hyp)

        ref_b = self._lengths_to_boundaries(ref)
        hyp_b = self._lengths_to_boundaries(hyp)

        n = len(ref_b)
        k = self.k or self._default_k(ref)

        if k <= 0 or k >= n:
            raise ValueError(f"Invalid window size k={k} for n={n}")

        errors = 0
        total = n - k

        for i in range(total):
            ref_count = sum(ref_b[i : i + k])
            hyp_count = sum(hyp_b[i : i + k])

            if ref_count != hyp_count:
                errors += 1

        return errors / total

    # ---------- Core helpers ----------

    @staticmethod
    def _lengths_to_boundaries(lengths: list[int]) -> list[int]:
        """
        Convert segment lengths → boundary vector.

        Example:
        [3,2,4] → [0,0,1, 0,1, 0,0,0]
        """
        boundaries = []
        for len in lengths[:-1]:
            if len <= 0:
                raise ValueError("Segment lengths must be positive")
            boundaries.extend([0] * (len - 1))
            boundaries.append(1)
        return boundaries

    @staticmethod
    def _same_segment(boundaries: list[int], i: int, j: int) -> bool:
        """
        True if i and j are in same segment (no boundary between them)
        """
        return sum(boundaries[i:j]) == 0

    @staticmethod
    def _default_k(lengths: list[int]) -> int:
        """
        Standard choice: half the average segment length (from reference)
        """
        avg = sum(lengths) / len(lengths)
        return max(1, int(round(avg / 2)))

    @staticmethod
    def _validate(ref: list[int], hyp: list[int]):
        """
        Ensure both segmentations cover the same number of sentences
        """
        if not ref or not hyp:
            raise ValueError("Segmentations must not be empty")

        if sum(ref) != sum(hyp):
            raise ValueError(
                f"Mismatch in total length: ref={sum(ref)}, hyp={sum(hyp)}"
            )
