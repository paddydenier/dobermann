from .base import PostProcessor


class BoundaryToLengthProcessor(PostProcessor):
    def process(
        self,
        boundaries: list[int],
        n_sentences: int,
    ) -> list[int]:
        boundaries = sorted(int(b) for b in boundaries)

        lengths = []
        start = 0

        for b in boundaries:
            end = b + 1
            lengths.append(end - start)
            start = end

        lengths.append(n_sentences - start)

        return lengths
