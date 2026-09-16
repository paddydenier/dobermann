from .base import PostProcessor


class LabelsToSegmentsPostProcessor(PostProcessor):
    """Convert contiguous labels into segment lengths."""

    def process(self, labels: list[int]) -> list[int]:
        if not labels:
            return []

        lengths = []

        current = labels[0]
        run = 1

        for label in labels[1:]:
            if label == current:
                run += 1
            else:
                lengths.append(run)
                run = 1
                current = label

        lengths.append(run)

        return lengths
