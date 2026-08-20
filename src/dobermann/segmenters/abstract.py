from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass(slots=True, frozen=True)
class SegmentationResult:
    """Result returned by segmenters.

    Attributes:
        segment_lengths:
            Length of each predicted segment.

        runtime:
            Segmentation runtime in seconds.

        metadata:
            Optional algorithm-specific intermediate values.
    """

    segment_lengths: list[int]
    runtime: float
    # method: str
    metadata: dict = field(default_factory=dict)

    def iter_spans(self):
        start = 0

        for length in self.segment_lengths:
            end = start + length
            yield start, end
            start = end

    # TODO: redesign akwared api
    # curr: result.split(document.sentences)
    # goal: result.split()
    def split(self, sentences: list[str]) -> list[list[str]]:
        chunks = []

        for start, end in self.iter_spans():
            chunks.append(sentences[start:end])

        return chunks


class Segmenter(ABC):
    """Abstract topic segmentation interface."""

    def segment(self, sentences: list[str]) -> SegmentationResult:
        """Segment sentences into topical regions.

        Args:
            sentences:
                Ordered sentence sequence.

        Returns:
            Segmentation result containing:
            - segment lengths
            - runtime information
            - optional metadata
        """
        self._validate_input(sentences)
        return self._segment(sentences)

    @abstractmethod
    def _segment(self, sentences: list[str]) -> SegmentationResult: ...

    def _validate_input(self, sentences: list[str]):

        # 1. Must be a list of str --> 1.1 List, 1.2 Str
        # 2. Cannot be empty list

        if not isinstance(sentences, list):
            raise TypeError("sentences must be a list of str")

        if any(not isinstance(s, str) for s in sentences):
            raise TypeError("all elements in sentences must be str")

        if len(sentences) == 0:
            raise ValueError("sentences must be nonempty")
