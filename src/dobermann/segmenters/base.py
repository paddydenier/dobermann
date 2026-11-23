from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Generic, TypeVar

T = TypeVar("T")


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
    sentences: list[str]
    runtime: float
    # method: str
    metadata: dict = field(default_factory=dict)

    def iter_spans(self):
        start = 0

        for length in self.segment_lengths:
            end = start + length
            yield start, end
            start = end

    @property
    def segments(self) -> list[list[str]]:
        return [self.sentences[start:end] for start, end in self.iter_spans()]


class Segmenter(ABC, Generic[T]):
    """Abstract topic segmentation interface."""

    @abstractmethod
    def segment(self, input: T) -> SegmentationResult:
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
        ...
