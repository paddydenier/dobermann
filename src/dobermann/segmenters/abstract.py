from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass(slots=True, frozen=True)
class SegmentationResult:
    segment_lengths: list[int]
    runtime: float
    # method: str
    metadata: dict = field(default_factory=dict)


# TODO: validation make segment wrap around _segment


# takes in a smoother
class Segmenter(ABC):
    def segment(self, sentences: list[str]) -> SegmentationResult:
        self._validate_input(sentences)
        return self._segment(sentences)

    # TODO: underscore current segment function
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
