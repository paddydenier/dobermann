from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass(slots=True, frozen=True)
class SegmentationResult:
    segment_lengths: list[int]
    runtime: float
    # method: str
    metadata: dict = field(default_factory=dict)


# takes in a smoother
class Segmenter(ABC):
    @abstractmethod
    def segment(self, sentences: list[str]) -> SegmentationResult:
        pass
