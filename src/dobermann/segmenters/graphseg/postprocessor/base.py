from abc import ABC, abstractmethod


class PostProcessor(ABC):
    """Abstract interface for postprocessing segmentation results."""

    @abstractmethod
    def process(self, labels: list[int]) -> list[int]: ...
