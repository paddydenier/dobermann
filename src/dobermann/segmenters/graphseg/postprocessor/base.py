from abc import ABC, abstractmethod


class LabelPostProcessor(ABC):
    """Abstract interface for postprocessing segmentation results."""

    @abstractmethod
    def process(self, labels: list[int]) -> list[int]: ...
