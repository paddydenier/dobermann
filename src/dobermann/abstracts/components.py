from abc import ABC, abstractmethod
from typing import Any, List


class PreProcessor(ABC):
    @abstractmethod
    def process(self, file_path: str) -> Any:
        pass


class Representation(ABC):
    @abstractmethod
    def transform(self, processed_text: Any) -> Any:
        pass


class Similarity(ABC):
    @abstractmethod
    def compute(self, representation: Any) -> Any:
        pass


class SegmentationSignal(ABC):
    @abstractmethod
    def build(self, similarity_matrix: Any) -> Any:
        pass


class BoundarySelector(ABC):
    @abstractmethod
    def select(self, signal: Any) -> List[int]:
        pass


class PostProcessor(ABC):
    @abstractmethod
    def process(self, text: str, boundaries: List[int]) -> List[str]:
        pass
