from abc import ABC, abstractmethod


# takes in a smoother
class Segmenter(ABC):
    @abstractmethod
    def segment(self, sentences: list[str]) -> list[int]:
        pass
