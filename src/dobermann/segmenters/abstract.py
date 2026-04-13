from abc import ABC, abstractmethod


# takes in a smoother
class Segmenter(ABC):
    @abstractmethod
    def segment(self, sentences: list[str]) -> list[int]:
        pass

    @abstractmethod
    def _preprocess(self, sentences: list[str]):
        pass

    @abstractmethod
    def _transform(self):
        pass

    @abstractmethod
    def _similairty(self):
        pass

    @abstractmethod
    def _smooth(self):
        pass

    def _boundaries(self):
        pass

    def _postprocess(self):
        pass
