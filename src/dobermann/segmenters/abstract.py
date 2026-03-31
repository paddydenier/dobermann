from abc import ABC, abstractmethod


# takes in a smoother
class Segmenter(ABC):
    @abstractmethod
    def __init__(self):
        self.name = "test"

    @abstractmethod
    def transform(self):
        pass

    @abstractmethod
    def compute_similarity(self):
        pass

    @abstractmethod
    def smooth_similarity_curve(self):
        pass

    def select_boundaries(self):
        pass

    def generate_segments(self):
        pass
