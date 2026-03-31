from abc import ABC, abstractmethod


class Segmenter(ABC):
    @abstractmethod
    def __init__(self):
        self.name = "test"
