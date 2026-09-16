from abc import ABC, abstractmethod


class PostProcessor(ABC):
    @abstractmethod
    def process(
        self,
        boundaries: list[int],
        n_sentences: int,
    ) -> list[int]: ...
