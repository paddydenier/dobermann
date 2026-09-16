from abc import ABC, abstractmethod


class Labeler(ABC):
    @abstractmethod
    def label(
        self,
        communities: list[list[int]],
        n_sentences: int,
    ) -> list[int]: ...
