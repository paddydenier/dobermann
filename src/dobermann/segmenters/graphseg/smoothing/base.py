from abc import ABC, abstractmethod


class Smoother(ABC):
    @abstractmethod
    def smooth(
        self,
        labels: list[int],
        window: int = 2,
    ) -> list[int]: ...
