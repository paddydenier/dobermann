from abc import ABC, abstractmethod


class LabelSmoother(ABC):
    @abstractmethod
    def smooth(
        self,
        labels: list[int],
        window: int = 2,
    ) -> list[int]: ...
