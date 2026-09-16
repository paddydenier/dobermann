from abc import ABC, abstractmethod
from typing import Generic, TypeVar

T = TypeVar("T")


class PreProcessor(ABC, Generic[T]):
    @abstractmethod
    def process(self, input: T) -> list[str]: ...
