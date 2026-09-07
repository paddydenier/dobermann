from abc import ABC, abstractmethod

import networkx as nx


class CommunityDetector(ABC):
    @abstractmethod
    def detect(self, graph: nx.Graph) -> list[list[int]]: ...
