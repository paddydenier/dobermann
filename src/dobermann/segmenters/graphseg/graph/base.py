from abc import ABC, abstractmethod

import networkx as nx
import numpy as np


class GraphBuilder(ABC):
    @abstractmethod
    def build(self, sim_matrix: np.ndarray) -> nx.Graph: ...
