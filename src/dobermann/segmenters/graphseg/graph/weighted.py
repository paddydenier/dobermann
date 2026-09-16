import networkx as nx
import numpy as np

from .base import GraphBuilder


class WeightedGraphBuilder(GraphBuilder):
    def __init__(
        self,
        max_distance: int = 15,
        min_similarity: float = 0.30,
        decay: float = 0.15,
    ):
        self.max_distance = max_distance
        self.min_similarity = min_similarity
        self.decay = decay

    def build(self, sim_matrix: np.ndarray) -> nx.Graph:
        n = len(sim_matrix)
        graph = nx.Graph()

        for i in range(n):
            graph.add_node(i)

        for i in range(n):
            for j in range(
                i + 1,
                min(n, i + self.max_distance + 1),
            ):
                sim = float(sim_matrix[i, j])

                if sim < self.min_similarity:
                    continue

                distance = abs(i - j)
                weight = sim * np.exp(-self.decay * distance)

                if weight > 0:
                    graph.add_edge(i, j, weight=weight)

        return graph
