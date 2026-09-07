import networkx as nx

from .base import CommunityDetector


class GreedyModularityCommunityDetector(CommunityDetector):
    def detect(self, graph: nx.Graph) -> list[list[int]]:
        communities = nx.algorithms.community.greedy_modularity_communities(
            graph,
            weight="weight",
        )

        return [sorted(list(c)) for c in communities]
