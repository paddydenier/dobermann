import logging
import time

import networkx as nx
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import logging as hf_logging

from .abstract import SegmentationResult, Segmenter


class GraphSegEmbeddings(Segmenter):
    """
    Improved GraphSeg-style topic segmentation using sentence embeddings.

    Pipeline:
    1. Encode sentences
    2. Build weighted similarity graph with positional decay
    3. Detect communities
    4. Convert communities -> ordered labels
    5. Smooth labels with neighborhood majority vote
    6. Convert smoothed labels -> segment lengths
    """

    def __init__(self, model: str):
        logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
        hf_logging.set_verbosity_error()
        self.model = SentenceTransformer(model)

    # --------------------------------------------------
    # MAIN
    # --------------------------------------------------

    def segment(self, sentences: list[str]) -> SegmentationResult:
        start = time.perf_counter()

        embeddings = self._vectorize(sentences)
        sim_matrix = self._similarity_matrix(embeddings)

        graph = self._build_graph(sim_matrix)
        communities = self._communities(graph)

        labels = self._communities_to_labels(
            communities=communities,
            n_sentences=len(sentences),
        )

        smoothed_labels = self._smooth_labels(labels, window=2)

        segment_lengths = self._labels_to_segments(smoothed_labels)

        runtime = time.perf_counter() - start

        metadata = {
            "embeddings": embeddings,
            "similarity_matrix": sim_matrix,
            "graph": graph,
            "communities": communities,
            "labels": labels,
            "smoothed_labels": smoothed_labels,
        }

        return SegmentationResult(
            segment_lengths=segment_lengths,
            runtime=runtime,
            metadata=metadata,
        )

    # --------------------------------------------------
    # VECTORIZE
    # --------------------------------------------------

    def _vectorize(self, sentences: list[str]) -> np.ndarray:
        return self.model.encode(sentences)

    # --------------------------------------------------
    # SIMILARITY
    # --------------------------------------------------

    def _similarity_matrix(self, embeddings: np.ndarray) -> np.ndarray:
        return cosine_similarity(embeddings)

    # --------------------------------------------------
    # GRAPH BUILDING
    # --------------------------------------------------

    def _build_graph(
        self,
        sim_matrix: np.ndarray,
        max_distance: int = 15,
        min_similarity: float = 0.30,
        decay: float = 0.15,
    ) -> nx.Graph:
        """
        Weighted graph.

        Edge weight:
            similarity * exp(-decay * distance)

        Keeps softer structure than hard thresholding.
        """

        n = len(sim_matrix)
        G = nx.Graph()

        for i in range(n):
            G.add_node(i)

        for i in range(n):
            for j in range(i + 1, min(n, i + max_distance + 1)):
                sim = float(sim_matrix[i, j])

                if sim < min_similarity:
                    continue

                distance = abs(i - j)
                weight = sim * np.exp(-decay * distance)

                if weight > 0:
                    G.add_edge(i, j, weight=weight)

        return G

    # --------------------------------------------------
    # COMMUNITIES
    # --------------------------------------------------

    def _communities(self, graph: nx.Graph) -> list[list[int]]:
        """
        Greedy modularity clustering.
        """

        communities = nx.algorithms.community.greedy_modularity_communities(
            graph,
            weight="weight",
        )

        return [sorted(list(c)) for c in communities]

    # --------------------------------------------------
    # COMMUNITIES -> LABELS
    # --------------------------------------------------

    def _communities_to_labels(
        self,
        communities: list[list[int]],
        n_sentences: int,
    ) -> list[int]:
        """
        Convert unordered communities into sentence-order labels.
        """

        labels = [-1] * n_sentences

        for cid, community in enumerate(communities):
            for idx in community:
                labels[idx] = cid

        # isolated nodes remain unique singleton labels
        next_label = len(communities)

        for i in range(n_sentences):
            if labels[i] == -1:
                labels[i] = next_label
                next_label += 1

        return labels

    # --------------------------------------------------
    # LABEL SMOOTHING
    # --------------------------------------------------

    def _smooth_labels(
        self,
        labels: list[int],
        window: int = 2,
    ) -> list[int]:
        """
        Majority-vote smoothing.

        Example:
            A A B A A -> A A A A A
        """

        smoothed = labels.copy()
        n = len(labels)

        for i in range(n):
            left = max(0, i - window)
            right = min(n, i + window + 1)

            neighborhood = labels[left:right]

            values, counts = np.unique(neighborhood, return_counts=True)
            majority = values[np.argmax(counts)]

            smoothed[i] = int(majority)

        return smoothed

    # --------------------------------------------------
    # LABELS -> SEGMENTS
    # --------------------------------------------------

    def _labels_to_segments(self, labels: list[int]) -> list[int]:
        """
        Convert contiguous labels into segment lengths.
        """

        lengths = []

        current = labels[0]
        run = 1

        for i in range(1, len(labels)):
            if labels[i] == current:
                run += 1
            else:
                lengths.append(run)
                run = 1
                current = labels[i]

        lengths.append(run)

        return lengths
