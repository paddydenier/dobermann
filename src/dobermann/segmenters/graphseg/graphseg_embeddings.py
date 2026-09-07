import logging
import time

import numpy as np
from transformers import logging as hf_logging

from dobermann.embeddings import Embedder
from dobermann.segmenters.graphseg.community.base import CommunityDetector

from ..abstract import SegmentationResult, Segmenter
from .graph import GraphBuilder
from .labeling.base import Labeler
from .similarity_matrix import SimilarityMatrix
from .smoothing.base import Smoother


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

    def __init__(
        self,
        embedder: Embedder,
        similarity: SimilarityMatrix,
        graph: GraphBuilder,
        community_detector: CommunityDetector,
        labeler: Labeler,
        smoother: Smoother,
    ):
        logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
        hf_logging.set_verbosity_error()

        self.embedder = embedder
        self.similarity = similarity
        self.graph = graph
        self.community = community_detector
        self.labeler = labeler
        self.smoother = smoother

    def _segment(self, sentences: list[str]) -> SegmentationResult:
        start = time.perf_counter()

        embeddings = self.embedder.embed(sentences)
        sim_matrix = self.similarity.compute(embeddings)
        graph = self.graph.build(sim_matrix)
        communities = self.community.detect(graph)
        labels = self.labeler.label(communities=communities, n_sentences=len(sentences))
        smoothed_labels = self.smoother.smooth(labels, window=2)

        # TODO: convert to module: "post-processing" -> could be shared for both
        # take some kind of stracture, return segment lengths
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
