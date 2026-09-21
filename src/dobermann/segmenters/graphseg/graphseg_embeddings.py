import logging
import time

from transformers import logging as hf_logging

from dobermann.embeddings import Embedder
from dobermann.segmenters.graphseg.community.base import CommunityDetector
from dobermann.segmenters.graphseg.postprocessor import PostProcessor

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
        postprocessor: PostProcessor,
    ):
        logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
        hf_logging.set_verbosity_error()

        self.embedder = embedder
        self.similarity = similarity
        self.graph = graph
        self.community = community_detector
        self.labeler = labeler
        self.smoother = smoother
        self.postprocessor = postprocessor

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
        # segment_lengths = self._labels_to_segments(smoothed_labels)
        segment_lengths = self.postprocessor.process(smoothed_labels)

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
