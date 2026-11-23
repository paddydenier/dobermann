from dobermann.embeddings import Embedder, SentenceTransformerEmbedder
from dobermann.preprocessors import IdentityPreProcessor, PreProcessor
from dobermann.segmenters.abstract import SegmentationResult, Segmenter

from .community import GreedyModularityCommunityDetector
from .graph import WeightedGraphBuilder
from .graphseg_embeddings import GraphSegEmbeddings
from .labeling import CommunityLabeler
from .postprocessor import LabelsToSegmentsPostProcessor
from .similarity_matrix import CosineSimilarityMatrix
from .smoothing import MajorityVoteSmoother


class GraphSeg(Segmenter):
    """High-level facade for GraphSeg topic segmentation."""

    def __init__(
        self,
        input_processor: PreProcessor,
        algorithm: GraphSegEmbeddings,
    ):
        self.input_processor = input_processor
        self.algorithm = algorithm

    @classmethod
    def default(cls, *, embedder: Embedder | None = None) -> "GraphSeg":
        """Create GraphSeg with the default pipeline.

        Args:
            embedder: Embedder that turns sentences into vectors. Defaults to
                a new `SentenceTransformerEmbedder.default()`, which loads its
                own model. Passing the same embedder to several segmenters
                shares one model between them.
        """
        input_processor = IdentityPreProcessor()
        if embedder is None:
            embedder = SentenceTransformerEmbedder.default()

        algorithm = GraphSegEmbeddings(
            embedder=embedder,
            similarity=CosineSimilarityMatrix(),
            graph=WeightedGraphBuilder(),
            community_detector=GreedyModularityCommunityDetector(),
            labeler=CommunityLabeler(),
            smoother=MajorityVoteSmoother(),
            postprocessor=LabelsToSegmentsPostProcessor(),
        )

        return cls(
            input_processor=input_processor,
            algorithm=algorithm,
        )

    def segment(self, input) -> SegmentationResult:
        sentences = self.input_processor.process(input)
        return self.algorithm.segment(sentences)
