from dobermann.segmenters.abstract import SegmentationResult
from sentence_transformers import SentenceTransformer

from ...embeddings import Embedder, SentenceTransformerEmbedder
from ...preprocessors import IdentityPreProcessor, PreProcessor
from .community import CommunityDetector, GreedyModularityCommunityDetector
from .graph import GraphBuilder, WeightedGraphBuilder
from .graphseg_embeddings import GraphSegEmbeddings
from .labeling import CommunityLabeler, Labeler
from .postprocessor import LabelsToSegmentsPostProcessor, PostProcessor
from .similarity_matrix import CosineSimilarityMatrix, SimilarityMatrix
from .smoothing import MajorityVoteSmoother, Smoother


class GraphSeg(GraphSegEmbeddings):
    """High-level facade for GraphSeg topic segmentation."""

    def __init__(
        self,
        pre_processor: PreProcessor | None = None,
        embedder: Embedder | None = None,
        similarity: SimilarityMatrix | None = None,
        graph: GraphBuilder | None = None,
        community_detector: CommunityDetector | None = None,
        labeler: Labeler | None = None,
        smoother: Smoother | None = None,
        postprocessor: PostProcessor | None = None,
    ):
        self.pre_processor = pre_processor or IdentityPreProcessor()

        super().__init__(
            embedder=embedder
            or SentenceTransformerEmbedder(
                SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            ),
            similarity=similarity or CosineSimilarityMatrix(),
            graph=graph or WeightedGraphBuilder(),
            community_detector=community_detector
            or GreedyModularityCommunityDetector(),
            labeler=labeler or CommunityLabeler(),
            smoother=smoother or MajorityVoteSmoother(),
            postprocessor=postprocessor or LabelsToSegmentsPostProcessor(),
        )

    def segment(self, input) -> SegmentationResult:
        sentences = self.pre_processor.process(input)
        return super().segment(sentences)
