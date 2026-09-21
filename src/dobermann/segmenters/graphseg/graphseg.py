from sentence_transformers import SentenceTransformer

from dobermann.segmenters.abstract import SegmentationResult

from ...embeddings import SentenceTransformerEmbedder
from ...preprocessors import IdentityPreProcessor, PreProcessor
from .community import GreedyModularityCommunityDetector
from .graph import WeightedGraphBuilder
from .graphseg_embeddings import GraphSegEmbeddings
from .labeling import CommunityLabeler
from .postprocessor import LabelsToSegmentsPostProcessor
from .similarity_matrix import CosineSimilarityMatrix
from .smoothing import MajorityVoteSmoother


class GraphSeg:
    """High-level facade for GraphSeg topic segmentation."""

    def __init__(
        self,
        input_processor: PreProcessor,
        algorithm: GraphSegEmbeddings,
    ):
        self.input_processor = input_processor
        self.algorithm = algorithm

    @classmethod
    def default(cls) -> "GraphSeg":
        input_processor = IdentityPreProcessor()

        algorithm = GraphSegEmbeddings(
            embedder=SentenceTransformerEmbedder(
                SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            ),
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
