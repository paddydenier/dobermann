from dobermann.segmenters.abstract import SegmentationResult
from sentence_transformers import SentenceTransformer

from ...embeddings import Embedder, SentenceTransformerEmbedder
from ...preprocessors import PreProcessor, IdentityPreProcessor
from .boundaries import AdaptiveValleyBoundaryDetector, BoundaryDetector
from .postprocessor import BoundaryToLengthProcessor, PostProcessor
from .similarity import CosineSimilarity, Similarity
from .smoothing import MovingAverageSmoother, Smoother
from .texttiling_embeddings import TextTilingEmbeddings


class TextTiling(TextTilingEmbeddings):
    def __init__(
        self,
        pre_processor: PreProcessor | None = None,
        embedder: Embedder | None = None,
        similarity: Similarity | None = None,
        smoother: Smoother | None = None,
        boundary: BoundaryDetector | None = None,
        post_procesor: PostProcessor | None = None,
    ):
        self.pre_processor = pre_processor or IdentityPreProcessor()
        super().__init__(
            embedder=embedder
            or SentenceTransformerEmbedder(
                SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            ),
            similarity=similarity or CosineSimilarity(),
            smoother=smoother or MovingAverageSmoother(),
            boundary=(boundary or AdaptiveValleyBoundaryDetector()),
            post_procesor=(post_procesor or BoundaryToLengthProcessor()),
        )

        def segment(self, input) -> SegmentationResult:
            sentences = self.pre_processor.process(input)
            # TODO: add output processor
            return super().segment(sentences)
