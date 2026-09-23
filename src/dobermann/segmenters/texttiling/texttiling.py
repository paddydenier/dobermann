from sentence_transformers import SentenceTransformer

from dobermann.segmenters.abstract import SegmentationResult

from ...embeddings import SentenceTransformerEmbedder
from ...preprocessors import IdentityPreProcessor, PreProcessor
from .boundaries import AdaptiveValleyBoundaryDetector
from .postprocessor import BoundaryToLengthProcessor
from .similarity import CosineSimilarity
from .smoothing import MovingAverageSmoother
from .texttiling_embeddings import TextTilingEmbeddings

from ..abstract import Segmenter


class TextTiling(Segmenter):
    """High-level facade for TextTiling topic segmentation."""

    def __init__(
        self,
        input_processor: PreProcessor,
        algorithm: TextTilingEmbeddings,
    ):
        self.input_processor = input_processor
        self.algorithm = algorithm

    @classmethod
    def default(
        cls,
        input_processor: PreProcessor | None = None,
    ) -> "TextTiling":
        if input_processor is None:
            input_processor = IdentityPreProcessor()

        algorithm = TextTilingEmbeddings(
            embedder=SentenceTransformerEmbedder(
                SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            ),
            similarity=CosineSimilarity(),
            smoother=MovingAverageSmoother(),
            boundary=AdaptiveValleyBoundaryDetector(),
            post_procesor=BoundaryToLengthProcessor(),
        )

        return cls(
            input_processor=input_processor,
            algorithm=algorithm,
        )

    def segment(self, input) -> SegmentationResult:
        sentences = self.input_processor.process(input)
        return self.algorithm.segment(sentences)
