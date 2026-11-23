from dobermann.embeddings import Embedder, SentenceTransformerEmbedder
from dobermann.preprocessors import IdentityPreProcessor, PreProcessor
from dobermann.segmenters.base import SegmentationResult, Segmenter

from .boundaries import AdaptiveValleyBoundaryDetector
from .postprocessor import BoundaryToLengthProcessor
from .similarity import CosineSimilarity
from .smoothing import MovingAverageSmoother
from .texttiling_embeddings import TextTilingEmbeddings


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
        *,
        embedder: Embedder | None = None,
    ) -> "TextTiling":
        """Create TextTiling with the default pipeline.

        Args:
            input_processor: Preprocessor applied to the input before
                segmentation. Defaults to `IdentityPreProcessor`.
            embedder: Embedder that turns sentences into vectors. Defaults to
                a new `SentenceTransformerEmbedder.default()`, which loads its
                own model. Passing the same embedder to several segmenters
                shares one model between them.
        """
        if input_processor is None:
            input_processor = IdentityPreProcessor()
        if embedder is None:
            embedder = SentenceTransformerEmbedder.default()

        algorithm = TextTilingEmbeddings(
            embedder=embedder,
            similarity=CosineSimilarity(),
            smoother=MovingAverageSmoother(),
            boundary=AdaptiveValleyBoundaryDetector(),
            postprocessor=BoundaryToLengthProcessor(),
        )

        return cls(
            input_processor=input_processor,
            algorithm=algorithm,
        )

    def segment(self, input) -> SegmentationResult:
        sentences = self.input_processor.process(input)
        return self.algorithm.segment(sentences)
