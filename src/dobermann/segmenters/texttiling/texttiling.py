from sentence_transformers import SentenceTransformer

from ...embeddings import Embedder, SentenceTransformerEmbedder
from .boundaries import AdaptiveValleyBoundaryDetector, BoundaryDetector
from .postprocessor import BoundaryToLengthProcessor, PostProcessor
from .similarity import CosineSimilarity, Similarity
from .smoothing import MovingAverageSmoother, Smoother
from .texttiling_embeddings import TextTilingEmbeddings


class TextTiling(TextTilingEmbeddings):
    def __init__(
        self,
        embedder: Embedder | None = None,
        similarity: Similarity | None = None,
        smoother: Smoother | None = None,
        boundary: BoundaryDetector | None = None,
        post_procesor: PostProcessor | None = None,
    ):
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
