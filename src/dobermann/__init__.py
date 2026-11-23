# Public facing API
from .document import Document
from .embeddings import Embedder, SentenceTransformerEmbedder
from .evaluators import EvaluationResult, SegmentationEvaluator
from .preprocessors import IdentityPreProcessor, PreProcessor
from .segmenters import GraphSegEmbeddings, SegmentationResult, TextTilingEmbeddings
from .segmenters.base import Segmenter
from .segmenters.graphseg import GraphSeg
from .segmenters.graphseg.graph import GraphBuilder, WeightedGraphBuilder
from .segmenters.graphseg.similarity_matrix import (
    CosineSimilarityMatrix,
    SimilarityMatrix,
)
from .segmenters.texttiling import TextTiling

# import boundary
from .segmenters.texttiling.boundaries import (
    AdaptiveValleyBoundaryDetector,
    BoundaryDetector,
)
from .segmenters.texttiling.postprocessor import (
    BoundaryPostProcessor,
    BoundaryToLengthProcessor,
)

# import similarity
from .segmenters.texttiling.similarity import CosineSimilarity, Similarity

# import smoothing
from .segmenters.texttiling.smoothing import MovingAverageSmoother, SignalSmoother

__all__ = [
    "Document",
    "TextTiling",
    "TextTilingEmbeddings",
    "GraphSegEmbeddings",
    "SegmentationEvaluator",
    "EvaluationResult",
    "SegmentationResult",
    "SentenceTransformerEmbedder",
    "Embedder",
    "MovingAverageSmoother",
    "SignalSmoother",
    "Similarity",
    "CosineSimilarity",
    "AdaptiveValleyBoundaryDetector",
    "BoundaryDetector",
    "BoundaryPostProcessor",
    "BoundaryToLengthProcessor",
    "SimilarityMatrix",
    "CosineSimilarityMatrix",
    "GraphBuilder",
    "WeightedGraphBuilder",
    "GraphSeg",
    "Segmenter",
    "PreProcessor",
    "IdentityPreProcessor",
]
