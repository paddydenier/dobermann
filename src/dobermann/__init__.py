# Public facing API
from .document import Document
from .embeddings import Embedder, SentenceTransformerEmbedder
from .evaluators import EvaluationResult, SegmentationEvaluator
from .preprocessors import IdentityPreProcessor, PreProcessor
from .segmenters import GraphSegEmbeddings, SegmentationResult, TextTilingEmbeddings
from .segmenters.abstract import Segmenter
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
    BoundaryToLengthProcessor,
    PostProcessor,
)

# import similarity
from .segmenters.texttiling.similarity import CosineSimilarity, Similarity

# import smoothing
from .segmenters.texttiling.smoothing import MovingAverageSmoother, Smoother

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
    "Smoother",
    "Similarity",
    "CosineSimilarity",
    "AdaptiveValleyBoundaryDetector",
    "BoundaryDetector",
    "PostProcessor",
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
