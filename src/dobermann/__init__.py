# Public facing API
from .document import Document
from .embeddings import Embedder, SentenceTransformerEmbedder
from .evaluators import EvaluationResult, SegmentationEvaluator
from .segmenters import GraphSegEmbeddings, SegmentationResult, TextTilingEmbeddings

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
    "SentenceTransformerEmbedder",
    "MovingAverageSmoother",
    "Smoother",
    "Similarity",
    "CosineSimilarity",
]
