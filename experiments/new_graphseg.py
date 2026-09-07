from sentence_transformers import SentenceTransformer

from dobermann import (
    CosineSimilarityMatrix,
    GraphSegEmbeddings,
    SentenceTransformerEmbedder,
    WeightedGraphBuilder,
)
from dobermann.segmenters.graphseg.community.greedy_modularity import (
    GreedyModularityCommunityDetector,
)

# 1. Load the model from Hugging Face
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
embedder = SentenceTransformerEmbedder(model)
similarity = CosineSimilarityMatrix()
graph = WeightedGraphBuilder()
community = GreedyModularityCommunityDetector()

segmenter = GraphSegEmbeddings(embedder, similarity, graph, community)
sentences = [
    "This is a sentence.",
]


segmentation_result = segmenter.segment(sentences)

chunks = segmentation_result.split(sentences)

print(chunks)
