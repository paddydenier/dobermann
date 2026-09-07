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
from dobermann.segmenters.graphseg.labeling.community import CommunityLabeler

# 1. Load the model from Hugging Face
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
embedder = SentenceTransformerEmbedder(model)
similarity = CosineSimilarityMatrix()
graph = WeightedGraphBuilder()
community = GreedyModularityCommunityDetector()
labeler = CommunityLabeler()

segmenter = GraphSegEmbeddings(embedder, similarity, graph, community, labeler)
sentences = [
    "This is a sentence.",
]


segmentation_result = segmenter.segment(sentences)

chunks = segmentation_result.split(sentences)

print(chunks)
