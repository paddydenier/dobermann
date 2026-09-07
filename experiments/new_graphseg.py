from sentence_transformers import SentenceTransformer

from dobermann import (
    CosineSimilarityMatrix,
    GraphSegEmbeddings,
    SentenceTransformerEmbedder,
)

# 1. Load the model from Hugging Face
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
embedder = SentenceTransformerEmbedder(model)
similarity = CosineSimilarityMatrix()

segmenter = GraphSegEmbeddings(embedder, similarity)
sentences = [
    "This is a sentence.",
]


segmentation_result = segmenter.segment(sentences)

chunks = segmentation_result.split(sentences)

print(chunks)
