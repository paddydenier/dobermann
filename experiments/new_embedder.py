from sentence_transformers import SentenceTransformer

from dobermann import SentenceTransformerEmbedder

# 1. Load the model from Hugging Face
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
embedder = SentenceTransformerEmbedder(model)
print(embedder.embed(["This is a test sentence and I like computer science"]))
