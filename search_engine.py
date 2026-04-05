import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

class SemanticSearch:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.documents = []

    def build_index(self, documents):
        self.documents = documents
        embeddings = self.model.encode(documents)
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings.astype("float32"))
        print(f"Index built with {len(documents)} documents.")

    def search(self, query, k=5):
        query_embedding = self.model.encode([query])
        distances, indices = self.index.search(query_embedding.astype("float32"), k)
        results = []
        for i in range(len(indices[0])):
            idx = indices[0][i]
            results.append({
                "document": self.documents[idx],
                "distance": float(distances[0][i])
            })
        return results

if __name__ == "__main__":
    docs = [
        "Artificial intelligence is transforming the world.",
        "Machine learning is a subset of AI.",
        "Deep learning uses neural networks.",
        "Python is a popular language for data science.",
        "FAISS is a library for efficient similarity search."
    ]
    engine = SemanticSearch()
    engine.build_index(docs)
    results = engine.search("How does AI impact technology?")
    for res in results:
        print(res)
