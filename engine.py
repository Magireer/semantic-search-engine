import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional

class SemanticSearchEngine:
    """A vector-based search engine using FAISS and Sentence-Transformers."""
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.index: Optional[faiss.Index] = None
        self.documents: List[str] = []

    def build_index(self, documents: List[str]):
        self.documents = documents
        embeddings = self.model.encode(documents)
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings.astype("float32"))
        print(f"Index built with {len(documents)} documents.")

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        if self.index is None:
            raise ValueError("Index must be built before searching.")
        
        query_embedding = self.model.encode([query])
        distances, indices = self.index.search(query_embedding.astype("float32"), k)
        
        results = []
        for i in range(len(indices[0])):
            idx = indices[0][i]
            results.append({
                "document": self.documents[idx],
                "distance": float(distances[0][i]),
                "index": int(idx)
            })
        return results

    def save_index(self, path: str):
        if self.index:
            faiss.write_index(self.index, path)

    def load_index(self, path: str, documents: List[str]):
        self.index = faiss.read_index(path)
        self.documents = documents
