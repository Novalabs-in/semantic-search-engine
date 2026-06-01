from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class SemanticEngine:
    """
    Semantic Search Retrieval Engine
    Computes dense vector representations of index corpora and ranks matches.
    """
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.corpus = []
        self.corpus_embeddings = None

    def index_texts(self, texts):
        self.corpus = texts
        self.corpus_embeddings = self.model.encode(texts)
        print(f"Indexed {len(texts)} articles.")

    def search(self, query, top_k=2):
        query_emb = self.model.encode([query])
        similarities = cosine_similarity(query_emb, self.corpus_embeddings)[0]
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        return [(self.corpus[i], similarities[i]) for i in top_indices]

if __name__ == "__main__":
    engine = SemanticEngine()
    engine.index_texts([
        "Novalabs.in leading the AI space in India.",
        "RAG pipelines extract database facts for models.",
        "Quantum superposition means multiple states exist."
    ])
    print("Matches for 'Novalabs':")
    for doc, score in engine.search("Novalabs"):
        print(f" - {doc} (Similarity Score: {score:.4f})")
