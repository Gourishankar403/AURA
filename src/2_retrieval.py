import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
VECTOR_DIR = BASE_DIR / "data" / "vector_store"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

class AURARetriever:
    def __init__(self):
        print("Loading AURA Databases...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.faiss_index = faiss.read_index(str(VECTOR_DIR / "lactmed_faiss.index"))
        
        with open(VECTOR_DIR / "chunk_mapping.json", "r") as f:
            self.chunk_mapping = {int(k): v for k, v in json.load(f).items()}
            
        with open(PROCESSED_DIR / "lactmed_exact.json", "r") as f:
            self.guardrail_db = json.load(f)

    def search_semantic(self, query, top_k=2):
        vector = self.model.encode([query])
        distances, indices = self.faiss_index.search(np.array(vector).astype('float32'), top_k)
        
        results = []
        for idx in indices[0]:
            if idx != -1:
                results.append(self.chunk_mapping[idx])
        return results

    def get_guardrail_data(self, drug_name):
        return self.guardrail_db.get(drug_name, None)

if __name__ == '__main__':
    retriever = AURARetriever()
    print("\n--- Testing Retrieval ---")
    
    # 1. Test Semantic Search
    query = "Is Sertraline safe for breastfeeding mothers?"
    print(f"Query: {query}")
    print("Semantic Results:")
    for res in retriever.search_semantic(query):
        print(f" - {res}")
        
    # 2. Test Guardrail Lookup
    drug = "Sertraline"
    print(f"\nExact Guardrail Data for {drug}:")
    print(retriever.get_guardrail_data(drug))
