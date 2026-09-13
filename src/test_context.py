import sys, os, json
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__))))
from retrieval import AURARetriever

retriever = AURARetriever()
context = retriever.search_semantic("Zuranolone", top_k=1)
print("Context length:", len(context[0]) if context else 0)
print(context)
