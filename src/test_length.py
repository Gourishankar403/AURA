import sys, os, json
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__))))
from retrieval import AURARetriever

retriever = AURARetriever()
for drug in ["Sertraline", "Zuranolone", "Brexanolone", "Fluoxetine", "Escitalopram", "Paroxetine", "Venlafaxine"]:
    results = retriever.search_semantic(drug, top_k=1)
    if results:
        print(f"{drug} chunk length: {len(results[0])} characters")
