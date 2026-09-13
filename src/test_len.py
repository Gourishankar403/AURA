import json
with open('C:/Users/gouri/Downloads/]RAG_POST/AURA_Project/data/vector_store/chunk_mapping.json', 'r') as f:
    mapping = json.load(f)
    for i, text in list(mapping.items())[:5]:
        print(f"Chunk {i} length: {len(text)} characters")
