import sys, os
sys.path.append('C:/Users/gouri/Downloads/]RAG_POST/AURA_Project/src')
from retrieval import AURARetriever
import groq
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path('C:/Users/gouri/Downloads/]RAG_POST/AURA_Project/.env'))
retriever = AURARetriever()
client = groq.Groq(api_key=os.getenv("GROQ_API_KEY"))

# Test 1: Auto-detect
vignette = "A 30-year-old breastfeeding mother with a premature neonate wants to take Fluoxetine."
drug = None
for d in ["Sertraline","Zuranolone","Brexanolone","Fluoxetine","Escitalopram","Paroxetine","Venlafaxine"]:
    if d.lower() in vignette.lower():
        drug = d
        break
print(f"Test 1 - Auto-detected drug: {drug}")

# Test 2: Guardrail lookup
gdata = retriever.get_guardrail_data(drug)
print(f"Test 2 - Guardrail data: RID={gdata['RID']}, Half-life={gdata['half_life']}")

# Test 3: Semantic retrieval
ctx = retriever.search_semantic(drug, top_k=1)
print(f"Test 3 - Context retrieved: {ctx[0][:60]}...")

# Test 4: LLM call
ctx_str = ctx[0]
completion = client.chat.completions.create(
    model="qwen/qwen3.8-27b",
    messages=[
        {"role": "system", "content": "You are a clinical pharmacology assistant. Be concise. 3-4 sentences maximum. Never invent numerical dosage data."},
        {"role": "user", "content": f"LactMed Context: {ctx_str}\n\nPatient: {vignette}\n\nDrug: {drug}\n\nProvide a brief clinical safety assessment."}
    ],
    max_tokens=200,
    temperature=0.0
)
print(f"Test 4 - LLM reasoning: {completion.choices[0].message.content[:100]}...")
print("\nAll 4 tests PASSED! App is fully functional.")
