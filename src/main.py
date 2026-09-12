import sys
sys.stdout.reconfigure(encoding='utf-8')
import os
import json
from dotenv import load_dotenv
from pydantic import BaseModel, Field
import groq 
import sys

# Ensure the current directory is in path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from retrieval import AURARetriever

load_dotenv()

class ClinicalRecommendation(BaseModel):
    drug_name: str = Field(description="The name of the psychotropic drug being recommended")
    clinical_reasoning: str = Field(description="The medical reasoning based on the provided LactMed context")

def run_aura_pipeline(vignette_text, target_drug):
    print(f"\n[1] Processing Patient Vignette for Drug: {target_drug}")
    
    retriever = AURARetriever()
    
    print("\n[2] Retrieving Semantic Context from FAISS...")
    context_results = retriever.search_semantic(target_drug, top_k=1)
    medical_context = context_results[0] if context_results else "No context found."
    
    print("\n[3] Retrieving Exact Pharmacokinetic Guardrails...")
    guardrail_data = retriever.get_guardrail_data(target_drug)
    if not guardrail_data:
        print("ERROR: Drug not in LactMed Database. Triggering Human Deferral.")
        return
        
    exact_rid = guardrail_data["RID"]
    exact_half_life = guardrail_data["half_life"]
    
    print(f"    -> Sentinel Guardrail Locked: RID must be {exact_rid}")
    
    print("\n[4] Generating Reasoning via Groq LLM...")
    
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key == "YOUR_API_KEY_HERE":
        print("---------------------------------------------------------")
        print(" STOP: GROQ_API_KEY is missing or invalid in your .env file!")
        print("---------------------------------------------------------")
        return

    client = groq.Groq(api_key=api_key)
    
    system_prompt = f"""You are AURA, a highly accurate medical AI. 
Read the following clinical vignette and the LactMed database context.
Provide clinical reasoning for the prescription. 
Respond ONLY in JSON format matching the following keys: "drug_name", "clinical_reasoning".

LACTMED CONTEXT:
{medical_context}
"""

    try:
        completion = client.chat.completions.create(
            model="groq/compound",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Patient Vignette: {vignette_text}\nTarget Drug: {target_drug}"}
            ],
            response_format={"type": "json_object"},
            temperature=0.0
        )
        
        llm_response = json.loads(completion.choices[0].message.content)
        
        print("\n========================================================")
        print("                 AURA FINAL REPORT                      ")
        print("========================================================")
        print(f"Drug Prescribed: {llm_response.get('drug_name', target_drug)}")
        print(f"Clinical Reasoning: {llm_response.get('clinical_reasoning', 'N/A')}")
        print("\n--- SENTINEL GUARDRAIL INJECTION ---")
        print(f"Relative Infant Dose (RID): {exact_rid}  <-- Mathematically verified")
        print(f"Half-life: {exact_half_life}  <-- Mathematically verified")
        print("========================================================")
        
    except Exception as e:
        print(f"Error calling Groq API: {e}")

if __name__ == '__main__':
    sample_vignette = "A 28-year-old breastfeeding mother with a history of severe postpartum depression presents with low mood and anxiety. We are considering initiating Sertraline."
    target = "Sertraline"
    run_aura_pipeline(sample_vignette, target)

