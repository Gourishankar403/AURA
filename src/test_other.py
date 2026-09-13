import os
import groq
from dotenv import load_dotenv

load_dotenv('C:/Users/gouri/Downloads/]RAG_POST/AURA_Project/.env')
client = groq.Groq(api_key=os.getenv("GROQ_API_KEY"))

prompt = "A 28-year-old breastfeeding mother with severe PPD is starting Zuranolone."
context = "Zuranolone is a medication."
system_prompt = f"You are AURA. Read the clinical vignette and LactMed context. Provide clinical reasoning ONLY. JSON format: 'drug_name', 'clinical_reasoning'. CONTEXT: {context}"

for m in ["qwen/qwen3.8-27b", "openai/gpt-oss-20b"]:
    try:
        completion = client.chat.completions.create(
            model=m,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.0
        )
        print(f"Success with {m}!")
    except Exception as e:
        print(f"Failed with {m}: {e}")
