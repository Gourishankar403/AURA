import os
import groq
from dotenv import load_dotenv

load_dotenv('C:/Users/gouri/Downloads/]RAG_POST/AURA_Project/.env')
client = groq.Groq(api_key=os.getenv("GROQ_API_KEY"))

# qwen/qwen3.8-27b works in plain text. Test with larger max_tokens and real AURA prompt.
context = "Zuranolone is a neuroactive steroid GABA-A receptor positive modulator for PPD. Oral, 14-day course. RID is approximately 0.74%."
vignette = "A 28-year-old breastfeeding mother with a healthy newborn is starting Zuranolone for severe PPD."

prompt = f"CONTEXT: {context}\n\nVIGNETTE: {vignette}\n\nIn 3 sentences, explain if Zuranolone is clinically appropriate for this patient."

try:
    completion = client.chat.completions.create(
        model="qwen/qwen3.8-27b",
        messages=[
            {"role": "system", "content": "You are a clinical pharmacology assistant. Be brief and clinical."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=150,
        temperature=0.0
    )
    print("SUCCESS!")
    print(completion.choices[0].message.content)
except Exception as e:
    print(f"Error: {e}")
