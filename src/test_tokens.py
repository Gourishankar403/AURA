import os
import groq
from dotenv import load_dotenv

load_dotenv('C:/Users/gouri/Downloads/]RAG_POST/AURA_Project/.env')
client = groq.Groq(api_key=os.getenv("GROQ_API_KEY"))

prompt_str = "A 28-year-old breastfeeding mother with severe PPD is starting Zuranolone. Based strictly on LactMed, what is the exact Relative Infant Dose (RID) percentage?"
context_str = "['Drug: Zuranolone. Pharmacokinetics and Safety: Zuranolone is a neuroactive steroid GABA-A receptor positive modulator indicated for the treatment of postpartum depression. It is taken orally once daily for 14 days. Caution is advised as long-term infant effects are still being studied, and sedation can occur.']"

for tokens in [100, 200, 300, 500]:
    try:
        completion = client.chat.completions.create(
            model="groq/compound",
            messages=[
                {"role": "system", "content": f"You are AURA. Read the clinical vignette and LactMed context. Provide clinical reasoning ONLY. JSON format: 'drug_name', 'clinical_reasoning'. CONTEXT: {context_str}"},
                {"role": "user", "content": f"Patient Vignette: {prompt_str}\nTarget Drug: Zuranolone"}
            ],
            response_format={"type": "json_object"},
            max_tokens=tokens,
            temperature=0.0
        )
        print(f"Success with max_tokens={tokens}!")
    except Exception as e:
        print(f"Error with max_tokens={tokens}: {e}")
