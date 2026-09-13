import os
import groq
from dotenv import load_dotenv

load_dotenv('C:/Users/gouri/Downloads/]RAG_POST/AURA_Project/.env')
client = groq.Groq(api_key=os.getenv("GROQ_API_KEY"))

MODELS = [
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
    "groq/compound-mini",
    "groq/compound",
    "qwen/qwen3.8-27b",
]

print("Testing models for JSON output under tight token budgets:")
for m in MODELS:
    try:
        completion = client.chat.completions.create(
            model=m,
            messages=[
                {"role": "system", "content": 'Output JSON with key "reasoning".'},
                {"role": "user", "content": "Is Sertraline safe while breastfeeding?"}
            ],
            response_format={"type": "json_object"},
            max_tokens=80,
            temperature=0.0
        )
        print(f"OK  [{m}] -> {completion.choices[0].message.content[:60]}")
    except Exception as e:
        print(f"ERR [{m}] -> {str(e)[:100]}")
