import os
import groq
from dotenv import load_dotenv

load_dotenv('C:/Users/gouri/Downloads/]RAG_POST/AURA_Project/.env')
client = groq.Groq(api_key=os.getenv("GROQ_API_KEY"))

# JSON mode failing for all - test plain text mode instead
MODELS = [
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
    "qwen/qwen3.8-27b",
    "groq/compound-mini",
]

print("Testing plain text mode:")
for m in MODELS:
    try:
        completion = client.chat.completions.create(
            model=m,
            messages=[
                {"role": "user", "content": "In 2 sentences: Is Sertraline safe for breastfeeding?"}
            ],
            max_tokens=60,
            temperature=0.0
        )
        print(f"OK  [{m}] -> {completion.choices[0].message.content[:80]}")
    except Exception as e:
        print(f"ERR [{m}] -> {str(e)[:120]}")
