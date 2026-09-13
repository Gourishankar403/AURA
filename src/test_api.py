import os
import groq
from dotenv import load_dotenv

load_dotenv('C:/Users/gouri/Downloads/]RAG_POST/AURA_Project/.env')
client = groq.Groq(api_key=os.getenv("GROQ_API_KEY"))

try:
    completion = client.chat.completions.create(
        model="groq/compound",
        messages=[
            {"role": "system", "content": "You are AURA. Output JSON."},
            {"role": "user", "content": "Test"}
        ],
        response_format={"type": "json_object"},
        max_tokens=100
    )
    print("Success!")
    print(completion.choices[0].message.content)
except Exception as e:
    print(f"Error: {e}")
