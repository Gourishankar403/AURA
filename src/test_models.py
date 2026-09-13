import os
import groq
from dotenv import load_dotenv

load_dotenv('C:/Users/gouri/Downloads/]RAG_POST/AURA_Project/.env')
client = groq.Groq(api_key=os.getenv("GROQ_API_KEY"))

try:
    completion = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[
            {"role": "system", "content": "You are AURA. Output JSON."},
            {"role": "user", "content": "Test"}
        ],
        response_format={"type": "json_object"},
        max_tokens=100
    )
    print("Success with llama3-70b-8192!")
except Exception as e:
    print(f"Error 1: {e}")

try:
    completion = client.chat.completions.create(
        model="llama-3.1-70b-versatile",
        messages=[
            {"role": "system", "content": "You are AURA. Output JSON."},
            {"role": "user", "content": "Test"}
        ],
        response_format={"type": "json_object"},
        max_tokens=100
    )
    print("Success with llama-3.1-70b-versatile!")
except Exception as e:
    print(f"Error 2: {e}")
