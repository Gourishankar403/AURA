import os
import groq
from dotenv import load_dotenv

load_dotenv('C:/Users/gouri/Downloads/]RAG_POST/AURA_Project/.env')
client = groq.Groq(api_key=os.getenv("GROQ_API_KEY"))

try:
    models = client.models.list()
    print("Available Groq Models:")
    for m in models.data:
        print(f"- {m.id}")
except Exception as e:
    print(f"Error: {e}")
