import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GOOGLE_API_KEY")
)

def professor_chat():

    print("\nProfessor Mode")
    print("Type 'exit' to quit.\n")

    while True:

        query = input("Professor: ")

        if query.lower() == "exit":
            break

        prompt = f"""
You are an AI research assistant for university professors.

Professor Question:
{query}

Answer with:
1. Current research trends
2. Recent advancements
3. Future research directions

Keep the answer concise.
"""

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        print("\nAssistant:\n")
        print(response.text)