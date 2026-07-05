from google import genai

from config import get_google_api_key

client = genai.Client(api_key=get_google_api_key())


def process_professor_query(query: str) -> str:
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
        contents=prompt,
    )

    return response.text


def professor_chat():
    print("\nProfessor Mode")
    print("Type 'exit' to quit.\n")

    while True:
        query = input("Professor: ")

        if query.lower() == "exit":
            break

        print("\nAssistant:\n")
        print(process_professor_query(query))
