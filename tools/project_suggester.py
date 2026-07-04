import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GOOGLE_API_KEY")
)

def suggest_project(context, interest):

    prompt = f"""
You are an AI Research Guide.

Faculty Research:
{context}

Student Interest:
{interest}

Suggest:
1. Three project ideas
2. Difficulty (Easy/Medium/Hard)
3. Required Skills
4. Expected Outcome.

Keep the answer short and clear.
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return response.text