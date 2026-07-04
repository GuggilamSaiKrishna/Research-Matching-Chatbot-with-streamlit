import os
from dotenv import load_dotenv

from google import genai
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# ----------------------------
# Load Environment Variables
# ----------------------------
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# ----------------------------
# Gemini Client
# ----------------------------
client = genai.Client(api_key=GOOGLE_API_KEY)

# ----------------------------
# Embedding Model
# ----------------------------
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    google_api_key=GOOGLE_API_KEY
)

# ----------------------------
# Load ChromaDB
# ----------------------------
db = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings
)

print("=" * 60)
print("🎓 Research Matching Chatbot")
print("Type 'exit' to quit")
print("=" * 60)

while True:

    query = input("\nStudent : ")

    if query.lower() == "exit":
        break

    # Retrieve top 3 matching faculty
    results = db.similarity_search_with_score(query, k=3)

    if not results:
        print("No matching faculty found.")
        continue

    context = ""

    print("\nTop Matches\n")

    for i, (doc, score) in enumerate(results, start=1):

        similarity = round((1 / (1 + score)) * 100, 2)

        print(f"{i}. {doc.metadata['name']} ({similarity}% Match)")

        context += doc.page_content + "\n\n"

    prompt = f"""
You are a university research assistant.

Use ONLY the faculty information below.

Faculty Information:
{context}

Student Question:
{query}

Answer in a friendly and concise way.
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    print("\nAssistant:\n")
    print(response.text)