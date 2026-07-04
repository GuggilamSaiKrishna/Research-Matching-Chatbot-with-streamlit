import os
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

db = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings
)


def retrieve_faculty(query, k=3):
    results = db.similarity_search_with_score(query, k=k)

    matches = []

    for doc, score in results:
        similarity = round((1 / (1 + score)) * 100, 2)

        matches.append({
            "name": doc.metadata["name"],
            "score": similarity,
            "content": doc.page_content
        })

    return matches