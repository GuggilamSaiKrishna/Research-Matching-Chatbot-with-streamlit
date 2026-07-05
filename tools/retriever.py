from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from config import CHROMA_DIR, get_google_api_key

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    google_api_key=get_google_api_key(),
)

db = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function=embeddings,
)


def retrieve_faculty(query, k=None):
    total = db._collection.count()
    if total == 0:
        return []

    results = db.similarity_search_with_score(query, k=k or total)

    seen = {}
    for doc, score in results:
        similarity = round((1 / (1 + score)) * 100, 2)
        name = doc.metadata.get("name", "Unknown")

        if name not in seen or similarity > seen[name]["score"]:
            seen[name] = {
                "name": name,
                "department": doc.metadata.get("department", "N/A"),
                "research_areas": doc.metadata.get("research_areas", "N/A"),
                "score": similarity,
                "content": doc.page_content,
            }

    matches = list(seen.values())
    matches.sort(key=lambda match: match["score"], reverse=True)
    return matches
