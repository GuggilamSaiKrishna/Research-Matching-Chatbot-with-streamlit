import chromadb
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from config import CHROMA_DIR, get_google_api_key
from tools.load_data import ensure_chroma_loaded


def get_db() -> Chroma:
    ensure_chroma_loaded()
    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        google_api_key=get_google_api_key(),
    )
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return Chroma(
        client=client,
        collection_name="langchain",
        embedding_function=embeddings,
    )


def retrieve_faculty(query, k=5):
    db = get_db()
    total = db._collection.count()
    if total == 0:
        return []

    fetch_k = min(total, (k * 2) if k else total)
    results = db.similarity_search_with_score(query, k=fetch_k or total)

    seen = {}
    for doc, score in results:
        similarity = round((1 / (1 + score)) * 100, 2)
        name = doc.metadata.get("name", "Unknown")

        if name not in seen or similarity > seen[name]["score"]:
            seen[name] = {
                "name": name,
                "department": doc.metadata.get("department", "N/A"),
                "mobile_number": doc.metadata.get("mobile_number", "N/A"),
                "research_areas": doc.metadata.get("research_areas", "N/A"),
                "score": similarity,
                "content": doc.page_content,
            }

    matches = list(seen.values())
    matches.sort(key=lambda match: match["score"], reverse=True)
    return matches[:k] if k else matches

