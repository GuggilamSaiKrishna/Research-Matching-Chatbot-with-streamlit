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


def _calculate_similarity(score: float) -> float:
    # Chroma squared L2 distance on normalized embeddings ranges [0, 2]
    cos_sim = max(0.0, 1.0 - (score / 2.0))
    return round(cos_sim * 100, 2)


def retrieve_faculty(query, k=5):
    db = get_db()
    total = db._collection.count()
    if total == 0:
        return []

    fetch_k = min(total, max(20, (k * 4) if k else total))
    results = db.similarity_search_with_score(query, k=fetch_k or total)

    clean_query = query.strip().lower()

    seen = {}
    for doc, score in results:
        similarity = _calculate_similarity(score)
        name = doc.metadata.get("name", "Unknown")

        # Check for exact substring match in publications or research areas
        item_text = doc.metadata.get("item_text", "").lower()
        research_areas_text = doc.metadata.get("research_areas", "").lower()
        publications_text = doc.metadata.get("publications", "").lower()

        if (clean_query and (
            clean_query == item_text or
            clean_query in item_text or
            clean_query in publications_text or
            clean_query in research_areas_text
        )):
            similarity = 100.0

        full_profile = doc.metadata.get("full_profile") or doc.page_content

        if name not in seen or similarity > seen[name]["score"]:
            seen[name] = {
                "name": name,
                "department": doc.metadata.get("department", "N/A"),
                "mobile_number": doc.metadata.get("mobile_number", "N/A"),
                "research_areas": doc.metadata.get("research_areas", "N/A"),
                "publications": doc.metadata.get("publications", "N/A"),
                "score": similarity,
                "content": full_profile,
            }

    matches = list(seen.values())
    matches.sort(key=lambda match: match["score"], reverse=True)
    return matches[:k] if k else matches

