import json
import re
from pathlib import Path

import chromadb
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from config import CHROMA_DIR, FACULTY_JSON, get_google_api_key
from tools.load_data import ensure_chroma_loaded


def get_db() -> Chroma | None:
    try:
        ensure_chroma_loaded()
        api_key = get_google_api_key()
        if not api_key:
            return None
        embeddings = GoogleGenerativeAIEmbeddings(
            model="gemini-embedding-001",
            google_api_key=api_key,
        )
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        return Chroma(
            client=client,
            collection_name="langchain",
            embedding_function=embeddings,
        )
    except Exception as e:
        print(f"Warning: Could not connect to Chroma DB: {e}")
        return None


def _calculate_similarity(score: float) -> float:
    cos_sim = max(0.0, 1.0 - (score / 2.0))
    return round(cos_sim * 100, 2)


def fallback_retrieve_faculty(query: str, k: int = 5):
    if not Path(FACULTY_JSON).exists():
        return []

    with open(FACULTY_JSON, "r", encoding="utf-8") as f:
        faculty = json.load(f)

    query_tokens = set(re.findall(r"\w+", query.lower()))
    clean_query = query.strip().lower()

    scored = []
    for prof in faculty:
        name = prof.get("name", "Unknown")
        department = prof.get("department", "N/A")
        mobile = prof.get("mobile_number") or prof.get("mobile") or "N/A"
        research_areas = prof.get("research_areas", [])
        publications = prof.get("publications", [])

        full_profile = f"""Name: {name}
Department: {department}
Mobile Number: {mobile}
Research Areas: {', '.join(research_areas)}
Publications: {', '.join(publications)}"""

        research_text = " ".join(research_areas + publications).lower()
        prof_tokens = set(re.findall(r"\w+", research_text))

        is_exact = any(clean_query == pub.lower() or clean_query in pub.lower() for pub in publications) or \
                   any(clean_query == area.lower() or clean_query in area.lower() for area in research_areas)

        if is_exact:
            score = 100.0
        elif query_tokens and prof_tokens:
            common = query_tokens.intersection(prof_tokens)
            if common:
                score = round(min(95.0, (len(common) / len(query_tokens)) * 90.0), 2)
            else:
                score = 0.0
        else:
            score = 0.0

        if score > 0:
            scored.append({
                "name": name,
                "department": department,
                "mobile_number": str(mobile),
                "research_areas": ", ".join(research_areas),
                "publications": ", ".join(publications),
                "score": score,
                "content": full_profile,
            })

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:k] if k else scored


def retrieve_faculty(query: str, k: int = 5):
    try:
        db = get_db()
        if db is None:
            return fallback_retrieve_faculty(query, k=k)

        total = db._collection.count()
        if total == 0:
            return fallback_retrieve_faculty(query, k=k)

        fetch_k = min(total, (k * 2) if k else total)
        results = db.similarity_search_with_score(query, k=fetch_k or total)

        clean_query = query.strip().lower()

        seen = {}
        for doc, score in results:
            similarity = _calculate_similarity(score)
            name = doc.metadata.get("name", "Unknown")

            research_areas_text = doc.metadata.get("research_areas", "").lower()
            publications_text = doc.metadata.get("publications", "").lower()

            if clean_query and (
                clean_query in publications_text
                or clean_query in research_areas_text
            ):
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
        if matches:
            return matches[:k] if k else matches
        return fallback_retrieve_faculty(query, k=k)

    except Exception as e:
        print(f"Vector retrieval exception ({e}), falling back to direct search.")
        return fallback_retrieve_faculty(query, k=k)
