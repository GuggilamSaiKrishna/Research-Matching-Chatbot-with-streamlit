import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from config import CHROMA_DIR, FACULTY_JSON, get_google_api_key

HASH_FILE = Path(CHROMA_DIR) / ".data_hash"


def _get_data_hash() -> str:
    if not Path(FACULTY_JSON).exists():
        return ""
    with open(FACULTY_JSON, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _build_documents():
    if not Path(FACULTY_JSON).exists():
        return []

    with open(FACULTY_JSON, "r", encoding="utf-8") as f:
        faculty = json.load(f)

    documents = []
    for prof in faculty:
        name = prof.get("name", "Unknown").strip()
        department = prof.get("department", "N/A").strip()
        mobile = prof.get("mobile_number") or prof.get("mobile") or "N/A"
        research_areas = prof.get("research_areas", [])
        publications = prof.get("publications", [])

        full_profile = f"""Name: {name}
Department: {department}
Mobile Number: {mobile}
Research Areas: {', '.join(research_areas)}
Publications: {', '.join(publications)}"""

        base_metadata = {
            "name": name,
            "department": department,
            "mobile_number": str(mobile).strip(),
            "research_areas": ", ".join(research_areas),
            "publications": ", ".join(publications),
            "full_profile": full_profile,
        }

        # 1. Chunk per publication
        for pub in publications:
            documents.append(
                Document(
                    page_content=f"Publication: {pub}",
                    metadata={**base_metadata, "chunk_type": "publication", "item_text": pub},
                )
            )

        # 2. Chunk per research area
        for area in research_areas:
            documents.append(
                Document(
                    page_content=f"Research Area: {area}",
                    metadata={**base_metadata, "chunk_type": "research_area", "item_text": area},
                )
            )

        # 3. Combined research summary chunk (free of non-research metadata noise)
        summary_text = f"Research Areas: {', '.join(research_areas)}. Publications: {', '.join(publications)}."
        documents.append(
            Document(
                page_content=summary_text,
                metadata={**base_metadata, "chunk_type": "summary", "item_text": summary_text},
            )
        )

    return documents


def chroma_is_ready() -> bool:
    return chroma_is_up_to_date()


def chroma_is_up_to_date() -> bool:
    if not Path(CHROMA_DIR).exists() or not HASH_FILE.exists():
        return False

    stored_hash = HASH_FILE.read_text(encoding="utf-8").strip()
    if stored_hash != _get_data_hash():
        return False

    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model="gemini-embedding-001",
            google_api_key=get_google_api_key(),
        )
        import chromadb
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        db = Chroma(client=client, collection_name="langchain", embedding_function=embeddings)
        return db._collection.count() > 0
    except Exception:
        return False


def load_faculty_data(rebuild: bool = False) -> bool:
    current_hash = _get_data_hash()

    if not rebuild and chroma_is_up_to_date():
        return False

    Path(CHROMA_DIR).mkdir(parents=True, exist_ok=True)

    documents = _build_documents()
    if documents:
        embeddings = GoogleGenerativeAIEmbeddings(
            model="gemini-embedding-001",
            google_api_key=get_google_api_key(),
        )
        import chromadb
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        try:
            client.delete_collection("langchain")
        except Exception:
            pass

        db = Chroma(
            client=client,
            collection_name="langchain",
            embedding_function=embeddings,
        )

        batch_size = 20
        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            max_retries = 5
            for attempt in range(max_retries):
                try:
                    db.add_documents(batch)
                    break
                except Exception as e:
                    if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                        wait_time = 25 * (attempt + 1)
                        print(f"Rate limited. Waiting {wait_time}s before retrying batch {i // batch_size + 1}...")
                        time.sleep(wait_time)
                    else:
                        raise e
            time.sleep(1)

    HASH_FILE.write_text(current_hash, encoding="utf-8")
    return True


def ensure_chroma_loaded(force: bool = False) -> bool:
    return load_faculty_data(rebuild=force)


if __name__ == "__main__":
    reloaded = load_faculty_data(rebuild=True)
    print("Faculty profiles loaded successfully!")

