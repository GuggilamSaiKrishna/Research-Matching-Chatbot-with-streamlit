import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from config import CHROMA_DIR, FACULTY_JSON, get_google_api_key


def _build_documents():
    with open(FACULTY_JSON, "r", encoding="utf-8") as f:
        faculty = json.load(f)

    documents = []
    for prof in faculty:
        text = f"""
Name: {prof['name']}
Department: {prof['department']}
Research Areas: {', '.join(prof['research_areas'])}
Publications: {', '.join(prof['publications'])}
"""
        documents.append(
            Document(
                page_content=text,
                metadata={
                    "name": prof["name"],
                    "department": prof["department"],
                    "research_areas": ", ".join(prof["research_areas"]),
                },
            )
        )
    return documents


def chroma_is_ready() -> bool:
    if not Path(CHROMA_DIR).exists():
        return False

    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        google_api_key=get_google_api_key(),
    )
    db = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
    return db._collection.count() > 0


def load_faculty_data(rebuild: bool = False) -> None:
    if rebuild:
        shutil.rmtree(CHROMA_DIR, ignore_errors=True)
    elif chroma_is_ready():
        return

    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        google_api_key=get_google_api_key(),
    )

    Chroma.from_documents(
        documents=_build_documents(),
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
    )


def ensure_chroma_loaded() -> None:
    load_faculty_data(rebuild=False)


if __name__ == "__main__":
    load_faculty_data(rebuild=True)
    print("Faculty profiles loaded successfully!")
