import json
from dotenv import load_dotenv
import os

from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

# Load environment variables
load_dotenv()

# Embedding model
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# Read faculty data
with open("data/faculty_profiles/faculty.json", "r", encoding="utf-8") as f:
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
            metadata={"name": prof["name"]}
        )
    )

# Store in ChromaDB
db = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    persist_directory="chroma_db"
)

print("Faculty profiles loaded successfully!")