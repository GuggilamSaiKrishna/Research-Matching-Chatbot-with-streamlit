import os
from dotenv import load_dotenv
from google import genai

from tools.retriever import retrieve_faculty
from tools.project_suggester import suggest_project
from tools.tavily_search import web_search
from tools.collaboration import collaboration_match

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GOOGLE_API_KEY")
)


def student_chat():

    print("\n========== Student Mode ==========")
    print("Type 'exit' to quit.\n")

    while True:

        query = input("Student: ")

        if query.lower() == "exit":
            print("Goodbye!")
            break

        # -----------------------------
        # Collaboration Matching
        # -----------------------------
        if "collaboration" in query.lower():

            print("\nSuggested Collaboration\n")
            print(collaboration_match(query))
            print()
            continue

        # -----------------------------
        # Web Search
        # -----------------------------
        if "latest" in query.lower() or "trend" in query.lower():

            search_result = web_search(query)

            prompt = f"""
You are a research assistant.

Answer using the web search results below.

Web Results:
{search_result}

Question:
{query}
"""

            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )

            print("\nAssistant:\n")
            print(response.text)
            print()

            continue

        # -----------------------------
        # Faculty Retrieval
        # -----------------------------
        matches = retrieve_faculty(query)

        if len(matches) == 0:

            print("\nNo matching faculty found.\n")
            continue

        print("\nTop Faculty Matches\n")

        context = ""

        for i, faculty in enumerate(matches, start=1):

            print(f"{i}. {faculty['name']} ({faculty['score']}% Match)")

            context += faculty["content"] + "\n\n"

        # -----------------------------
        # Human-in-the-loop
        # -----------------------------
        confirm = input(
            "\nDo you want to continue with these recommendations? (yes/no): "
        )

        if confirm.lower() != "yes":

            print("Recommendation cancelled.\n")
            continue

        # -----------------------------
        # Project Suggestion
        # -----------------------------
        if "project" in query.lower():

            projects = suggest_project(context, query)

            print("\nSuggested Projects\n")
            print(projects)
            print()

            continue

        # -----------------------------
        # Gemini Answer
        # -----------------------------
        prompt = f"""
You are a university research assistant.

Use ONLY the faculty information below.

Faculty Information:
{context}

Student Question:
{query}

Answer naturally.

If the answer is unavailable, reply:

I couldn't find that information in the faculty database.
"""

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        print("\nAssistant:\n")
        print(response.text)
        print()