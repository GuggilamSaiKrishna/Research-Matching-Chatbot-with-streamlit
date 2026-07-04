import os
from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv()

client = TavilyClient(
    api_key=os.getenv("TAVILY_API_KEY")
)


def web_search(query):

    response = client.search(
        query=query,
        search_depth="advanced",
        max_results=5
    )

    results = ""

    for item in response["results"]:
        results += f"""
Title: {item['title']}
Content: {item['content']}
URL: {item['url']}

"""

    return results