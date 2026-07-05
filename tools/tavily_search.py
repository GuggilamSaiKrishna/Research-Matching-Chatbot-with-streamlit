import os

from tavily import TavilyClient

import config  # loads .env from project root
from config import get_tavily_api_key

client = TavilyClient(api_key=get_tavily_api_key())


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