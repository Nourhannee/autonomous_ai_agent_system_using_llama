"""
Tool: Web Search
Uses DuckDuckGo (no API key required) to search the web.
"""

from langchain_core.tools import tool
from duckduckgo_search import DDGS


@tool
def web_search_tool(query: str) -> str:
    """
    Search the web using DuckDuckGo and return top results.
    Input: a search query string e.g. LangChain ReAct agent tutorial.
    Returns titles, URLs, and summaries of top 5 results.
    No API key required.
    """
    query = query.strip()
    if not query:
        return "Error: Empty search query."
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
        if not results:
            return f"No results found for: '{query}'."
        lines = [f"Search results for: '{query}'\n"]
        for i, r in enumerate(results, 1):
            lines.append(
                f"{i}. {r.get('title', 'No title')}\n"
                f"   URL: {r.get('href', '')}\n"
                f"   {r.get('body', 'No description')}\n"
            )
        return "\n".join(lines)
    except Exception as e:
        return f"Search failed: {e}"