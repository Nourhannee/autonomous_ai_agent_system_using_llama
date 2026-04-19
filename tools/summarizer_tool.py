"""
Tool: Summarizer
Summarizes long text using the local Llama model via Ollama.
"""

import os
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

_llm = None

def _get_llm():
    global _llm
    if _llm is None:
        model = os.getenv("LLAMA_MODEL", "llama3.2:3b")
        _llm  = ChatOllama(model=model, temperature=0.1)
    return _llm


@tool
def summarizer_tool(text: str) -> str:
    """
    Summarize a long block of text into bullet points using Llama.
    Input: the raw text to summarize.
    Use this after FileReader or WebSearch when content is too long.
    """
    text = text.strip()
    if not text:
        return "Error: No text provided."
    if len(text) < 100:
        return f"Text too short to summarize: {text}"
    if len(text) > 8000:
        text = text[:8000] + "\n... [truncated]"

    prompt = (
        "Summarize the following text in clear bullet points.\n"
        "Focus on key facts and conclusions only.\n\n"
        f"TEXT:\n{text}\n\nSUMMARY:"
    )
    try:
        llm    = _get_llm()
        result = llm.invoke([HumanMessage(content=prompt)])
        return f"Summary:\n{result.content.strip()}"
    except Exception as e:
        return f"Summarization failed: {e}"