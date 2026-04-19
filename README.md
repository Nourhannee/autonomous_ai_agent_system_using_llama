# Autonomous AI Agent System using Llama
**AIE 418 – Selected Topics in AI 2 · Application 2**

A production-grade autonomous agent that uses a **local Llama model** (via Ollama) and LangChain's ReAct framework to perform multi-step reasoning and task execution with dynamic tool selection, sliding-window memory, and structured output.

---

## Table of Contents
1. [System Overview](#system-overview)
2. [Project Structure](#project-structure)
3. [Setup Instructions](#setup-instructions)
4. [Required Dependencies](#required-dependencies)
5. [How to Run](#how-to-run)
6. [Architecture & Design](#architecture--design)
7. [Tools Reference](#tools-reference)
8. [Example Usage Scenarios](#example-usage-scenarios)
9. [Configuration](#configuration)
10. [Testing](#testing)
11. [Extending the System](#extending-the-system)

---

## System Overview

The agent implements the **ReAct (Reasoning + Acting)** pattern:

```
User Input
    │
    ▼
┌────────────────────────────────────┐
│         ReAct Agent Core           │
│   Llama 3.1 via Ollama (local)     │
│                                    │
│  Thought → Action → Observation    │  ← repeated up to 8 times
│       ↑__________________________|  │
└──────────────┬─────────────────────┘
               │  dynamic tool selection
    ┌──────────┼──────────────────┐
    ▼          ▼                  ▼
Calculator  FileReader   WebSearch / Weather / Summarizer
               │
    ┌──────────▼──────────┐
    │  Sliding Memory     │  ← 10-turn window
    │  Session Logger     │  ← JSON logs in logs/
    └─────────────────────┘
               │
    ┌──────────▼──────────┐
    │   Structured Output │
    │  final_answer       │
    │  tools_used         │
    │  reasoning_chain    │
    └─────────────────────┘
```

**Key design decisions:**

| Decision | Rationale |
|---|---|
| `temperature=0.2` | Low temp → focused, consistent reasoning; reduces hallucination in tool selection |
| `handle_parsing_errors=True` | When Llama misformats ReAct output, the executor retries gracefully |
| `max_iterations=8` | Prevents infinite loops on complex tasks |
| `ConversationBufferWindowMemory(k=10)` | Keeps the last 10 turns without overflowing context |
| AST-based calculator | Never uses `eval()` — safe against injection attacks |
| DuckDuckGo search | No API key required; real web results |
| Open-Meteo weather | Free, no API key; worldwide coverage |

---

## Project Structure

```
autonomous_agent/
├── agent.py                  # Main entry point + programmatic API
├── config.py                 # Centralised settings (model, memory, limits)
├── logger.py                 # JSON session logger
├── output_formatter.py       # Rich terminal output + structured dict
├── streamlit_app.py          # Web UI (optional)
├── example_scenarios.py      # 5 demo multi-step tasks
├── requirements.txt
├── tools/
│   ├── __init__.py
│   ├── calculator_tool.py    # Safe AST math evaluator
│   ├── file_reader_tool.py   # txt / csv / json reader
│   ├── web_search_tool.py    # DuckDuckGo search
│   ├── weather_tool.py       # Open-Meteo current weather
│   └── summarizer_tool.py    # Llama-powered bullet summarizer
├── tests/
│   ├── __init__.py
│   └── test_tools.py         # 33 offline unit tests
└── logs/                     # Auto-created; stores session JSON logs
```

---

## Setup Instructions

### Step 1 — Install Ollama

Ollama runs Llama locally on your machine (CPU or GPU).

```bash
# Linux / macOS
curl -fsSL https://ollama.ai/install.sh | sh

# Windows — download from https://ollama.ai/download
```

### Step 2 — Pull the Llama model

Choose based on your available RAM:

| Model | RAM | Speed | Command |
|---|---|---|---|
| `llama3.1` (8B) | 8 GB+ | Medium | `ollama pull llama3.1` |
| `llama3.2:3b` | 4 GB+ | Fast | `ollama pull llama3.2:3b` |
| `llama3.1:70b` | 40 GB+ | Slow | `ollama pull llama3.1:70b` |

> **Tip:** Start Ollama before running the agent: `ollama serve`

### Step 3 — Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### Step 4 — Install dependencies

```bash
pip install -r requirements.txt
```

---

## Required Dependencies

```
langchain>=0.2.0
langchain-community>=0.2.0
langchain-core>=0.2.0
ollama>=0.2.0
duckduckgo-search>=6.1.0
requests>=2.31.0
rich>=13.7.0

# Optional — for Web UI
streamlit>=1.35.0

# Optional — for testing
pytest>=8.0.0
```

> **No API keys needed.** DuckDuckGo and Open-Meteo are both free and keyless.

---

## How to Run

### Option A — Interactive CLI (recommended)

```bash
# Make sure Ollama is running first
ollama serve &

# Start the agent
python agent.py

# Use a different model
python agent.py llama3.2:3b
```

**CLI commands:**

| Command | Effect |
|---|---|
| `exit` | End session (shows log summary) |
| `clear` | Reset conversation memory |
| `tools` | List all available tools |
| `history` | Show conversation history |

### Option B — Web UI with Streamlit

```bash
pip install streamlit
streamlit run streamlit_app.py
```

Opens at `http://localhost:8501`. Features a chat interface, reasoning chain expander, and per-turn metrics (tools used, steps, time).

### Option C — Run the 5 Example Scenarios

```bash
python example_scenarios.py
# Results saved to output/scenario_1.json … output/scenario_5.json
```

### Option D — Programmatic API

```python
from agent import run_task, build_agent

# Single task
result = run_task("What is the compound interest on 5000 EGP at 12% for 7 years?")
print(result["final_answer"])
print(result["tools_used"])        # ['Calculator']
print(result["reasoning_steps"])   # 2

# Multi-turn session (shared memory)
agent = build_agent("llama3.1")
r1 = agent.invoke({"input": "Search for LangChain ReAct agents"})
r2 = agent.invoke({"input": "Summarize what you just found"})  # uses memory
```

---

## Architecture & Design

### ReAct Loop

The agent follows the Reasoning + Acting pattern strictly:

```
Thought:       I need to calculate compound interest.
Action:        Calculator
Action Input:  5000 * (1.12^7)
Observation:   Result: 11057.89

Thought:       Now I need the profit percentage.
Action:        Calculator
Action Input:  ((11057.89 - 5000) / 5000) * 100
Observation:   Result: 121.16

Thought:       I have all data needed.
Final Answer:  After 7 years at 12%, your 5000 EGP grows to 11,057.89 EGP
               — a profit of 6,057.89 EGP (121.16% return).
```

### Memory

`ConversationBufferWindowMemory(k=10)` stores the last 10 human/assistant turns as a string block injected into the prompt under `{chat_history}`. This lets the agent reference previous answers without exceeding Llama's context window.

### Session Logging

Every interaction is appended to `logs/session_<timestamp>.json`:

```json
{
  "session_id": "20240415_143022",
  "turns": [
    {
      "timestamp": "2024-04-15T14:30:25",
      "input": "What is 15% of 3500?",
      "output": "Final Answer: 525 EGP",
      "tools_used": ["Calculator"],
      "reasoning_steps": 1,
      "chain": [{"tool": "Calculator", "tool_input": "3500 * 0.15", "observation": "Result: 525"}]
    }
  ]
}
```

Disable logging: `export AGENT_LOGGING=false`

---

## Tools Reference

### Calculator
- **Input:** Any math expression string, e.g. `(100 * 1.08^5) / 2`
- **Supports:** `+`, `-`, `*`, `/`, `^` (power), `%` (percent), parentheses
- **Safety:** Uses Python's `ast` module — never calls `eval()`
- **Errors:** Division by zero, invalid syntax, unsupported operations — all handled

### FileReader
- **Input:** File path string, e.g. `/home/user/data.csv`
- **Formats:** `.txt`, `.md`, `.csv`, `.json`, `.log`, `.py`, `.html`, and other text files
- **Large files:** Truncated at 3000 characters with `[truncated]` notice
- **CSV:** Returns header + first 20 rows + row count
- **Errors:** File not found, binary files, permission errors — all handled

### WebSearch
- **Input:** Search query string, e.g. `LangChain ReAct agent tutorial 2024`
- **Returns:** Top 5 results with title, URL, and summary snippet
- **Backend:** DuckDuckGo (no API key, no rate limits for normal use)

### Weather
- **Input:** City name, e.g. `Cairo`, `London`, `New York`
- **Returns:** Temperature (°C), feels-like, humidity, wind speed, precipitation, condition
- **Backend:** Open-Meteo geocoding API + forecast API (both free)

### Summarizer
- **Input:** Long text block to summarize
- **Returns:** Bullet-point summary generated by the local Llama model
- **Best used after:** FileReader or WebSearch when content is too long
- **Truncation:** Input over 8000 characters is truncated before summarization

---

## Example Usage Scenarios

### 1 — Multi-step Finance Calculation
```
You: I'm investing 5000 EGP at 12% annual compound interest for 7 years.
     What's the final amount and profit percentage?

Agent:
  Thought: I need compound interest, then profit%.
  Action:  Calculator → 5000 * (1.12^7) → 11057.89
  Action:  Calculator → ((11057.89-5000)/5000)*100 → 121.16

  Final Answer: 11,057.89 EGP — profit of 6,057.89 EGP (121.16% return)
```

### 2 — Multi-city Weather + Calculation
```
You: Compare temperatures in Cairo and London, convert difference to Fahrenheit.

Agent:
  Action: Weather → Cairo   → 28°C
  Action: Weather → London  → 14°C
  Action: Calculator → (28-14) * 9/5 → 25.2

  Final Answer: Cairo is 14°C warmer (25.2°F difference).
```

### 3 — Web Research + Summarization
```
You: Search for 'transformer attention mechanism 2024' and summarize findings.

Agent:
  Action: WebSearch  → [5 results]
  Action: Summarizer → [bullet points]

  Final Answer: • Attention weights determine token relevance...
```

### 4 — File Analysis
```
You: Read students.csv and calculate the average grade.

Agent:
  Action: FileReader  → [CSV contents]
  Action: Calculator  → (88+92+75+95+81+78) / 6

  Final Answer: 6 students, average grade: 84.83
```

### 5 — Multi-tool Cross-domain Task
```
You: Find Egypt's population, calculate 1.5% annual growth, check Alexandria weather.

Agent:
  Action: WebSearch   → "Egypt population 2024" → ~105 million
  Action: Calculator  → 105000000 * 0.015 → 1,575,000
  Action: Weather     → Alexandria → 24°C, partly cloudy

  Final Answer: Population ~105M, annual growth ~1.575M.
               Alexandria today: 24°C, partly cloudy — good day to go outside!
```

---

## Configuration

All settings are in `config.py`. You can also override via environment variables:

```bash
export LLAMA_MODEL=llama3.2:3b        # model to use
export OLLAMA_BASE_URL=http://localhost:11434
export AGENT_TEMP=0.1                 # lower = more deterministic
export AGENT_MAX_TOKENS=4096
export AGENT_LOGGING=false            # disable session logs
export AGENT_LOG_DIR=/tmp/agent_logs  # custom log directory
```

---

## Testing

Run all 33 unit tests (no Ollama required):

```bash
python -m pytest tests/ -v
```

Tests cover:
- **Calculator:** 16 cases — arithmetic, power, percentage, edge cases, error handling
- **FileReader:** 9 cases — txt, csv, json, large files, missing files, quoted paths
- **OutputFormatter:** 3 cases — empty steps, single tool, multiple tools
- **Config:** 3 cases — type checks, value ranges
- **Logger:** 2 cases — file creation, empty summary

---

## Extending the System

### Add a new tool

1. Create `tools/my_tool.py`:

```python
from langchain_core.tools import Tool

def my_function(input_str: str) -> str:
    try:
        # Your logic here
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {e}"

my_tool = Tool(
    name="MyTool",
    func=my_function,
    description="One clear sentence describing what this tool does and when to use it.",
)
```

2. Add to `tools/__init__.py`:
```python
from .my_tool import my_tool
```

3. Add to `TOOLS` list in `agent.py`:
```python
from tools.my_tool import my_tool
TOOLS = [..., my_tool]
```

The agent will automatically discover, describe, and select the tool.

### Use a HuggingFace model instead of Ollama

```python
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline

pipe = pipeline("text-generation", model="meta-llama/Meta-Llama-3.1-8B-Instruct",
                max_new_tokens=2048, temperature=0.2)
llm  = HuggingFacePipeline(pipeline=pipe)
```

Replace the `load_llm()` return value with this `llm` object.

---

*Built for AIE 418 — Alamein International University*
