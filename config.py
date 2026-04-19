"""
config.py — Centralised configuration for the Autonomous Agent System.
Edit values here; they are imported by agent.py and all tools.
"""

import os

# ── Model ──────────────────────────────────────────────────────────────────
LLAMA_MODEL      = os.getenv("LLAMA_MODEL", "llama3.1")   # override via env var
OLLAMA_BASE_URL  = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
TEMPERATURE      = float(os.getenv("AGENT_TEMP", "0.2"))
MAX_TOKENS       = int(os.getenv("AGENT_MAX_TOKENS", "2048"))
TOP_P            = 0.9
REPEAT_PENALTY   = 1.1

# ── Agent behaviour ────────────────────────────────────────────────────────
MAX_ITERATIONS       = 8       # ReAct loop limit
MAX_EXECUTION_TIME   = 120     # seconds before forced stop
MEMORY_WINDOW        = 10      # turns kept in sliding-window memory
VERBOSE              = True    # print Thought/Action/Observation to terminal

# ── File reader limits ─────────────────────────────────────────────────────
FILE_READER_MAX_CHARS = 3000   # characters returned to agent
CSV_PREVIEW_ROWS      = 20

# ── Summarizer ─────────────────────────────────────────────────────────────
SUMMARIZER_MAX_INPUT  = 8000   # chars before truncation
SUMMARIZER_MAX_TOKENS = 512

# ── Web search ─────────────────────────────────────────────────────────────
WEB_SEARCH_MAX_RESULTS = 5

# ── Logging ────────────────────────────────────────────────────────────────
LOG_DIR      = os.getenv("AGENT_LOG_DIR", "logs")
LOG_ENABLED  = os.getenv("AGENT_LOGGING", "true").lower() == "true"
