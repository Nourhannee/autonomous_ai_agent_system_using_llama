"""
logger.py — Session logger for the Autonomous Agent System.
Saves each interaction (input, reasoning chain, output) as a JSON log file.
"""

import json
import os
from datetime import datetime
from config import LOG_DIR, LOG_ENABLED


class SessionLogger:
    """Logs agent interactions to timestamped JSON files."""

    def __init__(self, session_id: str | None = None):
        self.enabled = LOG_ENABLED
        if not self.enabled:
            return

        os.makedirs(LOG_DIR, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_id = session_id or ts
        self.log_path   = os.path.join(LOG_DIR, f"session_{self.session_id}.json")
        self.entries: list[dict] = []

    def log(self, user_input: str, result: dict) -> None:
        """Record one interaction turn."""
        if not self.enabled:
            return

        steps = result.get("intermediate_steps", [])
        chain = []
        for action, observation in steps:
            chain.append({
                "tool":       getattr(action, "tool", ""),
                "tool_input": getattr(action, "tool_input", ""),
                "observation": str(observation)[:500],  # truncate long obs
            })

        entry = {
            "timestamp":       datetime.now().isoformat(),
            "input":           user_input,
            "output":          result.get("output", ""),
            "tools_used":      list({s["tool"] for s in chain}),
            "reasoning_steps": len(chain),
            "chain":           chain,
        }
        self.entries.append(entry)
        self._save()

    def _save(self) -> None:
        with open(self.log_path, "w", encoding="utf-8") as f:
            json.dump(
                {"session_id": self.session_id, "turns": self.entries},
                f, indent=2, ensure_ascii=False,
            )

    def summary(self) -> str:
        if not self.entries:
            return "No interactions logged."
        tools_all = []
        for e in self.entries:
            tools_all.extend(e["tools_used"])
        unique_tools = set(tools_all)
        return (
            f"Session {self.session_id}: {len(self.entries)} turns | "
            f"Tools used: {', '.join(unique_tools) or 'none'} | "
            f"Log: {self.log_path}"
        )
