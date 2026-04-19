"""
Tool: File Reader
Reads plain-text, CSV, and JSON files from the local filesystem.
"""

import os
import json
import csv
from langchain_core.tools import tool

MAX_CHARS = 3000


@tool
def file_reader_tool(file_path: str) -> str:
    """
    Read a local file and return its contents.
    Input: the full file path e.g. /home/user/data.csv or report.txt.
    Supports .txt, .md, .csv, .json, .log, .py and other text files.
    """
    file_path = file_path.strip().strip('"').strip("'")
    if not os.path.exists(file_path):
        return f"Error: File not found at '{file_path}'."
    if not os.path.isfile(file_path):
        return f"Error: '{file_path}' is a directory, not a file."

    ext  = os.path.splitext(file_path)[1].lower()
    size = os.path.getsize(file_path)

    try:
        if ext == ".json":
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            content = json.dumps(data, indent=2, ensure_ascii=False)
            if len(content) > MAX_CHARS:
                content = content[:MAX_CHARS] + "\n... [truncated]"
            return f"[JSON file - {size} bytes]\n{content}"

        elif ext == ".csv":
            with open(file_path, "r", encoding="utf-8", newline="") as f:
                rows = list(csv.reader(f))
            header = rows[0] if rows else []
            preview = rows[:20]
            lines = [",".join(header)] + [",".join(r) for r in preview[1:]]
            note = f"\n... [{len(rows)-20} more rows]" if len(rows) > 20 else ""
            return f"[CSV - {len(rows)} rows, columns: {header}]\n" + "\n".join(lines) + note

        else:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            if len(content) > MAX_CHARS:
                content = content[:MAX_CHARS] + "\n... [truncated]"
            return f"[{ext or 'text'} file - {size} bytes]\n{content}"

    except UnicodeDecodeError:
        return f"Error: Cannot read binary file '{file_path}' as text."
    except PermissionError:
        return f"Error: Permission denied reading '{file_path}'."
    except Exception as e:
        return f"Unexpected error: {e}"