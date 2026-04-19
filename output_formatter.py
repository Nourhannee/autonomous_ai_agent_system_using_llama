"""
Output Formatter
Converts raw AgentExecutor output into a structured, readable Rich panel.
"""

from rich.panel import Panel
from rich.table import Table
from rich.text  import Text
from rich       import box
import json


def format_agent_response(result: dict) -> Panel:
    """
    Format AgentExecutor output into a Rich Panel for display.

    Args:
        result: dict from AgentExecutor.invoke() containing:
                - output (str)
                - intermediate_steps (list of (AgentAction, str) tuples)
    Returns:
        A Rich Panel ready for console.print()
    """
    output = result.get("output", "No response.")
    steps  = result.get("intermediate_steps", [])

    # ── Build reasoning chain table ──────────────────────────
    table = Table(
        title="Reasoning Chain",
        box=box.SIMPLE_HEAD,
        show_header=True,
        header_style="bold magenta",
        expand=True,
    )
    table.add_column("#",          style="dim",         width=3)
    table.add_column("Tool",       style="cyan",        width=14)
    table.add_column("Input",      style="yellow",      max_width=40, overflow="fold")
    table.add_column("Observation",style="white",       max_width=60, overflow="fold")

    for i, (action, observation) in enumerate(steps, 1):
        tool_name  = getattr(action, "tool",       "?")
        tool_input = getattr(action, "tool_input", "")

        # Truncate long inputs / observations for display
        if isinstance(tool_input, str) and len(tool_input) > 120:
            tool_input = tool_input[:120] + "…"
        if isinstance(observation, str) and len(observation) > 200:
            observation = observation[:200] + "…"

        table.add_row(str(i), tool_name, str(tool_input), str(observation))

    # ── Final answer ──────────────────────────────────────────
    final = Text(output, style="bold green")

    from rich.console import Console
    from rich.columns import Columns
    from io import StringIO

    sio = StringIO()
    tmp = Console(file=sio, highlight=False)
    if steps:
        tmp.print(table)
    tmp.print("\n[bold green]Final Answer[/bold green]")
    tmp.print(final)

    return Panel(
        sio.getvalue(),
        title="[bold white]Agent Response[/bold white]",
        border_style="blue",
        expand=False,
    )


def format_structured_output(result: dict) -> dict:
    """
    Return a clean JSON-serialisable dict from AgentExecutor output.
    Useful when calling the agent programmatically.
    """
    steps = result.get("intermediate_steps", [])
    chain = []
    for action, observation in steps:
        chain.append({
            "tool":        getattr(action, "tool",       ""),
            "tool_input":  getattr(action, "tool_input", ""),
            "observation": str(observation),
            "log":         getattr(action, "log",        ""),
        })

    return {
        "final_answer":      result.get("output", ""),
        "reasoning_steps":   len(chain),
        "tools_used":        list({s["tool"] for s in chain}),
        "reasoning_chain":   chain,
    }
