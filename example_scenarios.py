"""
example_scenarios.py
Demonstrates five real-world multi-step agent tasks.
Run:  python example_scenarios.py
"""

from agent import run_task, build_agent
from output_formatter import format_structured_output
from rich.console import Console
from rich.panel import Panel
import json

console = Console()

SCENARIOS = [
    {
        "title": "📐 Multi-step Math",
        "task": (
            "I am investing 5000 EGP at 12% annual compound interest for 7 years. "
            "What is the final amount? Then calculate the profit and express it as a percentage of the initial investment."
        ),
    },
    {
        "title": "🌦 Weather + Calculation",
        "task": (
            "Check the current temperature in Cairo and in London. "
            "Calculate the difference in Celsius and convert it to Fahrenheit (multiply by 9/5)."
        ),
    },
    {
        "title": "🔍 Web Research Summary",
        "task": (
            "Search for 'LangChain ReAct agent 2024' and summarize the key points "
            "from the top search results in bullet points."
        ),
    },
    {
        "title": "📄 File Analysis",
        "task": (
            "Read the file 'sample_data/students.csv' and tell me: "
            "how many students are there, what columns exist, and calculate the average grade."
        ),
    },
    {
        "title": "🧠 Multi-tool Reasoning",
        "task": (
            "Search for the current population of Egypt. "
            "Then calculate what 1.5% of that population is (approximate annual growth). "
            "Also check the weather in Alexandria and tell me if it's a good day to be outside."
        ),
    },
]


def main():
    console.print(Panel.fit(
        "[bold cyan]Autonomous Llama Agent — Example Scenarios[/bold cyan]\n"
        "[dim]Runs 5 multi-step tasks to demonstrate dynamic tool selection.[/dim]",
        border_style="cyan",
    ))

    for i, scenario in enumerate(SCENARIOS, 1):
        console.print(f"\n[bold white]{'─'*60}[/bold white]")
        console.print(f"[bold yellow]Scenario {i}: {scenario['title']}[/bold yellow]")
        console.print(f"[dim]Task:[/dim] {scenario['task']}\n")

        try:
            result     = run_task(scenario["task"])
            structured = format_structured_output(result)

            console.print(f"[bold green]✓ Final Answer:[/bold green] {structured['final_answer']}")
            console.print(
                f"[dim]Tools used: {', '.join(structured['tools_used'])} | "
                f"Reasoning steps: {structured['reasoning_steps']}[/dim]"
            )

            # Save structured output
            out_path = f"output/scenario_{i}.json"
            import os; os.makedirs("output", exist_ok=True)
            with open(out_path, "w") as f:
                json.dump(structured, f, indent=2)
            console.print(f"[dim]Saved to {out_path}[/dim]")

        except Exception as e:
            console.print(f"[bold red]Error in scenario {i}:[/bold red] {e}")

    console.print(f"\n[bold green]All scenarios complete.[/bold green]")


if __name__ == "__main__":
    # Create sample CSV for scenario 4
    import os
    os.makedirs("sample_data", exist_ok=True)
    with open("sample_data/students.csv", "w") as f:
        f.write("name,grade,subject\n")
        f.write("Ahmed,88,Math\nFatima,92,Math\nMohamed,75,Math\n")
        f.write("Sara,95,Math\nOmar,81,Math\nNour,78,Math\n")

    main()
