"""
Autonomous AI Agent System using Llama via Ollama + LangChain
AIE 418 - Selected Topics in AI 2
"""

from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver

from langchain_groq import ChatGroq

from tools.calculator_tool  import calculator_tool
from tools.file_reader_tool import file_reader_tool
from tools.web_search_tool  import web_search_tool
from tools.weather_tool     import weather_tool
from tools.summarizer_tool  import summarizer_tool

from output_formatter import format_structured_output
from logger import SessionLogger
import config

from rich.console import Console
from rich.panel   import Panel
import sys, uuid

console = Console()

TOOLS = [
    calculator_tool,
    file_reader_tool,
    web_search_tool,
    weather_tool,
    summarizer_tool,
]

SYSTEM_PROMPT = (
    "You are an autonomous AI research and task assistant powered by Llama.\n"
    "You are methodical, precise, and always explain your reasoning before acting.\n"
    "Use the available tools to find real data. Never fabricate facts or numbers.\n"
    "Give a clear, well-structured final response."
)


def load_llm(model_name: str = config.LLAMA_MODEL) -> ChatGroq:
    console.print(f"[bold cyan]Model:[/bold cyan] {model_name} [dim](Groq)[/dim]")
    
    return ChatGroq(
        model=model_name,           # مثال: llama3-70b-8192 أو llama3-8b-8192
        temperature=config.TEMPERATURE,
        api_key=config.GROQ_API_KEY,   # مهم جداً
        max_tokens=1024,
    )

def build_agent(model_name: str = config.LLAMA_MODEL):
    """Build agent using langchain 1.x create_agent with MemorySaver."""
    llm    = load_llm(model_name)
    memory = MemorySaver()
    return create_agent(
        model=llm,
        tools=TOOLS,
        system_prompt=SYSTEM_PROMPT,
        checkpointer=memory,
    )

def extract_output(response) -> str:
    """Extract the last AI message text from agent response."""
    messages = response.get("messages", [])
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content:
            return msg.content
    return str(response)

def run_cli(model_name: str = config.LLAMA_MODEL) -> None:
    agent     = build_agent(model_name)
    logger    = SessionLogger()
    thread_id = str(uuid.uuid4())
    cfg       = {"configurable": {"thread_id": thread_id}}

    console.print(Panel.fit(
        f"[bold green]Autonomous Llama Agent[/bold green]  [dim]({model_name})[/dim]\n"
        "[dim]Commands:  exit / tools / clear[/dim]",
        border_style="green",
    ))

    while True:
        try:
            user_input = console.input("\n[bold yellow]You >[/bold yellow] ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print(f"\n[dim]{logger.summary()}[/dim]")
            break

        if not user_input:
            continue
        if user_input.lower() == "exit":
            console.print(f"[dim]{logger.summary()}[/dim]")
            break
        if user_input.lower() == "clear":
            thread_id = str(uuid.uuid4())
            cfg       = {"configurable": {"thread_id": thread_id}}
            console.print("[dim]Memory cleared.[/dim]")
            continue
        if user_input.lower() == "tools":
            for t in TOOLS:
                console.print(f"  [cyan]{t.name}[/cyan] - {t.description[:80]}...")
            continue

        try:
            response = agent.invoke(
                {"messages": [HumanMessage(content=user_input)]},
                config=cfg,
            )
            output = extract_output(response)
            console.print(f"\n[bold green]Agent >[/bold green] {output}\n")
            logger.log(user_input, {"output": output, "intermediate_steps": []})
        except Exception as exc:
            console.print(f"[bold red]Agent error:[/bold red] {exc}")

def run_task(task: str, model_name: str = config.LLAMA_MODEL) -> dict:
    agent    = build_agent(model_name)
    cfg      = {"configurable": {"thread_id": str(uuid.uuid4())}}
    response = agent.invoke(
        {"messages": [HumanMessage(content=task)]},
        config=cfg,
    )
    return {"final_answer": extract_output(response), "model": model_name}

if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else config.LLAMA_MODEL
    run_cli(model)
    
