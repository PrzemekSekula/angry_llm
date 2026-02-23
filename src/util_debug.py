"""
Debug utility functions for the LLM negotiation experiment.
"""
import csv
from datetime import datetime
from pathlib import Path
from typing import List
from langchain_core.messages import BaseMessage

log_dir = "../log"
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
DEBUG_LOG_PATH = Path(f"{log_dir}/debug_log_{timestamp}.csv")

def _messages_to_text(messages: List[BaseMessage]) -> str:
    """Zamienia listę wiadomości LangChain na czytelny, pełny tekst promptu."""
    out = []
    for m in messages:
        role = getattr(m, "type", m.__class__.__name__).upper()
        content = getattr(m, "content", "")
        out.append(f"[{role}]\n{content}")
    return "\n\n".join(out)

def log_debug_csv(
    iteration: int,
    agent: str,                 
    prompt_messages: List[BaseMessage],
    response_text: str,
    csv_path: Path = DEBUG_LOG_PATH,
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    file_exists = csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)

        if not file_exists:
            w.writerow([
                "iteration",
                "agent",
                "input",
                "output",
            ])

        w.writerow([
            iteration,
            agent,
            _messages_to_text(prompt_messages),
            response_text,
        ])
