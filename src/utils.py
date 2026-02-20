"""
Utility functions for the LLM negotiation experiment.
"""
import os
import csv
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from langchain_core.messages import BaseMessage



class CustomFormatter(logging.Formatter):
    """Custom formatter to log LLM name and iteration on a separate line."""
    
    def format(self, record):
        if hasattr(record, 'llm_name') and hasattr(record, 'iteration'):
            header = f"{record.llm_name} (Iteration {record.iteration})"
            message = record.getMessage()
            return f"{header}\n{message}\n"
        return super().format(record)

def setup_logging(log_dir: str = "../log"):
    """
    Sets up logging to a file in the specified directory.
    Creates a new log file for each experiment run.
    """
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"experiment_{timestamp}.log")

    logger = logging.getLogger("NegotiationLogger")
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    formatter = CustomFormatter('%(message)s')
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    return logger

def log_message(logger, llm_name: str, iteration: int, message: str):
    """
    Logs a message with the specified LLM name and iteration number.
    """
    logger.info(message, extra={'llm_name': llm_name, 'iteration': iteration})

def load_api_key(config_path: str = ".env") -> str:
    """
    as Reads the OpenAI API key from the specified config file.
    """
    try:
        with open(config_path, 'r') as f:
            return f.read().strip()
    except FileNotFoundError:
        raise FileNotFoundError(f"API key file not found at {config_path}")

# def load_prompt(prompt_name: str, prompts_dir: str = "prompts") -> str:
#     """
#     Reads a prompt from the prompts directory.
#     """
#     path = os.path.join(prompts_dir, f"{prompt_name}.txt")
#     try:
#         with open(path, 'r', encoding='utf-8') as f:
#             return f.read().strip()
#     except FileNotFoundError:
#         return f"System prompt for {prompt_name} not found."


BASE_DIR = Path(__file__).resolve().parent.parent  # katalog projektu (ten co ma prompts/)

def load_prompt(prompt_name: str, prompts_dir: str = "prompts") -> str:
    path = BASE_DIR / prompts_dir / f"{prompt_name}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Prompt not found: {path}")
    return path.read_text(encoding="utf-8").strip()



log_dir: str = "../log"
os.makedirs(log_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
AB_LOG_PATH = Path(f"{log_dir}/ab_prompts_{timestamp}.csv")

def _messages_to_text(messages: List[BaseMessage]) -> str:
    """Zamienia listę wiadomości LangChain na czytelny, pełny tekst promptu."""
    out = []
    for m in messages:
        role = getattr(m, "type", m.__class__.__name__).upper()
        content = getattr(m, "content", "")
        out.append(f"[{role}]\n{content}")
    return "\n\n".join(out)

# def append_ab_row(
#     iteration: int,
#     speaker: str,                 # "A" lub "B"
#     prompt_messages: List[BaseMessage],
#     response_text: str,
#     csv_path: Path = AB_LOG_PATH,
# ) -> None:
#     csv_path.parent.mkdir(parents=True, exist_ok=True)

#     file_exists = csv_path.exists()
#     with csv_path.open("a", newline="", encoding="utf-8") as f:
#         w = csv.writer(f)
#         if not file_exists:
#             w.writerow(["timestamp", "iteration", "speaker", "prompt_in", "response_out"])
#         w.writerow([
#             datetime.now().isoformat(timespec="seconds"),
#             iteration,
#             speaker,
#             _messages_to_text(prompt_messages),
#             response_text,
#         ])
def append_ab_row(
    iteration: int,
    speaker: str,                 # "A", "B", "EMO"
    prompt_messages: List[BaseMessage],
    response_text: str,
    anger_intensity: float | None = None,
    anger_state: str | None = None,
    csv_path: Path = AB_LOG_PATH,
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    file_exists = csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)

        if not file_exists:
            w.writerow([
                "timestamp",
                "iteration",
                "speaker",
                "prompt_in",
                "response_out",
                "anger_intensity",
                "anger_state",
            ])

        w.writerow([
            datetime.now().isoformat(timespec="seconds"),
            iteration,
            speaker,
            _messages_to_text(prompt_messages),
            response_text,
            anger_intensity,
            anger_state,
        ])
