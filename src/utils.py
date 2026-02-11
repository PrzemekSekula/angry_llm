"""
Utility functions for the LLM negotiation experiment.
"""
import os
import logging
from datetime import datetime

class CustomFormatter(logging.Formatter):
    """Custom formatter to log LLM name and iteration on a separate line."""
    
    def format(self, record):
        if hasattr(record, 'llm_name') and hasattr(record, 'iteration'):
            header = f"{record.llm_name} (Iteration {record.iteration})"
            message = record.getMessage()
            return f"{header}\n{message}\n"
        return super().format(record)

def setup_logging(log_dir: str = "log"):
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

def load_api_key(config_path: str = "configs/openai_key.txt") -> str:
    """
    as Reads the OpenAI API key from the specified config file.
    """
    try:
        with open(config_path, 'r') as f:
            return f.read().strip()
    except FileNotFoundError:
        raise FileNotFoundError(f"API key file not found at {config_path}")

def load_prompt(prompt_name: str, prompts_dir: str = "prompts") -> str:
    """
    Reads a prompt from the prompts directory.
    """
    path = os.path.join(prompts_dir, f"{prompt_name}.txt")
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except FileNotFoundError:
        return f"System prompt for {prompt_name} not found."
