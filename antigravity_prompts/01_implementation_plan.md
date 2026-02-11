# Implementation Plan - LLM Negotiation Pipeline

The goal is to implement a negotiation pipeline between 5 LLMs (`LLM_A`, `LLM_B`, `twist_LLM_A`, `twist_LLM_B`, `emo_LLM`) using LangGraph and Python.

## User Review Required

> [!IMPORTANT]
> - **API Key**: The system expects `configs/openai_key.txt` to contain the OpenAI API key. A placeholder will be created.
> - **Environment**: The code assumes the `AngryLLM` conda environment is active. I cannot verify this environment myself, so please ensure it is active before running the scripts.
> - **Model**: The system is configured to use `gpt-5.2` (as requested), but this model might not be publicly available yet. Please ensure the API key has access to it, or I can switch to `gpt-4o` or similar if needed. For now, I will use `gpt-5.2` as requested.

## Proposed Changes

### Project Structure (New Files)

#### [NEW] [requirements.txt](file:///d:/IITiS/angry_llm/requirements.txt)
- Add `langgraph`, `langchain-openai`, `langchain-core`, `python-dotenv`.

#### [NEW] [configs/openai_key.txt](file:///d:/IITiS/angry_llm/configs/openai_key.txt)
- Placeholder file for the API key.

#### [NEW] [prompts/](file:///d:/IITiS/angry_llm/prompts/)
- `llm_a.txt`
- `llm_b.txt`
- `twist_llm_a.txt`
- `twist_llm_b.txt`
- `emo_llm.txt`
- (All contain placeholders)


### Graph Structure & Logic

The LangGraph implementation will use a `StateGraph` with a defined schema to manage the conversation state.

#### State Schema
```python
class GraphState(TypedDict):
    history_a: List[BaseMessage]  # Context for LLM_A
    history_b: List[BaseMessage]  # Context for LLM_A (modified) + LLM_B
    emo_score: float              # Current evaluation (0.0 - 1.0)
    last_message: str             # Temporary storage for message passing
    iteration: int                # Counter
```

#### Nodes
1.  **`node_llm_a(state)`**:
    -   **Input**: `state['history_a']`
    -   **Action**: Calls `LLM_A` with `history_a`.
    -   **Output**: appends response to `history_a` and updates `state['last_message']`.
2.  **`node_twist_a(state)`**:
    -   **Input**: `state['last_message']` (from LLM_A), `state['emo_score']`.
    -   **Action**: Calls `twist_LLM_A` to modify the message.
    -   **Output**: Appends modified message to `history_b`.
3.  **`node_llm_b(state)`**:
    -   **Input**: `state['history_b']`
    -   **Action**: Calls `LLM_B` with `history_b`.
    -   **Output**: Appends response to `history_b` and updates `state['last_message']`.
4.  **`node_emo(state)`**:
    -   **Input**: `state['history_b']`.
    -   **Action**: Calls `emo_LLM` to evaluate the conversation.
    -   **Output**: Updates `state['emo_score']`.
5.  **`node_twist_b(state)`**:
    -   **Input**: `state['last_message']` (from LLM_B), `state['emo_score']`.
    -   **Action**: Calls `twist_LLM_B` to modify the message.
    -   **Output**: Appends modified message to `history_a`.

#### Edges (Flow)
`START` -> `node_llm_a` -> `node_twist_a` -> `node_llm_b` -> `node_emo` -> `node_twist_b` -> `node_llm_a` (loop)
-   The loop continues for a defined number of `iterations`.
-   We will likely add a conditional edge after `node_twist_b` to check if `iteration >= MAX_ITERATIONS` to `END`.

#### [NEW] [src/pipeline.py](file:///d:/IITiS/angry_llm/src/pipeline.py)


#### [NEW] [src/utils.py](file:///d:/IITiS/angry_llm/src/utils.py)
- `load_config()`: Reads API key.
- `setup_logging()`: Configures logging to `log/` folder.
- `log_message()`: Helper to format logs as requested (LLM name, iteration, message).

#### [NEW] [main.py](file:///d:/IITiS/angry_llm/main.py)
- Entry point.
- Parses arguments (e.g., `--iterations`).
- Initializes and runs the graph.

## Verification Plan

### Automated Tests
- Run `python main.py --help` to verify argument parsing and imports.
- I will run a dry-run (if possible without a valid API key) or just verify the script syntax. Since I cannot run python, I will rely on code review and user verification.

### Manual Verification
- User should populate `configs/openai_key.txt` and `prompts/*.txt`.
- User runs `python main.py`.
- Check `log/` folder for generated logs.
