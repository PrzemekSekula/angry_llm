"""
Core LangGraph pipeline implementation.
"""
import os
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

from .utils import load_api_key, load_prompt, setup_logging, log_message

# Load API Key
try:
    api_key = load_api_key()
    os.environ["OPENAI_API_KEY"] = api_key
except Exception as e:
    print(f"Warning: {e}")

# Define State
class GraphState(TypedDict):
    history_a: List[BaseMessage]
    history_b: List[BaseMessage]
    emo_score: float
    last_message: str
    iteration: int
    max_iterations: int

# Initialize Logger
logger = setup_logging()

# Initialize Models
# Using gpt-4o as a fallback if gpt-5.2 is not available/valid, but setting model name as requested.
# Note: The user requested gpt-5.2. If this model name is invalid for the API, it will fail.
model = ChatOpenAI(model="gpt-5.2", temperature=0.7) 

def run_pipeline(max_iterations: int = 5):
    """
    Runs the negotiation pipeline.
    """
    
    # --- Node Definitions ---

    def node_llm_a(state: GraphState):
        """
        LLM_A generates a message based on history_a.
        """
        iteration = state['iteration']
        prompt = load_prompt("llm_a")
        
        # Construct messages
        messages = [SystemMessage(content=prompt)] + state['history_a']
        
        # If it's the very first message and history is empty, add an initial trigger
        if not state['history_a']:
             messages.append(HumanMessage(content="Start the negotiation."))

        response = model.invoke(messages)
        content = response.content

        # Log
        log_message(logger, "LLM_A", iteration, content)

        # Update state
        return {
            "history_a": state['history_a'] + [response],
            "last_message": content
        }

    def node_twist_a(state: GraphState):
        """
        twist_LLM_A modifies the message from LLM_A.
        """
        iteration = state['iteration']
        prompt = load_prompt("twist _llm_a")
        original_msg = state['last_message']
        emo_score = state['emo_score']
        
        content_prompt = f"Original Message: {original_msg}\nEmotion Score: {emo_score}\nModify the message."
        messages = [SystemMessage(content=prompt), HumanMessage(content=content_prompt)]
        
        response = model.invoke(messages)
        content = response.content
        
        log_message(logger, "twist_LLM_A", iteration, content)
        
        # The modified message serves as input for LLM_B (HumanMessage from B's perspective)
        # We append this to history_b
        return {
            "history_b": state['history_b'] + [HumanMessage(content=content)],
            "last_message": content
        }

    def node_llm_b(state: GraphState):
        """
        LLM_B responds to the modified message.
        """
        iteration = state['iteration']
        prompt = load_prompt("llm_b")
        
        messages = [SystemMessage(content=prompt)] + state['history_b']
        
        response = model.invoke(messages)
        content = response.content
        
        log_message(logger, "LLM_B", iteration, content)
        
        return {
            "history_b": state['history_b'] + [response],
            "last_message": content
        }

    def node_emo(state: GraphState):
        """
        emo_LLM evaluates the conversation history B.
        """
        iteration = state['iteration']
        prompt = load_prompt("emo_llm")
        
        # Convert history to string for evaluation
        history_str = "\n".join([m.content for m in state['history_b']])
        messages = [SystemMessage(content=prompt), HumanMessage(content=f"Conversation History:\n{history_str}")]
        
        response = model.invoke(messages)
        content = response.content.strip()
        
        log_message(logger, "emo_LLM", iteration, content)
        
        try:
            score = float(content)
        except ValueError:
            score = 0.5 # fallback
            
        return {"emo_score": score}

    def node_twist_b(state: GraphState):
        """
        twist_LLM_B modifies the message from LLM_B.
        """
        iteration = state['iteration']
        prompt = load_prompt("twist_llm_b")
        original_msg = state['last_message']
        emo_score = state['emo_score']
        
        content_prompt = f"Original Message: {original_msg}\nEmotion Score: {emo_score}\nModify the message."
        messages = [SystemMessage(content=prompt), HumanMessage(content=content_prompt)]
        
        response = model.invoke(messages)
        content = response.content
        
        log_message(logger, "twist_LLM_B", iteration, content)
        
        # Modified message from B is input for A (HumanMessage from A's perspective)
        return {
            "history_a": state['history_a'] + [HumanMessage(content=content)],
            "iteration": iteration + 1
        }

    # --- Graph Construction ---
    
    workflow = StateGraph(GraphState)
    
    workflow.add_node("llm_a", node_llm_a)
    workflow.add_node("twist_a", node_twist_a)
    workflow.add_node("llm_b", node_llm_b)
    workflow.add_node("emo", node_emo)
    workflow.add_node("twist_b", node_twist_b)
    
    workflow.set_entry_point("llm_a")
    
    workflow.add_edge("llm_a", "twist_a")
    workflow.add_edge("twist_a", "llm_b")
    workflow.add_edge("llm_b", "emo")
    workflow.add_edge("emo", "twist_b")
    
    def check_loop(state: GraphState):
        if state['iteration'] > state['max_iterations']:
            return END
        return "llm_a"
        
    workflow.add_conditional_edges("twist_b", check_loop)
    
    app = workflow.compile()
    
    # --- Execution ---
    
    initial_state = {
        "history_a": [],
        "history_b": [],
        "emo_score": 0.0,
        "last_message": "",
        "iteration": 1,
        "max_iterations": max_iterations
    }
    
    print(f"Starting negotiation loop for {max_iterations} iterations...")
    result = app.invoke(initial_state)
    print("Negotiation finished.")
    return result
