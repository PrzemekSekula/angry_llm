"""
Main entry point for the LLM negotiation experiment.
"""
import argparse
from src.pipeline import run_pipeline


def main():
    """
    Parses arguments and runs the negotiation pipeline.
    """
    parser = argparse.ArgumentParser(description="Run LLM Negotiation Experiment")
    parser.add_argument(
        "--iterations", 
        type=int, 
        default=5, 
        help="Number of negotiation iterations to run (default: 5)"
    )
    
    args = parser.parse_args()
    
    try:
        run_pipeline(max_iterations=args.iterations)
    except Exception as e:
        print(f"An error occurred during execution: {e}")

if __name__ == "__main__":
    main()
