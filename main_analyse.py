import os
import re
import csv
import glob
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage


def parse_offer(text):
    """Extracts A and B values from an OFFER string."""
    match = re.search(r'OFFER:\s*A=(\d+),\s*B=(\d+)', text, re.IGNORECASE)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None

def check_end_state(text):
    """Returns 'ACCEPT', 'END', or None."""
    if re.search(r'\bACCEPT\b', text, re.IGNORECASE):
        return 'ACCEPT'
    if re.search(r'\bEND\b', text, re.IGNORECASE):
        return 'END'
    return None

def analyze_sentiment(history_text, model):
    """Uses LLM to score sentiment from 1 (low) to 10 (high)."""
    system_prompt = """
    You are an expert negotiation analyst. Your task is to score the following negotiation history on 5 dimensions.
    Provide a score from 1 (lowest) to 10 (highest) for the ENTIRE interaction up to this point.
    
    Dimensions:
    1. toxicity
    2. anger
    3. rudeness
    4. calmness
    5. rationality
    
    Output strictly in the following format, with only the numbers:
    TOXICITY: <number>
    ANGER: <number>
    RUDENESS: <number>
    CALMNESS: <number>
    RATIONALITY: <number>
    """
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=history_text)
    ]
    
    response = model.invoke(messages)
    content = response.content.strip()
    
    scores = {
        'toxicity': None,
        'anger': None,
        'rudeness': None,
        'calmness': None,
        'rationality': None
    }
    
    for line in content.splitlines():
        line = line.strip().upper()
        for key in scores.keys():
            if line.startswith(f"{key.upper()}:"):
                try:
                    scores[key] = int(re.search(r'\d+', line).group())
                except:
                    pass
    return scores

def process_file(filepath, model, max_iterations):
    """Processes a single CSV file and extracts iteration data."""
    filename = os.path.basename(filepath)
    match = re.match(r'^(\d+)_([^_]+(?:_[^_]+)*)_ab_prompts', filename)
    experiment_id = match.group(1) if match else "unknown"
    model_name = match.group(2) if match else "unknown"
    
    results = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        iterations = {}
        for row in reader:
            it = int(row['iteration'])
            if it not in iterations:
                iterations[it] = {'A': None, 'B': None, 'EMO': None}
            
            speaker = row['speaker'].upper()
            if speaker in ['A', 'B', 'EMO']:
                iterations[it][speaker] = row
                
        full_history = []
        
        for it in sorted(iterations.keys()):
            data = iterations[it]
            row_result = {
                'experiment': experiment_id,
                'model': model_name,
                'iteration': it,
                'offer_a_A': None, 'offer_a_B': None,
                'offer_b_A': None, 'offer_b_B': None,
                'end_state': None, 
                'anger_intensity': None,
                'toxicity': None, 'anger': None, 'rudeness': None, 'calmness': None, 'rationality': None,
                'b_share_offer': None
            }
            
            if data['EMO']:
                try:
                    row_result['anger_intensity'] = float(data['EMO'].get('anger_intensity', 0))
                except (ValueError, TypeError):
                    row_result['anger_intensity'] = None
            
            if data['B']:
                resp_b = data['B']['response_out']
                full_history.append(f"B: {resp_b}")
                a_val, b_val = parse_offer(resp_b)
                if b_val is not None:
                    row_result['offer_b_A'] = a_val
                    row_result['offer_b_B'] = b_val
                    row_result['b_share_offer'] = b_val
                
                end_state = check_end_state(resp_b)
                if end_state:
                    row_result['end_state'] = end_state
                elif it >= max_iterations: # Time out if MAX_ITERATIONS reached
                    row_result['end_state'] = 'END_TIMEOUT'
                    
            if data['A']:
                resp_a = data['A']['response_out']
                full_history.append(f"A: {resp_a}")
                a_val, b_val = parse_offer(resp_a)
                if b_val is not None:
                    row_result['offer_a_A'] = a_val
                    row_result['offer_a_B'] = b_val
                    if row_result['b_share_offer'] is None:
                        row_result['b_share_offer'] = b_val
                        
                end_state = check_end_state(resp_a)
                if end_state:
                    row_result['end_state'] = end_state
                elif it >= max_iterations:
                    row_result['end_state'] = 'END_TIMEOUT'
                    
            history_text = "\n".join(full_history)
            sentiment_scores = analyze_sentiment(history_text, model)
            row_result.update(sentiment_scores)
            
            results.append(row_result)
            
            if row_result['end_state'] is not None:
                break 
                
    return results

def create_visualizations(df, analysis_dir):
    """Generates the required plots from the compiled dataframe."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.cm import get_cmap
    
    # Generate dynamic jet colors based on number of experiments
    experiments = df['experiment'].unique()
    cmap = get_cmap('jet')
    palette = {exp: cmap(i/len(experiments)) for i, exp in enumerate(experiments)}
    
    # 1. Anger States in iterations
    plt.figure(figsize=(10, 6))
    anger_df = df.dropna(subset=['anger_intensity']).copy()
    if not anger_df.empty:
        sns.lineplot(data=anger_df, x='iteration', y='anger_intensity', hue='experiment', marker='o', errorbar=('ci', 95), palette=palette)
        plt.title('Anger State Over Iterations')
        plt.xlabel('Iteration')
        plt.ylabel('Anger Intensity (0-1)')
        plt.legend(title='Experiment', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(analysis_dir, 'anger_states.png'))
        plt.close()

    # 2. Negotiations in iterations (B share)
    plt.figure(figsize=(12, 7))
    if not df.empty and 'b_share_offer' in df.columns:
        valid_offers = df.dropna(subset=['b_share_offer']).copy()
        
        # Plot lines with error bars
        sns.lineplot(data=valid_offers, x='iteration', y='b_share_offer', hue='experiment', marker='o', errorbar=('ci', 95), palette=palette)
        
        # Mark successes and failures at the end of each experiment's trajectory
        for exp in df['experiment'].unique():
            exp_data = df[df['experiment'] == exp]
            if not exp_data.empty:
                last_row = exp_data.iloc[-1]
                end_state = str(last_row['end_state']).upper()
                
                # Fetch the last iteration with a valid b_share_offer
                exp_valid = valid_offers[valid_offers['experiment'] == exp]
                if not exp_valid.empty:
                    last_valid_row = exp_valid.iloc[-1]
                    
                    marker = 'o'
                    color = 'green'
                    edgecolors = 'black'
                    
                    # If it explicitly ended in an agreement:
                    if 'ACCEPT' in end_state:
                        marker = 'o'
                        color = 'green'
                    else:
                        marker = 'X'
                        color = 'red'
                    
                    plt.scatter(
                        last_valid_row['iteration'], 
                        last_valid_row['b_share_offer'], 
                        marker=marker, color=color, s=150, edgecolors=edgecolors, zorder=5
                    )

        plt.title('Negotiation Offers across Iterations (B share)')
        plt.xlabel('Iteration')
        plt.ylabel("B's Share Offer")
        plt.legend(title='Experiment', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(analysis_dir, 'negotiation_progress.png'))
        plt.close()

def compute_statistics(df, analysis_dir):
    """Computes total/model statistics (mean, std, anger state, max anger state, and expected max offer)."""
    stats_path = os.path.join(analysis_dir, "statistics.csv")
    results = []
    valid_offers_df = df.dropna(subset=['b_share_offer'])
    
    # --- HELPER LOGIC FOR STAT METRICS ---
    def get_aggregated_stats(subset_df, label_group, label_name):
        b_offers = subset_df['b_share_offer'].dropna()
        anger = subset_df['anger_intensity'].dropna()
        
        expected_max_offer_sum = 0
        experiment_count = len(subset_df['experiment'].unique())
        accept_count = 0
        
        for exp in subset_df['experiment'].unique():
            exp_data = subset_df[subset_df['experiment'] == exp]
            last_row = exp_data.iloc[-1]
            end_state = str(last_row['end_state']).upper()
            
            exp_valid = valid_offers_df[valid_offers_df['experiment'] == exp]
            if not exp_valid.empty:
                last_valid = exp_valid.iloc[-1]
                if 'ACCEPT' in end_state:
                    expected_max_offer_sum += last_valid['b_share_offer']
                    accept_count += 1
        
        expected_max_mean = expected_max_offer_sum / experiment_count if experiment_count > 0 else 0
        
        mean_offer = b_offers.mean() if not b_offers.empty else 0
        std_offer = b_offers.std() if len(b_offers) > 1 else 0
        
        mean_anger = anger.mean() if not anger.empty else 0
        max_anger = anger.max() if not anger.empty else 0
        
        # Determine aggregate final anger (average of all final angers across this group)
        final_angers = []
        for exp in subset_df['experiment'].unique():
            an_data = anger[subset_df['experiment'] == exp]
            if not an_data.empty:
                final_angers.append(an_data.iloc[-1])
        final_anger_mean = np.mean(final_angers) if final_angers else 0
        
        return {
            'Group': label_group,
            'Name': label_name,
            'No. Experiments': experiment_count,
            'No. Accepts': accept_count,
            'Mean Offer (B Share)': round(mean_offer, 2),
            'Std Offer (B Share)': round(std_offer, 2),
            'Mean Anger State': round(mean_anger, 4),
            'Max Anger State': round(max_anger, 4),
            'Final Anger State': round(final_anger_mean, 4),
            'Expected Max Offer': round(expected_max_mean, 2)
        }

    # Total Overall Statistics
    results.append(get_aggregated_stats(df, "Overall", "Total"))
    
    # Model-based Statistics
    for model in df['model'].unique():
        model_df = df[df['model'] == model]
        results.append(get_aggregated_stats(model_df, "Model", model))
        
    stats_df = pd.DataFrame(results)
    stats_df.to_csv(stats_path, index=False)
    print(f"Statistics saved to {stats_path}")

def main(results_dir, analysis_dir, model, max_iterations):
    os.makedirs(analysis_dir, exist_ok=True)
    all_files = glob.glob(os.path.join(results_dir, "*_ab_prompts_*.csv"))
    print(f"Found {len(all_files)} files to process in {results_dir}")
    all_results = []
    
    for f in all_files:
        print(f"Processing {os.path.basename(f)}...")
        file_results = process_file(f, model, max_iterations)
        all_results.extend(file_results)
        
    compiled_csv_path = os.path.join(analysis_dir, "compiled_results.csv")
    
    if all_results:
        keys = all_results[0].keys()
        with open(compiled_csv_path, 'w', newline='', encoding='utf-8') as out_f:
            writer = csv.DictWriter(out_f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(all_results)
        print(f"Compiled results saved to {compiled_csv_path}")
        
        df = pd.DataFrame(all_results)
        numeric_cols = ['iteration', 'offer_a_A', 'offer_a_B', 'offer_b_A', 'offer_b_B', 
                        'b_share_offer', 'anger_intensity', 'toxicity', 'anger', 
                        'rudeness', 'calmness', 'rationality']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        create_visualizations(df, analysis_dir)
        compute_statistics(df, analysis_dir)
    else:
        print("No data extracted.")

if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent
    RESULTS_DIR = str(BASE_DIR / "log")
    ANALYSIS_DIR = os.path.join(RESULTS_DIR, "analysis")
    MAX_ITERATIONS = 100
    
    LOCAL_LLM = True
    if LOCAL_LLM:
        _model = ChatOpenAI(
            model="llama4-scout",
            base_url="http://192.168.100.119:13000/v1",
            api_key="local",
            temperature=0.0
        )
    else:
        _model = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
        
    main(RESULTS_DIR, ANALYSIS_DIR, _model, MAX_ITERATIONS)
