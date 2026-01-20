import os
import json
import glob

RESULTS_DIR = '/home/unmars/Downloads/RL_Sailing_Challenge/results'

def calculate_new_score(mean_steps, success_rate):
    # Formula: 100 * 0.99^steps * success_rate
    # success_rate is expected to be 0.0-1.0 from the JSON 'evaluation' section
    return 100 * (0.99 ** mean_steps) * success_rate

def convert_file(filepath):
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print(f"Skipping {filepath}: Invalid JSON")
        return
    except Exception as e:
        print(f"Skipping {filepath}: {e}")
        return

    if 'evaluation' not in data:
        print(f"Skipping {filepath}: No 'evaluation' key")
        return

    modified = False
    
    for scenario_name, metrics in data['evaluation'].items():
        if not isinstance(metrics, dict):
            continue
            
        mean_steps = metrics.get('mean_steps')
        success_rate = metrics.get('success_rate')
        
        if mean_steps is None or success_rate is None:
            print(f"Skipping scenario {scenario_name} in {filepath}: Missing metrics")
            continue
            
        # Calculate new score
        new_score = calculate_new_score(mean_steps, success_rate)
        
        # Save old score if not already saved
        if 'custom_score' in metrics:
            if 'previous_custom_score' not in metrics:
                metrics['previous_custom_score'] = metrics['custom_score']
                modified = True
        
        # Update to new score
        metrics['custom_score'] = new_score
        metrics['custom_score_formula'] = "100 * 0.99^steps * success_rate"
        
        # Optional: Add expected_score_100pct for clarity if not present, but main goal is custom_score
        # metrics['expected_score_100pct'] = new_score 
        
        # Check if values changed significantly to avoid noise?? 
        # Actually just overwrite to ensure consistency.
        modified = True

    if modified:
        try:
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"Updated {filepath}")
        except Exception as e:
            print(f"Failed to write {filepath}: {e}")
    else:
        print(f"No changes for {filepath}")

def main():
    json_files = glob.glob(os.path.join(RESULTS_DIR, '*.json'))
    print(f"Found {len(json_files)} JSON files in {RESULTS_DIR}")
    
    for json_file in json_files:
        convert_file(json_file)

if __name__ == "__main__":
    main()
