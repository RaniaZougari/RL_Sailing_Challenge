#!/usr/bin/env python3
"""
Evaluate all models found in results/*.json on ALL scenarios.

Strategy:
1. Read each result JSON file.
2. Use the `exported_path` field to find the exported .py agent.
3. Identify which scenarios are already evaluated in `evaluation` dict.
4. Run `evaluate_submission.py` only for missing scenarios.
5. Update the original JSON file in-place with the new evaluation data.
"""
import os
import json
import glob
import subprocess
import sys
import tempfile

RESULTS_DIR = '/home/unmars/Downloads/RL_Sailing_Challenge/results'
SRC_DIR = '/home/unmars/Downloads/RL_Sailing_Challenge/src'
ALL_SCENARIOS = ['simple_static', 'static_headwind', 'training_1', 'training_2', 'training_3']


def calculate_custom_score(mean_steps, success_rate):
    """Calculate the custom score using the professor's formula."""
    return 100 * (0.99 ** mean_steps) * success_rate


def main():
    json_files = glob.glob(os.path.join(RESULTS_DIR, '*points.json'))
    print(f"Found {len(json_files)} result files. Scanning for missing evaluations...")

    for filepath in sorted(json_files):
        filename = os.path.basename(filepath)
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
        except Exception as e:
            print(f"[SKIP] {filename}: Cannot read JSON ({e})")
            continue

        exported_path = data.get('exported_path')
        if not exported_path:
            print(f"[SKIP] {filename}: No 'exported_path' field")
            continue

        if not os.path.exists(exported_path):
            print(f"[SKIP] {filename}: Exported file not found: {exported_path}")
            continue

        # Determine which scenarios are already evaluated
        existing_eval = data.get('evaluation', {})
        evaluated_scenarios = set(existing_eval.keys())
        missing_scenarios = [s for s in ALL_SCENARIOS if s not in evaluated_scenarios]

        if not missing_scenarios:
            print(f"[OK] {filename}: All scenarios already evaluated.")
            continue

        print(f"\n[EVAL] {filename}")
        print(f"       Exported agent: {os.path.basename(exported_path)}")
        print(f"       Missing scenarios: {missing_scenarios}")

        # Run evaluation for each missing scenario
        for scenario in missing_scenarios:
            print(f"       -> Evaluating on {scenario}...", end=" ", flush=True)

            # Create temp file for JSON output
            with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp:
                json_output_path = tmp.name

            cmd = [
                sys.executable,
                os.path.join(SRC_DIR, 'evaluate_submission.py'),
                exported_path,
                '--wind_scenario', scenario,
                '--seeds', '1',
                '--num-seeds', '100',
                '--output-json', json_output_path
            ]

            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
                print(result)
                if result.returncode != 0:
                    print(f"FAILED (exit code {result.returncode})")
                    if result.stderr:
                        print(f"           Error: {result.stderr[:200]}")
                    continue

                # Read the output JSON
                if os.path.exists(json_output_path) and os.path.getsize(json_output_path) > 0:
                    with open(json_output_path, 'r') as f:
                        eval_data = json.load(f)

                    # Extract scenario-specific data
                    if 'scenarios' in eval_data and scenario in eval_data['scenarios']:
                        scenario_data = eval_data['scenarios'][scenario]
                        
                        # Add success flag and custom score
                        scenario_data['success'] = scenario_data.get('success_rate', 0) > 0
                        
                        mean_steps = scenario_data.get('mean_steps', 1000)
                        success_rate = scenario_data.get('success_rate', 0)
                        scenario_data['custom_score'] = calculate_custom_score(mean_steps, success_rate)
                        scenario_data['custom_score_formula'] = "100 * 0.99^steps * success_rate"
                        
                        # Store in the original data
                        if 'evaluation' not in data:
                            data['evaluation'] = {}
                        data['evaluation'][scenario] = scenario_data
                        
                        print(f"OK (score: {scenario_data['custom_score']:.2f})")
                    else:
                        print(f"FAILED (no scenario data in output)")
                else:
                    print(f"FAILED (no output file)")

            except subprocess.TimeoutExpired:
                print(f"TIMEOUT")
            except Exception as e:
                print(f"ERROR ({e})")
            finally:
                # Cleanup temp file
                if os.path.exists(json_output_path):
                    os.remove(json_output_path)

        # Save updated data back to the original file
        try:
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"       ✅ Updated {filename}")
        except Exception as e:
            print(f"       ❌ Failed to save {filename}: {e}")

    print("\n✅ All evaluations complete.")


if __name__ == "__main__":
    main()
