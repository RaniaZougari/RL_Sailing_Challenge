#!/usr/bin/env python3
"""
Unified Pipeline for RL Agent Development

Automates the complete workflow:
1. Train agent
2. Export to standalone file
3. Evaluate performance
4. Log results

Usage:
    python Scripts/run_pipeline.py train+eval --agent my_agent_physics --scenario training_1
    python Scripts/run_pipeline.py eval --model src/models/physics_agent.pkl --scenarios training_1,training_2
"""

import argparse
import json
import os
import sys
import subprocess
from datetime import datetime
from pathlib import Path
import importlib.util

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))


def import_agent_class(agent_name: str):
    """Dynamically import agent class by name."""
    module_path = project_root / 'src' / 'agents' / f'{agent_name}.py'
    
    spec = importlib.util.spec_from_file_location(agent_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load agent from {module_path}")
    
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    # Find the agent class (should inherit from BaseAgent)
    from agents.base_agent import BaseAgent
    agent_classes = [
        cls for name, cls in module.__dict__.items()
        if isinstance(cls, type) and issubclass(cls, BaseAgent) and cls != BaseAgent
    ]
    
    if not agent_classes:
        raise ValueError(f"No BaseAgent subclass found in {module_path}")
    
    return agent_classes[0]


def train_stage(args, results):
    """Execute training stage."""
    print("\n" + "="*60)
    print("STAGE 1: TRAINING")
    print("="*60)
    
    # Import training function
    sys.path.insert(0, str(project_root / 'Scripts'))
    from training_agent import train_agent
    
    # Get agent class
    agent_class = import_agent_class(args.agent)
    agent = agent_class()
    
    # Setup paths
    model_dir = project_root / 'src' / 'models'
    model_dir.mkdir(exist_ok=True, parents=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    # Handle multiple scenarios in filename
    scenario_str = '_'.join(args.scenario) if isinstance(args.scenario, list) else args.scenario
    if len(args.scenario) > 1:
        scenario_str = f"multi_{len(args.scenario)}scenarios"
    model_path = model_dir / f"{args.agent}_{scenario_str}_{timestamp}.pkl"
    
    scenarios = args.scenario if isinstance(args.scenario, list) else [args.scenario]
    print(f"Training {args.agent} on {len(scenarios)} scenario(s) for {args.episodes} episodes...")
    print(f"Scenarios: {', '.join(scenarios)}")
    print(f"Model will be saved to: {model_path}")
    
    # Extract hyperparameters from agent for logging
    try:
        hyperparameters = {}
        # List of attributes to look for
        attributes_to_log = [
            'learning_rate', 'discount_factor', 'exploration_rate', 
            'exploration_decay', 'exploration_rate_min', 
            'position_bins', 'velocity_bins', 'wind_bins', 
            'wind_preview_steps', 'grid_size', 'q_init_high'
        ]
        
        for attr in attributes_to_log:
            if hasattr(agent, attr):
                hyperparameters[attr] = getattr(agent, attr)
                
        results['hyperparameters'] = hyperparameters
        print(f"Captured {len(hyperparameters)} hyperparameters for logging")
    except Exception as e:
        print(f"Warning: Failed to capture hyperparameters: {e}")

    
    # Train
    training_results = train_agent(
        agent=agent,
        wind_scenarios=scenarios,  # Pass list of scenarios
        num_episodes=args.episodes,
        save_path=str(model_path),
        seed=args.seed
    )
    
    # Store results
    results['training'] = {
        'scenarios': scenarios,
        'episodes': args.episodes,
        'final_success_rate': sum(training_results['success']) / len(training_results['success']),
        'avg_reward': sum(training_results['rewards']) / len(training_results['rewards']),
        'avg_steps': sum(training_results['steps']) / len(training_results['steps']),
        'q_table_size': len(agent.q_table) if hasattr(agent, 'q_table') else 0,
        'scenario_metrics': training_results.get('scenario_metrics', {})
    }
    results['model_path'] = str(model_path)
    
    print(f"\n✅ Training complete!")
    print(f"   Success rate: {results['training']['final_success_rate']:.1%}")
    print(f"   Avg reward: {results['training']['avg_reward']:.1f}")
    
    return str(model_path)


def export_stage(args, results, model_path=None):
    """Execute export stage using agent_utils.save_qlearning_agent."""
    print("\n" + "="*60)
    print("STAGE 2: EXPORT")
    print("="*60)
    
    if model_path is None:
        model_path = args.model
    
    print(f"Exporting {args.agent} to standalone file using agent_utils...")
    
    # Import agent_utils
    sys.path.insert(0, str(project_root / 'src'))
    from utils.agent_utils import save_qlearning_agent
    
    # Setup output path
    output_dir = project_root / 'src' / 'Submission_agents'
    output_dir.mkdir(exist_ok=True, parents=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = output_dir / f"{args.agent}_exported_{timestamp}.py"
    
    try:
        # Load the agent to get its class and q-table
        agent_class = import_agent_class(args.agent)
        agent = agent_class()
        
        if model_path:
            print(f"Loading model from {model_path}...")
            # We need to manually load data if agent.load expects specific format
            # or trust agent.load works. MyAgent.load works with file path.
            agent.load(model_path)
        else:
            print("Warning: No model path provided, exporting untested agent.")

        # Use the proven agent_utils function
        save_qlearning_agent(
            agent=agent,
            output_path=str(output_path),
            agent_class_name='MyAgentTrained',
            original_class_name=agent.__class__.__name__,
            source_file=str(project_root / 'src' / 'agents' / f'{args.agent}.py')
        )
        
        print(f"✅ Success! Exported to {output_path}")
        results['exported_path'] = str(output_path)
        return str(output_path)
        
    except Exception as e:
        print(f"❌ Export failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def eval_stage(args, results, exported_path=None):
    """Execute evaluation stage."""
    print("\n" + "="*60)
    print("STAGE 3: EVALUATION")
    print("="*60)
    
    if exported_path is None:
        # If no exported path, we need to export first
        if not hasattr(args, 'model') or not args.model:
            print("❌ No model specified for evaluation")
            return
        exported_path = export_stage(args, results, args.model)
        if not exported_path:
            return
    
    # Parse scenarios
    # Determine scenarios to run
    if hasattr(args, 'scenarios') and args.scenarios:
        scenarios = args.scenarios.split(',')
    else:
        scenarios = args.scenario
    
    print(f"Evaluating on scenarios: {scenarios}")
    
    eval_results = {}
    
    import tempfile
    import json
    
    # Create temp file for JSON results
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp:
        json_output_path = tmp.name
    
    for scenario in scenarios:
        print(f"\n📊 Evaluating on {scenario}...")
        
        cmd = [
            sys.executable,
            str(project_root / 'src' / 'evaluate_submission.py'),
            str(exported_path),
            '--wind_scenario', scenario,
            '--seeds', str(args.eval_seeds),
            '--num-seeds', str(args.eval_num_seeds),
            '--output-json', json_output_path,
            '--verbose'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(result.stdout)
        
        # Parse results 
        scenario_results = {
            'output': result.stdout,
            'success': result.returncode == 0
        }
        
        # Try to read JSON output
        try:
            if os.path.exists(json_output_path) and os.path.getsize(json_output_path) > 0:
                with open(json_output_path, 'r') as f:
                    json_data = json.load(f)
                    
                    # Merge structured data
                    if 'scenarios' in json_data and scenario in json_data['scenarios']:
                        scenario_data = json_data['scenarios'][scenario]
                        scenario_results.update(scenario_data)
                    
                    if 'overall' in json_data:
                        # Add overall data (custom score particularly)
                        scenario_results.update({k: v for k, v in json_data['overall'].items() if k not in scenario_results})
                    
                    # If we successfully loaded structured data, remove the verbose raw output
                    if 'output' in scenario_results:
                        del scenario_results['output']
                        
        except Exception as e:
            print(f"Warning: Could not read structured results: {e}")

        eval_results[scenario] = scenario_results
        
    # Clean up
    if os.path.exists(json_output_path):
        os.remove(json_output_path)
    
    results['evaluation'] = eval_results
    
    print(f"\n✅ Evaluation complete!")


def log_results(args, results):
    """Save results to JSON file."""
    print("\n" + "="*60)
    print("STAGE 4: LOGGING")
    print("="*60)
    
    # Setup results directory
    results_dir = project_root / 'results'
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = results_dir / f"{timestamp}_{args.agent}.json"
    
    # Add metadata
    results['metadata'] = {
        'agent': args.agent,
        'timestamp': timestamp,
        'command': ' '.join(sys.argv)
    }
    
    # Save
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"📝 Results saved to: {results_file}")
    print(f"\n" + "="*60)
    print("PIPELINE COMPLETE!")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description='Unified pipeline for RL agent training, export, and evaluation'
    )
    
    parser.add_argument(
        'mode',
        choices=['train', 'export', 'eval', 'train+eval', 'export+eval', 'train+export+eval'],
        help='Pipeline stage(s) to run'
    )
    
    parser.add_argument('--agent', help='Agent name (e.g., my_agent_physics)')
    parser.add_argument(
        '--scenario',
        nargs='+',
        help='Wind scenario(s) for training (space-separated, e.g., training_1 training_2)'
    )
    parser.add_argument('--scenarios', help='Comma-separated scenarios for evaluation')
    parser.add_argument('--episodes', type=int, default=100, help='Training episodes')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--model', help='Path to existing model file')
    parser.add_argument('--eval-seeds', type=int, default=1, help='Starting seed for evaluation')
    parser.add_argument('--eval-num-seeds', type=int, default=100, help='Number of eval seeds')
    
    args = parser.parse_args()
    
    # Validate arguments
    if 'train' in args.mode and (not args.agent or not args.scenario):
        parser.error("--agent and --scenario required for training")
    
    if 'eval' in args.mode and not args.scenarios and not args.scenario:
        parser.error("--scenario or --scenarios required for evaluation")
    
    # Initialize results dict
    results = {}
    model_path = None
    exported_path = None
    
    try:
        # Execute pipeline stages
        if 'train' in args.mode:
            model_path = train_stage(args, results)
        
        if 'export' in args.mode or 'eval' in args.mode:
            exported_path = export_stage(args, results, model_path)
        
        if 'eval' in args.mode:
            eval_stage(args, results, exported_path)
        
        # Always log results
        log_results(args, results)
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
