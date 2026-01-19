"""
Utility functions for saving and loading agents.

These functions help create standalone Python files for agent submission.
"""

import os
import re
import ast
import inspect
import numpy as np # type: ignore


def extract_methods_from_source(source_file, method_names):
    """
    Extract method source code from a Python file.
    
    Args:
        source_file: Path to the source .py file
        method_names: List of method names to extract
    
    Returns:
        Dict mapping method name -> source code (with proper indentation)
    """
    with open(source_file, 'r') as f:
        source = f.read()
    
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    
    methods = {}
    
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name in method_names:
                    # Get start and end lines
                    start_line = item.lineno - 1  # 0-indexed
                    end_line = item.end_lineno  # ast uses 1-indexed end
                    
                    # Extract the method source
                    method_lines = lines[start_line:end_line]
                    
                    # Find the base indentation (indentation of the 'def' line)
                    first_line = method_lines[0]
                    base_indent = len(first_line) - len(first_line.lstrip())
                    
                    # Remove base indentation and add 4 spaces for class method
                    reindented_lines = []
                    for line in method_lines:
                        if line.strip():  # Non-empty line
                            # Remove base indent and add 4 spaces
                            if len(line) >= base_indent:
                                reindented_lines.append('    ' + line[base_indent:].rstrip())
                            else:
                                reindented_lines.append('    ' + line.lstrip().rstrip())
                        else:  # Empty line
                            reindented_lines.append('')
                    
                    methods[item.name] = '\n'.join(reindented_lines)
    
    return methods


def extract_init_params_from_source(source_file):
    """
    Extract __init__ parameter assignments from source file.
    
    Returns the body of __init__ as a string (excluding super().__init__() and q_table init).
    """
    with open(source_file, 'r') as f:
        source = f.read()
    
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == '__init__':
                    start_line = item.lineno  # Line after 'def __init__'
                    end_line = item.end_lineno
                    
                    # Get body lines (skip the def line)
                    init_lines = []
                    for stmt in item.body:
                        stmt_start = stmt.lineno - 1
                        stmt_end = stmt.end_lineno
                        stmt_source = ''.join(lines[stmt_start:stmt_end])
                        
                        # Skip super().__init__(), np_random init, and q_table assignments
                        if 'super().__init__' in stmt_source:
                            continue
                        if 'np_random' in stmt_source and 'default_rng' in stmt_source:
                            continue
                        if 'q_table' in stmt_source and '{}' in stmt_source:
                            continue
                            
                        init_lines.append(stmt_source)
                    
                    return ''.join(init_lines)
    
    return ''


def save_qlearning_agent(agent, output_path, agent_class_name="QLearningTrainedAgent", 
                         original_class_name=None, source_file=None):
    """
    Save a trained Q-learning agent as a standalone Python file.
    
    Args:
        agent: The trained Q-learning agent instance
        output_path: Path where to save the agent file
        agent_class_name: Name for the agent class in the saved file
        original_class_name: Name of the original agent class (optional)
        source_file: Path to the original agent source file (optional, for auto-extraction)
    
    Returns:
        None
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Methods to extract from source
    methods_to_extract = [
        'discretize_state', 'act', 'learn', 'reset', 'seed', 'save', 'load'
    ]
    
    # Try to extract methods from source file if provided
    extracted_methods = {}
    init_params = ''
    
    if source_file and os.path.exists(source_file):
        print(f"Extracting methods from {source_file}...")
        extracted_methods = extract_methods_from_source(source_file, methods_to_extract)
        init_params = extract_init_params_from_source(source_file)
        print(f"  Extracted {len(extracted_methods)} methods: {list(extracted_methods.keys())}")
    
    # Start building the file content
    file_content = f'''"""
Q-Learning Agent for the Sailing Challenge - Trained Model

This file contains a Q-learning agent trained on the sailing environment.
The agent uses a discretized state space and a Q-table for decision making.
Auto-generated from: {source_file if source_file else 'N/A'}
"""

import numpy as np
from agents.base_agent import BaseAgent

class {agent_class_name}(BaseAgent):
    """
    A Q-learning agent trained on the sailing environment.
    Uses a discretized state space and a lookup table for actions.
    """
    
    def __init__(self):
        """Initialize the agent with the trained Q-table."""
        super().__init__()
        self.np_random = np.random.default_rng()

{init_params}
        # Q-table with learned values
        self.q_table = {{}}
        self._init_q_table()
    
    def _init_q_table(self):
        """Initialize the Q-table with learned values."""
'''
    
    # Add all Q-values
    for state, values in agent.q_table.items():
        q_values_str = np.array2string(values, precision=4, separator=', ')
        file_content += f"        self.q_table[{state}] = np.array({q_values_str})\n"
    
    # Add extracted methods or fallback to defaults
    for method_name in methods_to_extract:
        if method_name in extracted_methods:
            file_content += f"\n{extracted_methods[method_name]}\n"
        else:
            # Fallback: use default implementations
            file_content += _get_default_method(method_name)
    
    # Write the file
    with open(output_path, 'w') as f:
        f.write(file_content)
    
    print(f"Agent saved to {output_path}")
    print(f"The file contains {len(agent.q_table)} state-action pairs.")
    print(f"You can now use this file with validate_agent.ipynb and evaluate_agent.ipynb")


def _get_default_method(method_name):
    """Return default implementation for a method if not found in source."""
    defaults = {
        'discretize_state': '''
    def discretize_state(self, observation):
        """Discretize the continuous observation into a tuple state."""
        x, y = observation[0], observation[1]
        vx, vy = observation[2], observation[3]
        wx, wy = observation[4], observation[5]
        wind_flattened = observation[6:]

        x_bin = min(int(x / self.grid_size * self.position_bins), self.position_bins - 1)
        y_bin = min(int(y / self.grid_size * self.position_bins), self.position_bins - 1)

        v_mag = np.sqrt(vx**2 + vy**2)
        if v_mag < 0.1:
            v_bin = 0
        else:
            v_dir = np.arctan2(vy, vx)
            v_bin = int(((v_dir + np.pi) / (2 * np.pi) * (self.velocity_bins - 1)) + 1) % self.velocity_bins

        wind_angle = np.arctan2(wy, wx)
        wind_bin = int(((wind_angle + np.pi) / (2 * np.pi)) * self.wind_bins) % self.wind_bins

        return (x_bin, y_bin, v_bin, wind_bin)
''',
        'act': '''
    def act(self, observation, info=None):
        """Select an action using the Q-table."""
        state = self.discretize_state(observation)
        if state in self.q_table:
            return np.argmax(self.q_table[state])
        return self.np_random.integers(0, 9)
''',
        'learn': '''
    def learn(self, state, action, reward, next_state):
        """Learning is disabled for exported agent."""
        pass
''',
        'reset': '''
    def reset(self):
        self.last_state = None
        self.last_action = None
''',
        'seed': '''
    def seed(self, seed=None):
        self.np_random = np.random.default_rng(seed)
''',
        'save': '''
    def save(self, path):
        import pickle
        with open(path, "wb") as f:
            pickle.dump({
                'q_table': self.q_table,
                'exploration_rate': getattr(self, 'exploration_rate', 0.01)
            }, f)
''',
        'load': '''
    def load(self, path):
        import pickle
        with open(path, "rb") as f:
            data = pickle.load(f)
            self.q_table = data['q_table']
            self.exploration_rate = data.get('exploration_rate', 0.01)
'''
    }
    return defaults.get(method_name, '')