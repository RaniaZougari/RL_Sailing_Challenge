import os
import sys
sys.path.append(".")

from agents.my_agent_physics import MyAgent
from utils.agent_utils import save_qlearning_agent

agent = MyAgent()

# IMPORTANT: load trained Q-table
agent.load("models/physics_agent.pkl")

output_file = "Submission_agents/my_physics_agent.py"



save_qlearning_agent(agent, output_file)

print(f"✅ Submission agent saved to {output_file}")