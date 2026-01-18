import os
import sys
sys.path.append(".")

from agents.my_agent import MyAgent
from utils.agent_utils import save_qlearning_agent

agent = MyAgent()

# IMPORTANT: load trained Q-table
agent.load("models/trained_my_agent.pkl")

output_file = "src/agents/my_agent_submission.py"
save_qlearning_agent(agent, output_file)

print(f"✅ Submission agent saved to {output_file}")