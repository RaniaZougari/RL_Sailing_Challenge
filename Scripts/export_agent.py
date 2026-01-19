import os
import sys
sys.path.append(".")

from agents.my_agent_DQ import MyAgent
from utils.agent_utils import save_qlearning_agent

agent = MyAgent()

# IMPORTANT: load trained Q-table
agent.load("models/trainedDQ_FINAL2.pkl")

output_file = "Submission_agents/trained_FINAL_mieux.py"



save_qlearning_agent(agent, output_file)

print(f"✅ Submission agent saved to {output_file}")