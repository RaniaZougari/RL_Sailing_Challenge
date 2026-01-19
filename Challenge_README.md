This document will be used for teacher \> students communication and will feature answers to frequently asked questions, as well as possible remarks on the course content.  
If you encounter problems, or have specific questions related to the challenge you can email me at t.rahier@criteo.com

# Challenge-related questions

Q: **I tried to submit an agent on codabench but it failed. Why?**  
A: The first thing to check is that your agent is valid (see validation notebook, or associated command line) and that it can properly be evaluated locally (on your machine) on the training wind scenarios (see evaluation notebook, or associated command line).  
The second thing to check (if the above works) is that you are indeed importing BaseAgent **from evaluator.base\_agent** and not from agents.base\_agents. For your local training, BaseAgent is imported from agents.base\_agent (as it is coded that way in the repo) but codabench has a specific evaluator file, hence the change in import.  
The third thing to check is that the agent you are submitting is named **MyAgent**. Indeed, codabench searches your .zip for a MyAgent class (inheriting from BaseAgent).

Q: **Can we make several submissions on codabench?**  
A: YES. The limit is 5 submissions per student per day (100 max submissions per student overall). It is advised that you submit several times as this is the only way you will be able to evaluate your agent on the (hidden) test wind scenario. Moreover, only the best agent(s) will be taken into account for the evaluation, so you should feel free to submit agents even if they are simple and not very performant.

Q: **If I submit several agents, which one(s) will be taken into account for evaluation?**  
A: Only the best performing agent on the (hidden) test wind scenario will be taken into account. This is also the agent that will appear automatically in the leaderboard (“Results” tab on codabench). The only exception is: if your best performing agent was not learned using RL (but is rule-based for example), then your best “RL-based” agent will also be considered for evaluation.

Q: **How will students be evaluated?**  
A: The evaluation of each student on the project will be based on two elements:

1) The best submitted agent. The three criteria which will be taken into account are, in descending order of importance, (i) the performance of the agent on the (hidden) test wind scenario, (ii) the performance of the agent on the three training wind scenarios and (iii) the fact that the agent was learned using RL.  
2) A short (PDF format) report describing the student’s approach to solve the challenge. In this report, the student should describe what was tried, what worked and did not work, as well as what the student would do if (s)he was given more time and/or more computing power. This report should be attached to one of the student submissions (as part of the submitted .zip on codabench) or sent by email at t.rahier@criteo.com. It is not necessary to attach a report to each submission. If several reports are attached by a single student, only the last one which was submitted will be taken into account.

Q: **How can I submit my report?**  
You have two choices: either submit it via codabench (in your last zip file), either send it by email to t.rahier@criteo.com

Q: **How can students be identified?**  
A: It is advised you use your true name when subscribing on codabench. In case your username is not explicit and if your true name has not been registered, I will send a confirmation email to ensure your identity (it is therefore important that you provide a valid email at registration time).

Q: **Are my submissions automatically visible on the public leaderboard?**  
In the first weeks of challenge, no: it is your choice to make your agent appear (or not) in the leaderboard. Starting January 14th, your \*best\* agent will automatically appear in the leaderboard.

# 

# Some advice on agent design

**Keep calm**  
Don’t worry if your agent does not perform as well as you would have hoped. Make sure to describe the different approaches and ideas that you tried in your report, this will be taken into account for your final grade.

**Reward shaping**  
You can “help” your agent by using reward-shaping to give it a bit of reward as it gets closer to the goal, even if it does not reach it. The fact the reward is very sparse makes the learning a lot more difficult.

**Local wind**  
It is possible to design agents that perform very well (although they might not be the best) using only local wind information (and of course, boat position, speed and direction of the goal). The agent I have submitted (trahier on the leaderboard) is one of those.

**How favorable is the wind?**  
The challenge walkthrough notebook gives you information on how efficiently your boat moves depending on the angle between the (local) wind and the boat’s chosen direction (see right picture).

This information can be used to help agents choose actions. It is of course not necessary to use this information explicitly in the design / training of the agent (some top performing agents that have been submitted to this challenge in the past did not).

# Useful Links

Link to the codabench platform: ​​[https://www.codabench.org/competitions/12126/?secret\_key=ac57bd02-5c94-4b97-bccf-2e9e7eaf1c88](https://www.codabench.org/competitions/12126/?secret_key=ac57bd02-5c94-4b97-bccf-2e9e7eaf1c88)

Link to the repo: [https://github.com/trahier/RL\_project\_sailing](https://github.com/trahier/RL_project_sailing)