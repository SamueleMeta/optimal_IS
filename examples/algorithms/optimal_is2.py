"""Working example of REPS."""
import numpy as np
import torch

from rllib.agent import REPSAgent
from rllib.environment.mdps import RandomMDP
from rllib.util.training.agent_training import evaluate_agent, train_agent

from rllib.policy import TabularPolicy
from rllib.value_function import TabularValueFunction

ETA = 1.0
NUM_EPISODES = 15

GAMMA = 0.99
SEED = 0
#ENVIRONMENT = "FrozenLake-v0"
MAX_STEPS = 10

torch.manual_seed(SEED)
np.random.seed(SEED)

environment = RandomMDP(num_states=5, num_actions=5)

#print(environment.transitions)

critic = TabularValueFunction.default(environment)
policy = TabularPolicy.default(environment)

#print(policy.table)

agent = REPSAgent.default(environment, epsilon=ETA, regularization=True, gamma=GAMMA, critic=critic, policy=policy)
train_agent(agent, environment, num_episodes=NUM_EPISODES, max_steps=MAX_STEPS + 1)
evaluate_agent(agent, environment, num_episodes=1, max_steps=MAX_STEPS + 1)
