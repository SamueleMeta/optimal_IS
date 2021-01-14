"""Working example of REPS."""
import numpy as np
import torch

from rllib.agent import REPSAgent
from rllib.environment.mdps import RandomMDP
from rllib.util.training.agent_training import train_agent

from rllib.policy import TabularPolicy
from rllib.value_function import TabularValueFunction

from rllib.algorithms.performance import evaluate_performance
from rllib.util.rollout import step_env

from tqdm import tqdm

def rollout_episode(environment, agent, max_steps, render, gamma):
    """Rollout a full episode."""
    state = environment.reset()
    agent.set_goal(environment.goal)
    agent.start_episode()
    done = False
    time_step = 0

    ret = 0.
    while not done:
        action = agent.act(state)
        obs, state, done, info = step_env(
            environment=environment,
            state=state,
            action=action,
            action_scale=agent.policy.action_scale,
            pi=agent.pi,
            render=render,
        )

        ret += obs.reward * gamma ** time_step
        agent.observe(obs)
        # Log info.
        agent.logger.update(**info)

        time_step += 1
        if max_steps <= time_step:
            break

    agent.end_episode()

    return ret


def evaluate_agent(agent, environment, num_episodes, max_steps, gamma):
    rets = []
    for i in tqdm(range(num_episodes)):
        ret = rollout_episode(environment, agent, max_steps, False, gamma)
        rets.append(ret)

    print("\tEmpirical performance: %s" % np.mean(rets))
    return np.mean(rets)

ETA = 1.0
NUM_EPISODES = 15

GAMMA = 0.99
SEED = 0
#ENVIRONMENT = "FrozenLake-v0"
MAX_STEPS = 1000

torch.manual_seed(SEED)
np.random.seed(SEED)

environment = RandomMDP(num_states=10, num_actions=5)

#print(environment.transitions)

critic = TabularValueFunction.default(environment)
policy = TabularPolicy.default(environment)

#print(policy.table)

agent = REPSAgent.default(environment, epsilon=ETA, regularization=True, gamma=GAMMA, critic=critic, policy=policy)

print('Before training...')
evaluate_agent(agent, environment, num_episodes=10, max_steps=MAX_STEPS + 1, gamma=GAMMA)
evaluate_performance(environment, policy, GAMMA, agent)

train_agent(agent, environment, num_episodes=NUM_EPISODES, max_steps=MAX_STEPS + 1, plot_flag=False)

print('After training...')
evaluate_agent(agent, environment, num_episodes=10, max_steps=MAX_STEPS + 1, gamma=GAMMA)
res = evaluate_performance(environment, policy, GAMMA, agent)
