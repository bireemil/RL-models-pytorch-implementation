import gym
from utils import Agent
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    env = gym.make('LunarLander-v2',)
    agent = Agent(gamma=0.99,
    epsilon = 1,
    batch_size=64,
    n_actions = 4,
    eps_end = 0.01,
    input_dims = [8],
    lr=0.003)

    scores, eps_history = [], []
    n_games = 500

    for i in range(n_games):
        score = 0
        done = False
        observation, _ = env.reset()
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, _, _ = env.step(action)
            score += reward
            agent.store_transition(observation, action, reward,
                                observation_, done)
            agent.learn()
            observation = observation_
        scores.append(score)
        eps_history.append(agent.epsilon)

        avg_score = np.mean(scores[-100:])

        print(f'episode {i}, score {score:.2f}, average score {avg_score:.2f}')

    