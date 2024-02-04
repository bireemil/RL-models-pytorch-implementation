import os

import gym
import numpy as np
from matplotlib import pyplot as plt
from utils import Agent
from torch.utils.tensorboard import SummaryWriter



if __name__ == '__main__' :

    os.makedirs('plots', exist_ok=True)
    os.makedirs(os.path.join('tmp','ppo'), exist_ok=True)

    env = gym.make('LunarLander-v2')
    N=1024
    batch_size = 64
    n_epochs = 4
    alpha = 0.00025
    gamma=0.999
    gae_lambda=0.98
    fc1_dims=128
    fc2_dims=128
    agent = Agent(n_actions=env.action_space.n, batch_size=batch_size,
                  alpha=alpha, n_epochs=n_epochs,
                  input_dims=env.observation_space.shape,
                  policy_clip=0.2, gamma=gamma, gae_lambda=gae_lambda,
                  fc1_dims=fc1_dims, fc2_dims=fc2_dims)
    # agent.load_models()
    n_games = 50000
    figure_file = 'plots/cartpole.png'

    best_score = env.reward_range[0]
    score_history = []

    learn_iters = 0
    avg_score = 0
    n_steps = 0

    writer = SummaryWriter(os.path.join("runs",f"{N}_{batch_size}_{n_epochs}_{alpha}_{gamma}_{gae_lambda}_{fc1_dims}_{fc2_dims}_l1"))

    for i in range(n_games):
        ep_steps = 0
        observation = env.reset()[0]
        done = False
        score = 0
        total_loss, actor_loss, critic_loss = np.nan,np.nan,np.nan
        while not done :

            action, prob, val = agent.choose_action(observation)
            observation_, reward, done, info, _ = env.step(action)
            ep_steps+=1
            # env.render()
            n_steps +=1
            score += reward
            agent.remember(observation, action, prob,val, reward, done)
            if n_steps % N == 0 and n_steps!=0:
                total_loss, actor_loss, critic_loss = agent.learn()
                learn_iters +=1

            observation = observation_
            if ep_steps>500 :
                done = True
        score_history .append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()

        writer.add_scalar('Loss/total', total_loss, i)
        writer.add_scalar('Loss/critic_loss', critic_loss, i)
        writer.add_scalar('Loss/actor_loss', actor_loss, i)
        writer.add_scalar('Score/average_score', avg_score, i)

        print(f"episode {i}, score {score}, avg score {avg_score},\
              time_steps {n_steps}, learning_states {learn_iters}")

    # x = [i+1 for i in range((len(score_history)))]
    # plt.plot(x, score_history)
    # plt.show()
