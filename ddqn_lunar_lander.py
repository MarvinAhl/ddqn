from ddqn import DDQN
import torch
import numpy as np
import matplotlib.pyplot as plt
import gym
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

backup_episodes = 10

env = gym.make('LunarLander-v2')

nS = env.observation_space.shape[0]
nA = env.action_space.n

agent = DDQN(nS, nA, hidden_layers=(500, 500, 500), gamma=0.99, learning_rate_start=0.0005, learning_rate_decay_steps=20000, learning_rate_min=0.0003,
             epsilon_start=1.0, epsilon_decay_steps=20000, epsilon_min=0.1, buffer_size_min=200, buffer_size_max=50000, batch_size=50, replays=1, tau=0.01, device=device)

episode_rewards = []
step_count = 0
steps = []


def save_stats(curr_episode):
    global agent
    global episodes
    global episode_rewards
    global steps

    agent.save_net('lunar_lander.net')

    plt.clf()  # To prevent overlapping of old plots

    plt.subplot(211)
    plt.plot(np.arange(curr_episode) + 1, episode_rewards, 'k-')
    plt.grid(True)
    plt.title('Training Stats')
    plt.ylabel('Reward')

    plt.subplot(212)
    plt.plot(np.arange(curr_episode), steps, 'k-')
    plt.grid(True)
    plt.xlabel('Episodes')
    plt.ylabel('Steps')

    plt.savefig('rewards_train.png')

    print(f'Episode {curr_episode} saved')

episode = 0
while True:
    episode += 1
    obsv, done = env.reset(), False

    episode_reward = 0.0
    while not done:
        env.render()

        action = agent.act(obsv)
        new_obsv, reward, done, info = env.step(action)
        time_out = 'TimeLimit.truncated' in info and info['TimeLimit.truncated']
        terminal = done and not time_out

        agent.experience(obsv, action, reward, new_obsv, terminal)
        agent.train()

        obsv = new_obsv
        
        episode_reward += reward  # For statistics
        step_count += 1
    
    episode_rewards.append(episode_reward)
    steps.append(step_count)

    if episode % backup_episodes == 0:
        save_stats(episode)

env.close()

save_stats(episode)