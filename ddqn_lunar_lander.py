from ddqn import DDQN
import torch
import numpy as np
import matplotlib.pyplot as plt
import gym
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

episodes = 100
backup_episodes = 10

env = gym.make('LunarLander-v2')

nS = env.observation_space.shape[0]
nA = env.action_space.n

agent = DDQN(nS, nA, hidden_layers=(500, 500, 500), gamma=0.99, learning_rate_start=0.0003, learning_rate_decay_steps=1, learning_rate_min=0.0003,
             epsilon_start=0.1, epsilon_decay_steps=1, epsilon_min=0.1, buffer_size_min=200, buffer_size_max=50000, batch_size=50, replays=1, tau=0.01, device=device)
agent.load_net('lunar_lander_a2_r3.net')

episode_rewards = np.zeros(episodes, dtype=np.float32)
step_count = 0
steps = np.zeros(episodes, dtype=np.int32)


def save_stats(curr_episode):
    global agent
    global episodes
    global episode_rewards
    global steps

    agent.save_net('lunar_lander_a2_r3_v2.net')

    plt.clf()  # To prevent overlapping of old plots

    plt.subplot(211)
    plt.plot(np.arange(curr_episode), episode_rewards[:curr_episode], 'k-')
    plt.grid(True)
    plt.title('Training Stats')
    plt.ylabel('Reward')

    plt.subplot(212)
    plt.plot(np.arange(curr_episode), steps[:curr_episode], 'k-')
    plt.grid(True)
    plt.xlabel('Episodes')
    plt.ylabel('Steps')

    plt.savefig('rewards_train.png')


for episode in tqdm(range(episodes)):
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
    
    episode_rewards[episode] = episode_reward
    steps[episode] = step_count

    if episode % backup_episodes == 0:
        save_stats(episode+1)

env.close()

save_stats(episodes)