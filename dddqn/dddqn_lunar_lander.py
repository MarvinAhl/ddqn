from dddqn import DDDQN
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

agent = DDDQN(nS, nA, hidden_layers=(500, 500, 500), gamma=0.99, learning_rate_start=0.0005, learning_rate_decay_steps=20000, learning_rate_min=0.0003,
             epsilon_start=1.0, epsilon_decay_steps=20000, epsilon_min=0.1, temp_start=10, temp_decay_steps=20000, temp_min=0.1, buffer_size_min=200,
             buffer_size_max=50000, batch_size=50, replays=1, tau=0.01, device=device)

episode_rewards = []
steps = []
greedy_rates = []


def save_stats(curr_episode):
    global agent
    global episodes
    global episode_rewards
    global steps

    agent.save_net('lunar_lander.net')

    plt.clf()  # To prevent overlapping of old plots

    figure, axis = plt.subplots(3, 1)

    axis[0].plot(np.arange(curr_episode) + 1, episode_rewards, 'k-')
    axis[0].grid(True)
    axis[0].set_title('Training Stats')
    axis[0].set_ylabel('Reward')

    axis[1].plot(np.arange(curr_episode), steps, 'k-')
    axis[1].grid(True)
    axis[1].set_ylabel('Steps')

    axis[2].plot(np.arange(curr_episode), greedy_rates, 'k-')
    axis[2].grid(True)
    axis[2].set_ylabel('Greedy Rate')
    axis[2].set_xlabel('Episodes')

    plt.savefig('rewards_train.png')

    plt.close(figure)

    print(f'Episode {curr_episode} saved')

episode = 0
while True:
    episode += 1
    obsv, done = env.reset(), False

    episode_reward = 0.0
    step_count = 0
    greedy_count = 0
    while not done:
        env.render()

        action, is_greedy = agent.act_softmax(obsv)
        new_obsv, reward, done, info = env.step(action)
        time_out = 'TimeLimit.truncated' in info and info['TimeLimit.truncated']
        terminal = done and not time_out

        agent.experience(obsv, action, reward, new_obsv, terminal)
        agent.train()

        obsv = new_obsv
        
        episode_reward += reward  # For statistics
        step_count += 1
        greedy_count += is_greedy
    
    episode_rewards.append(episode_reward)
    overall_steps = step_count if len(steps) < 1 else steps[-1] + step_count
    steps.append(overall_steps)
    greedy_rates.append(greedy_count / step_count)

    if episode % backup_episodes == 0:
        save_stats(episode)

env.close()

save_stats(episode)