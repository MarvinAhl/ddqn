from dddqn import DDDQN
import torch
import numpy as np
import matplotlib.pyplot as plt
import gym
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f'Using device {device}')

training_runs = 6
episodes = 1000
backup_episodes = 250

env = gym.make('LunarLander-v2')

nS = env.observation_space.shape[0]
nA = env.action_space.n

agent0 = DDDQN(nS, nA, hidden_layers=(1000, 2000, 2000, 2000, 1000), gamma=0.99, learning_rate_start=0.0005, learning_rate_decay_steps=50000, learning_rate_min=0.0003,
              weight_decay=0.0, epsilon_start=1.0, epsilon_decay_steps=20000, epsilon_min=0.1, temp_start=10, temp_decay_steps=20000, temp_min=0.1, buffer_size_min=200,
              buffer_size_max=50000, batch_size=50, replays=1, tau=0.01, alpha=0.6, beta=0.1, beta_increase_steps=20000, device=device)

agent01 = DDDQN(nS, nA, hidden_layers=(1000, 2000, 2000, 2000, 1000), gamma=0.99, learning_rate_start=0.0005, learning_rate_decay_steps=50000, learning_rate_min=0.0003,
              weight_decay=0.1, epsilon_start=1.0, epsilon_decay_steps=20000, epsilon_min=0.1, temp_start=10, temp_decay_steps=20000, temp_min=0.1, buffer_size_min=200,
              buffer_size_max=50000, batch_size=50, replays=1, tau=0.01, alpha=0.6, beta=0.1, beta_increase_steps=20000, device=device)

agent001 = DDDQN(nS, nA, hidden_layers=(1000, 2000, 2000, 2000, 1000), gamma=0.99, learning_rate_start=0.0005, learning_rate_decay_steps=50000, learning_rate_min=0.0003,
              weight_decay=0.01, epsilon_start=1.0, epsilon_decay_steps=20000, epsilon_min=0.1, temp_start=10, temp_decay_steps=20000, temp_min=0.1, buffer_size_min=200,
              buffer_size_max=50000, batch_size=50, replays=1, tau=0.01, alpha=0.6, beta=0.1, beta_increase_steps=20000, device=device)

agent = agent0
episode_rewards = []
steps = []
greedy_rates = []

def save_stats(curr_run, curr_episode):
    global agent
    global episodes
    global episode_rewards
    global steps

    agent.save_net(f'lunar_lander_r{curr_run}_e{curr_episode}.net')

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

    plt.savefig(f'train_stats_r{curr_run}.png')

    plt.close(figure)

    print(f'Episode {curr_episode} saved')

for run in range(training_runs):
    print(f'Run {run+1} started')

    # First 2 runs with weight_decay of 0
    if run == 2:
        agent = agent01  # 3. and 4. run with weight_decay of 0.1
    elif run == 4:
        agent = agent001  # Final two runs with weight_decay of 0.01

    agent.reset()
    episode_rewards = []
    steps = []
    greedy_rates = []

    for episode in tqdm(range(episodes)):
        obsv, done = env.reset(), False

        episode_reward = 0.0
        step_count = 0
        greedy_count = 0
        while not done:
            #env.render()

            action, is_greedy = agent.act_e_greedy(obsv)
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

        if (episode+1) % backup_episodes == 0:
            save_stats(run+1, episode+1)
    
    save_stats(run+1, episodes)

env.close()