from dddqn import DDDQN
import torch
import numpy as np
import matplotlib.pyplot as plt
import gym
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

episodes = 100

env = gym.make('LunarLander-v2')

nS = env.observation_space.shape[0]
nA = env.action_space.n

agent = DDDQN(nS, nA, hidden_layers=(1000, 2000, 2000, 2000, 1000), device=device)
agent.load_net('experiments_20220112/lunar_lander_r5_e1000.net')

episode_rewards = np.zeros(episodes, dtype=np.float32)

for episode in tqdm(range(episodes)):
    obsv, done = env.reset(), False

    episode_reward = 0.0
    while not done:
        env.render()

        action, _ = agent.act_greedily(obsv)
        new_obsv, reward, done, info = env.step(action)
        time_out = 'TimeLimit.truncated' in info and info['TimeLimit.truncated']
        terminal = done and not time_out

        obsv = new_obsv
        
        episode_reward += reward  # For statistics
    
    episode_rewards[episode] = episode_reward

env.close()

plt.plot(np.arange(episodes), episode_rewards, 'k-')
plt.grid(True)
plt.title('Reward per Episode')
plt.xlabel('Episodes')
plt.ylabel('Reward')
plt.savefig('rewards_eval.png')