"""
Tests different Metaparameters.
"""

from ddqn import DDQN
import torch
import numpy as np
import matplotlib.pyplot as plt
import gym
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f'Using device {device}')

episodes = 250
repetitions = 3
eval_every_episodes = 50
eval_episodes = 10

env = gym.make('LunarLander-v2')

nS = env.observation_space.shape[0]
nA = env.action_space.n


agent= DDQN(nS, nA, hidden_layers=(500, 500, 500), gamma=0.99, learning_rate_start=0.0005,
            learning_rate_decay_steps=20000, learning_rate_min=0.0003, epsilon_start=1.0, epsilon_decay_steps=20000,
            epsilon_min=0.1, buffer_size_min=200, buffer_size_max=50000, batch_size=50, replays=1, tau=0.01, device=device)


agents = [agent]


for a, agent in enumerate(agents):
    print(f'Agent: {a+1}')
    for r in range(repetitions):
        print(f'Repetition: {r+1}')

        agent.reset()
        
        episode_rewards = np.zeros(episodes, dtype=np.float32)
        step_count = 0
        steps = np.zeros(episodes, dtype=np.int32)

        eval_episode_rewards = np.zeros(episodes // eval_every_episodes, dtype=np.float32)

        for episode in range(episodes):
            obsv, done = env.reset(), False

            episode_reward = 0.0
            while not done:
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

            # evaluation
            if (episode+1) % eval_every_episodes == 0:
                avg_eval_reward = 0.0
                for _ in range(eval_episodes):
                    obsv, done = env.reset(), False
                    eval_episode_reward = 0.0

                    while not done:
                        action = agent.act_greedily(obsv)
                        new_obsv, reward, done, _ = env.step(action)
                        obsv = new_obsv

                        eval_episode_reward += reward
                    
                    avg_eval_reward += eval_episode_reward
                avg_eval_reward /= eval_episodes
                eval_episode_rewards[(episode+1) // eval_every_episodes - 1] = avg_eval_reward


        # Save data
        agent.save_net(f'lunar_lander_a{a+1}_r{r+1}.net')

        plt.clf()  # To prevent overlapping of old plots

        plt.subplot(211)
        plt.plot(np.arange(episodes), episode_rewards, 'k-')
        plt.grid(True)
        plt.title(f'Training Stats Agent {a+1} Repetition {r+1}')
        plt.ylabel('Reward')

        plt.subplot(212)
        plt.plot(np.arange(episodes), steps, 'k-')
        plt.grid(True)
        plt.xlabel('Episodes')
        plt.ylabel('Steps')

        plt.savefig(f'train_rewards_a{a+1}_r{r+1}.png')

        plt.clf()

        # Now the evaluation data
        plt.plot((np.arange(episodes // eval_every_episodes)+1) * eval_every_episodes, eval_episode_rewards, 'k-')
        plt.grid(True)
        plt.title(f'Evaluation Stats Agent {a+1} Repetition {r+1}')
        plt.xlabel('Episodes')
        plt.ylabel('Reward')

        plt.savefig(f'eval_rewards_a{a+1}_r{r+1}.png')


env.close()