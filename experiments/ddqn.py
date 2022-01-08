import torch
from torch import tensor
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

class Network(nn.Module):
    def __init__(self, layers, device=None):
        super(Network, self).__init__()

        self.device = device

        modules = nn.ModuleList()
        
        for i in range(len(layers) - 2):
            modules.append(nn.Linear(layers[i], layers[i+1]))
            modules.append(nn.LeakyReLU(0.1))
        
        modules.append(nn.Linear(layers[-2], layers[-1]))
        self.linear_stack = nn.Sequential(*modules)
    
    def forward(self, state):        
        Q = self.linear_stack(state)
        return Q

class ExperienceBuffer:
    def __init__(self, max_len, state_dim):
        self.states = np.empty((max_len, state_dim), dtype=np.float32)
        self.actions = np.empty(max_len, dtype=np.int16)
        self.rewards = np.empty(max_len, dtype=np.float32)
        self.next_states = np.empty((max_len, state_dim), dtype=np.float32)
        self.terminals = np.empty(max_len, dtype=np.int8)

        self.index = 0
        self.full = False
        self.max_len = max_len
        self.rng = np.random.default_rng()
    
    def store_experience(self, state, action, reward, next_state, terminal):
        self.states[self.index] = state
        self.actions[self.index] = action
        self.rewards[self.index] = reward
        self.next_states[self.index] = next_state
        self.terminals[self.index] = terminal
        
        self.index += 1
        self.index %= self.max_len  # Replace oldest Experiences if Buffer is full
        self.full = True if self.index == 0 else self.full

    def get_experiences(self, batch_size):
        newest_index = self.index - 1 if self.index > 0 else self.max_len - 1
        indices = self.rng.choice(self.__len__(), batch_size - 1)  # One less than batch_size because newest element is safe in
        indices = np.append(indices, newest_index)

        states = np.array([self.states[i] for i in indices], dtype=np.float32)
        actions = np.array([self.actions[i] for i in indices], dtype=np.int16)
        rewards = np.array([self.rewards[i] for i in indices], dtype=np.float32)
        next_states = np.array([self.next_states[i] for i in indices], dtype=np.float32)
        terminals = np.array([self.terminals[i] for i in indices], dtype=np.int8)

        return states, actions, rewards, next_states, terminals

    def __len__(self):
        return self.max_len if self.full else self.index

class DDQN:
    def __init__(self, state_dim, action_num, hidden_layers=(500, 500, 500), gamma=0.99, learning_rate_start=0.0005,
                 learning_rate_decay_steps=20000, learning_rate_min=0.0003, epsilon_start=1.0, epsilon_decay_steps=20000,
                 epsilon_min=0.1, buffer_size_min=200, buffer_size_max=50000, batch_size=50, replays=1, tau=0.01, device='cuda'):
        """
        Gamme is the Discount, Epsilon is the probability of choosing random action as opposed to greedy one.
        Buffer Size determins max number of stored experiences, Batch Size is training batch size in one training step.
        Target frozen steps is the timer after which the target network is updated.
        """
        self.state_dim = state_dim
        self.action_num = action_num
        self.hidden_layers = hidden_layers
        layers = (state_dim, *hidden_layers, action_num)  # 3 hidden layers of sizes 500, 500 and 500
        self.q_net = Network(layers, device).to(device)
        self.target_q_net = Network(layers, device).to(device)
        self._update_target(1.0)  # Fully copy Online Net weights to Target Net

        self.optimizer = torch.optim.RMSprop(self.q_net.parameters(), lr=learning_rate_start)  # Using RMSProp because it's more stable than Adam
        self.loss_function = nn.HuberLoss()

        self.buffer = ExperienceBuffer(buffer_size_max, state_dim)
        self.buffer_size_max = buffer_size_max
        self.buffer_size_min = buffer_size_min
        self.batch_size = batch_size
        self.replays = replays

        self.gamma = gamma

        # Linearly decay Learning Rate and Epsilon from start to min in a given amount of steps
        self.learning_rate = learning_rate_start
        self.learning_rate_start = learning_rate_start
        self.learning_rate_decay = (learning_rate_start - learning_rate_min) / learning_rate_decay_steps
        self.learning_rate_min = learning_rate_min

        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_decay = (epsilon_start - epsilon_min) / epsilon_decay_steps
        self.epsilon_min = epsilon_min

        self.tau = tau

        self.device = device

        self.rng = np.random.default_rng()
    
    def reset(self):
        """
        Reset object to its initial state if you want to do multiple training passes with it
        """
        layers = (self.state_dim, *(self.hidden_layers), self.action_num)  # 3 hidden layers of sizes 500, 500 and 500
        self.q_net = Network(layers, self.device).to(self.device)
        self.target_q_net = Network(layers, self.device).to(self.device)
        self._update_target(1.0)  # Fully copy Online Net weights to Target Net

        self.optimizer = torch.optim.RMSprop(self.q_net.parameters(), lr=self.learning_rate_start)

        self.buffer = ExperienceBuffer(self.buffer_size, self.state_dim)

        self.learning_rate = self.learning_rate_start
        self.epsilon = self.epsilon_start
    
    def act(self, state):
        """
        Decides on action based on current state using epsilon-greedy Policy.
        """
        if self.rng.random() < self.epsilon:
            action = self.rng.integers(self.action_num)  # Random
        else:
            with torch.no_grad():
                state = tensor(state, device=self.device, dtype=torch.float32)
                Q = self.q_net(state)  # Greedy
                action = Q.argmax().item()
        return action

    def act_greedily(self, state):
        """
        Decides on action based on current state using greedy Policy.
        """
        with torch.no_grad():
            state = tensor(state, device=self.device, dtype=torch.float32)
            Q = self.q_net(state)
            action = Q.argmax().item()
        return action
    
    def experience(self, state, action, reward, next_state, terminal):
        """
        Takes experience and stores it for replay.
        """
        self.buffer.store_experience(state, action, reward, next_state, terminal)
    
    def train(self):
        """
        Train Q-Network on a batch from the replay buffer.
        """
        if len(self.buffer) < self.buffer_size_min:
            return  # Dont train until Replay Buffer has collected a certain number of initial experiences

        for r in range(self.replays):
            batch_size = len(self.buffer) if len(self.buffer) < self.batch_size else self.batch_size
            states, actions, rewards, next_states, terminals = self.buffer.get_experiences(self.batch_size)

            states = tensor(states, device=self.device, dtype=torch.float32)
            rewards = tensor(rewards, device=self.device, dtype=torch.float32)
            actions = tensor(actions, device=self.device, dtype=torch.int64)
            next_states = tensor(next_states, device=self.device, dtype=torch.float32)
            terminals = tensor(terminals, device=self.device, dtype=torch.int8)

            with torch.no_grad():
                max_actions = self.q_net(next_states).argmax(1).unsqueeze(1)
                max_action_vals = self.target_q_net(next_states).gather(1, max_actions).squeeze()

            targets = rewards + self.gamma * max_action_vals * (1 - terminals)
            predictions = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze()
            #loss = (targets - predictions).pow(2).mul(0.5).mean()
            loss = self.loss_function(predictions, targets)  # Huber Loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self._update_parameters()
        self._update_target(self.tau)
    
    def save_net(self, path):
        torch.save(self.q_net.state_dict(), path)
    
    def load_net(self, path):
        self.q_net.load_state_dict(torch.load(path))
        self._update_target(1.0)  # Also load weights into target net
    
    def _update_target(self, tau):
        """
        Update Target Network by blending Target und Online Network weights using the factor tau.
        A tau of 1 just copies the whole online network over to the target network
        """
        for param, target_param in zip(self.q_net.parameters(), self.target_q_net.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
    
    def _update_parameters(self):
        """
        Decays parameters like learning rate and epsilon one step
        """
        self.learning_rate -= self.learning_rate_decay
        self.learning_rate = max(self.learning_rate, self.learning_rate_min)

        self.optimizer.param_groups[0]['lr'] = self.learning_rate

        self.epsilon -= self.epsilon_decay
        self.epsilon = max(self.epsilon, self.epsilon_min)