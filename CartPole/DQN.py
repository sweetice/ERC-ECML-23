import torch
import torch as th
import torch.nn as nn
import copy
import numpy as np

class DQNNetwork(nn.Module):
    def __init__(self, obs_dim, hidden_dim, action_dim):
        super(DQNNetwork, self).__init__()
        self.repr_layer = nn.Sequential(nn.Linear(obs_dim, hidden_dim),
                                        nn.ReLU(),
                                        nn.Linear(hidden_dim, hidden_dim),
                                        nn.ReLU())

        self.fc = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        repr = self.repr_layer(x)
        Q_list = self.fc(repr)
        return Q_list

    def max_Q_value(self, obs):
        with torch.no_grad():
            Q_list = self.forward(obs)
            max_value, idx = th.max(Q_list,  dim=1, keepdim=True)
            return max_value


class DQN():
    def __init__(self, env, obs_dim, hidden_dim, action_dim, epsilon=0.1, gamma=0.95, tau=0.005, seed=0):
        super(DQN, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.epsilon = epsilon
        self.gamma = gamma
        self.num_action = action_dim
        self.tau = tau
        self.env = env
        self.dqn_net = DQNNetwork(obs_dim=obs_dim, hidden_dim=hidden_dim, action_dim=action_dim).to(self.device)
        self.dqn_target = copy.deepcopy(self.dqn_net)
        self.optimizer = torch.optim.Adam(self.dqn_net.parameters(), lr=1e-4)
        self.target_update_frequency = 100
        self.training_count = 0
        self.writer = False
        self.log = True
        self.algo = "DQN"
        self.seed = seed

        self.log_frequency = 10
        self.epoch = 0
        self.epoch_list = []

    def select_action(self, obs, deterministic=False):
        with torch.no_grad():
            if np.random.random() < self.epsilon and deterministic==False:
                action = np.random.choice(self.num_action)
                return action
            else:
                obs = np.array(obs, dtype=np.float32)
                obs = torch.from_numpy(obs).reshape(1,-1).to(self.device)
                Q_list = self.dqn_net(obs)
                max_value, idx = th.max(Q_list, dim=1, keepdim=True)
                return idx.cpu().numpy().flatten().item()

    def train_DQN(self, replay_buffer, batch_size):
        self.training_count += 1
        obs, action, next_obs, reward, not_done = replay_buffer.sample(batch_size)
        with torch.no_grad():

            target_Q = self.dqn_target.max_Q_value(next_obs)
            target_Q= reward + self.gamma * not_done * target_Q

        current_Q = self.dqn_net(obs)
        idx = torch.tensor(action, dtype=torch.int64)
        current_Q = current_Q.gather(1, idx)
        loss = torch.nn.functional.mse_loss(current_Q, target_Q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network
        for param, target_param in zip(self.dqn_net.parameters(), self.dqn_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e5)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim), dtype='float32')
        self.action = np.zeros((max_size, action_dim), dtype='float32')
        self.next_state = np.zeros((max_size, state_dim), dtype='float32')
        self.reward = np.zeros((max_size, 1), dtype='float32')
        self.not_done = np.zeros((max_size, 1), dtype='float32')

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.from_numpy(self.state[ind]).to(self.device),
            torch.from_numpy(self.action[ind]).to(self.device),
            torch.from_numpy(self.next_state[ind]).to(self.device),
            torch.from_numpy(self.reward[ind]).to(self.device),
            torch.from_numpy(self.not_done[ind]).to(self.device)
        )
