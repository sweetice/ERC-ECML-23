import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import DiagGaussianActor, weight_init

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class TwinCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=None, hidden_depth=None):
        super(TwinCritic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)
        self.apply(weight_init)

    def forward(self, state, action, feature=False):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        f1 = F.relu(self.l2(q1))
        q1 = self.l3(f1)

        q2 = F.relu(self.l4(sa))
        f2 = F.relu(self.l5(q2))
        q2 = self.l6(f2)
        if feature:
            return q1, f1, q2, f2
        else:
            return q1, q2


    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1

    def feature(self, state, action):
        sa = torch.cat([state, action], 1)
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))

        return q1


class ERC():
    def __init__(self,
                 state_dim,
                 action_dim,
                 max_action,
                 discount=0.99,
                 tau=0.005,
                 policy_freq=2,
                 batch_size=256,
                 init_temperature = 0.1,
                 alpha_lr = 1e-4,
                 learn_alpha = True,
                 max_size=int(1e6),
                 # ERC parameters
                 beta=0.005
                 ):
        super(ERC, self).__init__()

        # init networks
        self.actor = DiagGaussianActor(state_dim, action_dim, hidden_dim=256, hidden_depth=2,
                                       log_std_bounds=[-10,2]).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic = TwinCritic(state_dim, action_dim, hidden_dim=256, hidden_depth=2).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        # init entropy
        self.learn_alpha = learn_alpha
        self.log_alpha = torch.FloatTensor([np.log(init_temperature)]).to(device).requires_grad_(True)
        self.target_entropy = - action_dim
        if self.learn_alpha:
            self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)

        # init parameters
        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_freq = policy_freq

        self.total_it = 0
        self.batch_size = batch_size
        self.ones = torch.ones(batch_size, 1).to(device)
        self.beta = beta
        self.algo = "ERC"

        # internal buffer
        self.buffer_max_size = max_size
        self.buffer_ptr = 0
        self.buffer_size = 0
        self.buffer_state = np.zeros((max_size, state_dim))
        self.buffer_action = np.zeros((max_size, action_dim))
        self.buffer_next_state = np.zeros((max_size, state_dim))
        self.buffer_reward = np.zeros((max_size, 1))
        self.buffer_not_done = np.zeros((max_size, 1))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def buffer_add(self, state, action, next_state, reward, done_bool):
        self.buffer_state[self.buffer_ptr] = state
        self.buffer_action[self.buffer_ptr] = action
        self.buffer_next_state[self.buffer_ptr] = next_state
        self.buffer_reward[self.buffer_ptr] = reward
        self.buffer_not_done[self.buffer_ptr] = 1. - done_bool
        self.buffer_ptr = (self.buffer_ptr + 1) % self.buffer_max_size
        self.buffer_size = min(self.buffer_size + 1, self.buffer_max_size)

    def buffer_sample(self, batch_size):
        ind = np.random.randint(0, self.buffer_size, size=batch_size)
        return (
            torch.FloatTensor(self.buffer_state[ind]).to(self.device),
            torch.FloatTensor(self.buffer_action[ind]).to(self.device),
            torch.FloatTensor(self.buffer_next_state[ind]).to(self.device),
            torch.FloatTensor(self.buffer_reward[ind]).to(self.device),
            torch.FloatTensor(self.buffer_not_done[ind]).to(self.device)
        )

    def select_action(self, state, sample=False):
        with torch.no_grad():
            state = torch.from_numpy(state).reshape(1, -1).to(device)
            dist = self.actor(state)
            action = dist.sample() if sample else dist.mean
            action = action.clamp(-self.max_action, self.max_action)
            return action.cpu().numpy().flatten()

    def get_value(self, state, action):
        with torch.no_grad():
            state = torch.from_numpy(state).reshape(1, -1).to(device)
            action = torch.from_numpy(action).reshape(1, -1).to(device)
            Q1, Q2 = self.critic(state, action)
            Q = 0.5 * (Q1 + Q2)
            return Q.cpu().numpy().flatten().item()

    def train(self, batch_size):
        self.total_it += 1
        state, action, next_state, reward, not_done = self.buffer_sample(batch_size)

        # Compute critic loss
        with torch.no_grad():
            next_action_dist = self.actor(next_state)
            next_action = next_action_dist.rsample()
            next_action_log_prob = next_action_dist.log_prob(next_action).sum(-1, keepdim=True)
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * next_action_log_prob
            target_Q = reward + not_done * self.discount * target_V

        current_Q1, current_Q2 = self.critic(state, action)
        critic_loss1, critic_loss2 = F.mse_loss(current_Q1, target_Q, reduction='none'), F.mse_loss(current_Q2, target_Q, reduction='none')

        with torch.no_grad():
            approximation_error_1 = torch.mean(critic_loss1)
            approximation_error_2 = torch.mean(critic_loss2)

        erc_loss1, erc_loss2 = F.mse_loss(critic_loss1, self.ones * approximation_error_1, reduction='none'), F.mse_loss(critic_loss2, self.ones * approximation_error_2, reduction='none')
        erc_loss1 = (torch.clip_(self.beta * erc_loss1, min=0, max=1e-2)).mean()
        erc_loss2 = (torch.clip_(self.beta * erc_loss2, min=0, max=1e-2)).mean()

        critic_loss1, critic_loss2 = critic_loss1.mean(), critic_loss2.mean()
        critic_loss_without_er = critic_loss1 + critic_loss2
        critic_loss = critic_loss_without_er + erc_loss1 + erc_loss2

        # optimize critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # optimize actor and alpha
        if self.total_it % self.policy_freq == 0:
            current_action_dist = self.actor(state)
            current_action = current_action_dist.rsample()
            current_action_log_prob = current_action_dist.log_prob(current_action).sum(-1, keepdim=True)
            current_action_Q1, current_action_Q2 = self.critic(state, current_action)
            current_action_Q = torch.min(current_action_Q1, current_action_Q2)
            actor_loss = (self.alpha.detach() * current_action_log_prob - current_action_Q).mean()

            if self.learn_alpha:
                self.log_alpha_optimizer.zero_grad()
                alpha_loss = (self.alpha * (-current_action_log_prob - self.target_entropy).detach()).mean()
                alpha_loss.backward()
                self.log_alpha_optimizer.step()
            else:
                alpha_loss = 0

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    @property
    def alpha(self):
        return self.log_alpha.exp()
