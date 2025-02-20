import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from priority_replay_buffer import *
from torch.optim.lr_scheduler import MultiStepLR
"""


Action Space

Box(-1.0, 1.0, (6,), float32)

Observation Space

Box(-inf, inf, (17,), float64)

import

gymnasium.make("HalfCheetah-v4")



Action Space

Box(-1.0, 1.0, (6,), float32)

Observation Space

Box(-inf, inf, (17,), float64)

import

gymnasium.make("Walker2d-v4")



"""


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action

        self.fc1 = nn.Linear(self.state_dim, 1024)
        # self.bn1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 512)
        # self.bn2 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, self.action_dim)
        self.relu = nn.ReLU()

    def forward(self, state):
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = x * self.max_action  # scale action

        return x


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.relu = nn.ReLU()

        self.fc1 = nn.Linear(state_dim + action_dim, 1024)
        # self.bn1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 512)
        # self.bn2 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 1)

    def forward(self, state, action):
        x = self.relu(self.fc1(torch.cat([state, action], 1)))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)

        return x


class DDPG(object):

    def __init__(self, opt, device):
        self.device = device
        self.state_dim = opt.state_dim
        self.action_dim = opt.action_dim
        self.max_action = opt.max_action
        self.discount_factor = opt.gamma
        self.tau = opt.tau
        self.alpha = opt.alpha
        self.expl_noise = opt.expl_noise
        self.batch_size = opt.batch_size
        self.writer = opt.write
        self.actor_lr = opt.lr_init
        self.critic_lr = opt.lr_init
        self.summary_writer = opt.summary_writer
        self.grad_clip_norm = 5.0

        # Define actor target, behavior net
        self.actor = Actor(self.state_dim, self.action_dim, self.max_action).to(self.device)
        self.actor_target = copy.deepcopy(self.actor).eval()
        self.actor_optimizer = optim.AdamW(self.actor.parameters(), lr=self.actor_lr)

        # Define critic target, behavior net

        self.critic = Critic(self.state_dim, self.action_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic).eval()
        self.critic_optimizer = optim.AdamW(self.critic.parameters(), lr=self.critic_lr)


        self.sched_actor = MultiStepLR(self.actor_optimizer, milestones=[400], gamma=0.5)
        self.sched_critic = MultiStepLR(self.critic_optimizer, milestones=[400], gamma=0.5)



        self.scheds = [self.sched_actor, self.sched_critic]


    def select_action(self, state):
        state = torch.Tensor(state.reshape(1, -1)).to(self.device)
        action = (self.actor(state).cpu().data.numpy().flatten() + np.random.normal(0,self.max_action * self.expl_noise,size=self.action_dim)).clip(-self.max_action, self.max_action)  # clip : 하한 상한
        return action

    def has_enough_experience(self, buffer) -> bool:
        """True if buffer hase enough experience to train """

        return len(buffer) >= buffer.batch_size

    def train(self, beta, alpha,PER_buffer):
        idxs, experiences, sampling_weights = PER_buffer.sample(beta,alpha)

        states = np.array([e.state for e in experiences])
        actions = np.array([e.action for e in experiences])
        rewards = np.array([e.reward for e in experiences])
        next_states = np.array([e.next_state for e in experiences])
        dones = np.array([e.done for e in experiences])

        states = torch.Tensor(states).to(self.device)
        actions = torch.Tensor(actions).to(self.device)
        rewards = torch.Tensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.Tensor(next_states).to(self.device)
        dones = torch.Tensor(dones).unsqueeze(1).to(self.device)

        next_actions = self.actor_target(next_states)

        with torch.no_grad():
            target_Q = self.critic_target(next_states, next_actions)

            target_Q = rewards + ((1 - dones) * self.discount_factor * target_Q).detach()

        current_Q1 = self.critic(states, actions)


        TD_error = target_Q - current_Q1


        priority = abs(((TD_error)/2.0 + 1e-5).squeeze()).detach().cpu().numpy().flatten()
        PER_buffer.update_priorities(idxs, priority + 1e-6)  # priority must positiv

        _sampling_weights = (torch.Tensor(sampling_weights).view((-1, 1))).to(self.device)

        # MSE loss with importance sampling
        critic_loss = 0.5 * (TD_error.pow(2) * _sampling_weights).mean()

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip_norm)
        self.critic_optimizer.step()



        # compute actor loss
        # 기본이 GD라서 -를 해야 GA가 되므로 -를 곱한다
        actor_loss = -torch.mean(_sampling_weights * self.critic(states, self.actor(states)))

        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.soft_update(self.critic, self.critic_target, self.tau)
        self.soft_update(self.actor, self.actor_target, self.tau)



    def soft_update(self, network, target_network, rate):
        for network_params, target_network_params in zip(network.parameters(), target_network.parameters()):
            target_network_params.data.copy_(target_network_params.data * (1.0 - rate) + network_params.data * rate)

    def save(self, filename):

        # save model
        torch.save(self.actor.state_dict(), f"{filename}+_actor")
        torch.save(self.critic.state_dict(), f"{filename}+_critic")

    def load(self, filename):
        # load model
        self.actor.load_state_dict(torch.load(f"{filename}+_actor"))
        self.actor_target = copy.deepcopy(self.actor)

        self.critic.load_state_dict(torch.load(f"{filename}+_critic"))
        self.critic_target = copy.deepcopy(self.critic)



