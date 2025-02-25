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
    def __init__(self,state_dim,action_dim,max_action):
        super(Actor,self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.fc1 = nn.Linear(self.state_dim, 1024)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, self.action_dim)
        self.max_action = max_action

    def forward(self, state):

        state = self.relu(self.fc1(state))
        state = self.relu(self.fc2(state))
        return self.max_action * torch.tanh(self.fc3(state))

class Critic(nn.Module):
    def __init__(self,state_dim,action_dim):
        super(Critic,self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.relu = nn.ReLU()
        # Q1 architecture
        self.l1 = nn.Linear(self.state_dim + self.action_dim, 1024)
        self.l2 = nn.Linear(1024, 512)
        self.l3 = nn.Linear(512, 1)




    def forward(self, state, action):


        x = torch.cat([state, action], 1)

        x = self.relu(self.l1(x))
        x = self.relu(self.l2(x))
        x = self.l3(x)

        return x



class TD_3(object):
    def __init__(self,opt,device):
        self.device = device
        self.state_dim = opt.state_dim
        self.action_dim = opt.action_dim
        self.max_action = opt.max_action
        self.discount_factor = opt.gamma
        self.tau = opt.tau
        self.expl_noise = opt.expl_noise
        self.summary_writer = opt.summary_writer

        # Action scale에 맞추기 위해 max_action 곱하여서 scale맞춤
        self.policy_noise = opt.policy_noise * self.max_action
        self.noise_clip = opt.noise_clip * self.max_action
        self.grad_clip_norm = 5.0

        self.policy_freq = opt.policy_freq
        self.batch_size = opt.batch_size

        self.writer = opt.write
        self.actor_lr = opt.lr_init
        self.critic_lr = opt.lr_init

        self.actor = Actor(self.state_dim,self.action_dim,self.max_action).to(self.device)
        self.actor_target = copy.deepcopy(self.actor).eval()
        self.actor_optimizer = optim.AdamW(self.actor.parameters(),lr=self.actor_lr)

        self.critic1 = Critic(self.state_dim,self.action_dim).to(self.device)
        self.critic_target1 = copy.deepcopy(self.critic1).eval()
        self.critic_optimizer1 = optim.AdamW(self.critic1.parameters(),lr=self.critic_lr)

        self.critic2 = Critic(self.state_dim,self.action_dim).to(self.device)
        self.critic_target2 = copy.deepcopy(self.critic2).eval()
        self.critic_optimizer2 = optim.AdamW(self.critic2.parameters(),lr=self.critic_lr)

        self.sched_actor = MultiStepLR(self.actor_optimizer, milestones=[400], gamma=0.5)
        self.sched_critic = MultiStepLR(self.critic_optimizer1, milestones=[400], gamma=0.5)
        self.sched_critic2 = MultiStepLR(self.critic_optimizer2, milestones=[400], gamma=0.5)



        self.scheds = [self.sched_actor, self.sched_critic, self.sched_critic2]



        self.total_it = 0

    def select_action(self,state):

        state = torch.Tensor(state.reshape(1, -1)).to(self.device)
        action = (self.actor(state).cpu().data.numpy().flatten() + np.random.normal(0,self.max_action * self.expl_noise,size=self.action_dim)).clip(-self.max_action, self.max_action)  # clip : 하한 상한
        return action

    def has_enough_experience(self,buffer)->bool:
        """True if buffer hase enough experience to train """

        return len(buffer) >= buffer.batch_size

    def train(self,beta,alpha,PER_buffer):

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

        # target smoothing : add noise
        noise = (torch.randn_like(actions) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
        next_actions = (self.actor_target(next_states) + noise).clamp(-self.max_action, self.max_action)

        target_Q1 = self.critic_target1(next_states, next_actions)
        target_Q2 = self.critic_target2(next_states, next_actions)

        with torch.no_grad():
            # cllip , select min estimated value
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = rewards + ((1-dones) * self.discount_factor * target_Q).detach()

        current_Q1 = self.critic1(states, actions)
        current_Q2 = self.critic2(states, actions)

        TD_error1 = target_Q - current_Q1
        TD_error2 = target_Q - current_Q2

        # priority 선택시 overestimate 최소 위해 최소값을 선택하였는데 평균값으로 실험해봐도 괜찮을것 같다

        #TD_Error = torch.min(TD_error1, TD_error2)
        priority = abs(((TD_error1 + TD_error2)/2.0 + 1e-5).squeeze()).detach().cpu().numpy().flatten()
        PER_buffer.update_priorities(idxs, priority + 1e-6)  # priority must positiv

        _sampling_weights = (torch.Tensor(sampling_weights).view((-1, 1))).to(self.device)

        # MSE loss with importance sampling
        critic_loss = 0.5 * (TD_error1.pow(2) * _sampling_weights).mean()
        critic_loss2 = 0.5 * (TD_error2.pow(2) * _sampling_weights).mean()

        # Optimize the critic
        self.critic_optimizer1.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), self.grad_clip_norm)
        self.critic_optimizer1.step()

        self.critic_optimizer2.zero_grad()
        critic_loss2.backward()
        torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), self.grad_clip_norm)
        self.critic_optimizer2.step()

        self.total_it += 1
        if (self.total_it + 1) % self.policy_freq == 0:
            actor_loss = -torch.mean(_sampling_weights*self.critic1(states, self.actor(states)))

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self.soft_update(self.critic1, self.critic_target1, self.tau)
            self.soft_update(self.critic2, self.critic_target2, self.tau)
            self.soft_update(self.actor, self.actor_target, self.tau)






    def soft_update(self, network, target_network, rate):
        for network_params, target_network_params in zip(network.parameters(), target_network.parameters()):
            target_network_params.data.copy_(target_network_params.data * (1.0 - rate) + network_params.data * rate)

    def save(self,filename):

        # save model
        torch.save(self.actor.state_dict(), f"{filename}+_actor")
        torch.save(self.critic1.state_dict(), f"{filename}+_critic_1")
        torch.save(self.critic2.state_dict(), f"{filename}+_critic_2")


    def load(self, filename):
        # load model
        self.actor.load_state_dict(torch.load(f"{filename}+_actor"))
        self.actor_target = copy.deepcopy(self.actor)

        self.critic1.load_state_dict(torch.load(f"{filename}+_critic_1"))
        self.critic_target = copy.deepcopy(self.critic1)

        self.critic2.load_state_dict(torch.load(f"{filename}+_critic_2"))
        self.critic_target2 = copy.deepcopy(self.critic2)
