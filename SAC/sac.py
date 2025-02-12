import torch
import torch.nn as nn
import copy
import numpy as np



import torch.optim as optim

from torch.distributions import Normal
from torch.optim.lr_scheduler import MultiStepLR



LOG_STD_MAX = 2
LOG_STD_MIN = -5
# policy network
class Actor(nn.Module):
    def __init__(self,opt):

        super(Actor,self).__init__()

        self.state_dim = opt.state_dim
        self.action_dim = opt.action_dim
        self.log_std_min = LOG_STD_MIN
        self.log_std_max = LOG_STD_MAX

        self.action_scale = torch.tensor(opt.action_scale,dtype=torch.float32).to('cuda:0')
        self.action_bias = torch.tensor(opt.action_bias,dtype=torch.float32).to('cuda:0')

        self.fc1 = nn.Linear(self.state_dim,1024)
        self.fc2 = nn.Linear(1024, 512)
        self.relu = nn.ReLU()
        self.mu = nn.Linear(512,self.action_dim)
        self.log_std = nn.Linear(512,self.action_dim)

    def forward(self,state):

        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))

        mu = self.mu(x)
        log_std = self.log_std(x)

        log_std = torch.tanh(log_std)  # -1~1 사이로 제한
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1) # log min ~ log max 사이의 값으로 만든다

        return mu,log_std

    def get_action(self,state):
        """

        :param state:
        :return: the action based on a squashed gaussian policy
        """

        mean,log_std = self.forward(state)

        std = log_std.exp() # log 통해 std 계산하기 때문이다

        normal = Normal(mean,std) # mu(s), std(s) 따르는 정규 분포 만든다 for reparameterization trick

        x_t = normal.rsample() # reparameterization trick => (mean + std * N(0,1)) ==> Action
        y_t = torch.tanh(x_t)

        # scaling & bias
        action = y_t * self.action_scale + self.action_bias

        log_prob = normal.log_prob(x_t) # sampling한 action의 확률값을 로그로 계산한다
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6) # tanh로 변형된값을 역함수를 통해 복원해야 한다 1-tanh^2을 하는 과정
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias

        return action,log_prob,mean

class Critic(nn.Module):
    def __init__(self,opt):
        super(Critic,self).__init__()

        self.state_dim = opt.state_dim
        self.action_dim = opt.action_dim

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



class SAC(object):
    def __init__(self,opt,device):
        self.state_dim = opt.state_dim
        self.action_dim = opt.action_dim
        self.device = device


        self.writer = opt.write

        self.tau = opt.tau
        self.gamma = opt.gamma
        self.actor_lr = opt.lr_init
        self.critic_lr = opt.lr_init
        self.batch_size = opt.batch_size
        self.total_it = 0
        self.policy_freq = opt.policy_freq
        self.grad_clip_norm = 5.0
        self.start_alpha=4e-3

        # automatic entropy coefficient alpha -> but high variance
        self.target_entropy = -float(self.action_dim) #-torch.Tensor(self.action_dim).to(self.device)

        self.log_alpha = torch.zeros(1, requires_grad=True,device=self.device)
        self.autotune_alpha = self.log_alpha.exp().item()
        self.autotune_alpha_optimizer = optim.AdamW(params=[self.log_alpha], lr=self.critic_lr)

        # network
        self.actor = Actor(opt).to(self.device)

        self.actor_optimizer = optim.AdamW(self.actor.parameters(), lr=self.actor_lr)

        self.critic = Critic(opt).to(self.device)
        self.critic_target = copy.deepcopy(self.critic).eval()
        self.critic_optimizer = optim.AdamW(self.critic.parameters(), lr=self.critic_lr)

        self.critic2 = Critic(opt).to(self.device)
        self.critic_target2 = copy.deepcopy(self.critic2).eval()
        self.critic_optimizer2 = optim.AdamW(self.critic2.parameters(), lr=self.critic_lr)

        self.sched_actor = MultiStepLR(self.actor_optimizer, milestones=[400], gamma=0.5)
        self.sched_critic = MultiStepLR(self.critic_optimizer, milestones=[400], gamma=0.5)
        self.sched_critic2 = MultiStepLR(self.critic_optimizer2, milestones=[400], gamma=0.5)

        self.sched_alpha = MultiStepLR(self.autotune_alpha_optimizer, milestones=[400], gamma=0.5)

        self.scheds = [self.sched_actor, self.sched_critic, self.sched_critic2, self.sched_alpha]





    def select_action(self,state,env,random_sample=False):
        state = torch.Tensor(state).to(self.device)
        state = state.unsqueeze_(0)

        if random_sample:
            # under learning start step
            # 무작위 action 선택 할것이다

            action = np.array(env.action_space.sample())

        else:
            action, _, _ = self.actor.get_action(state)
            action = action.detach().cpu().numpy()


        action = action.flatten()


        return action

    def has_enough_experience(self,buffer)->bool:
        """True if buffer hase enough experience to train """

        return len(buffer) >= buffer.batch_size

    def train(self,beta,alpha,PER_buffer):
        """
            with PER Buffer + per alpha apply adavtive schedule -> my experiment
            : td mean, td std 비율에 따라 alpha 동적으로 변화한다  -> utills에 schedule추가
            Critic_loss = MSE(Q, Q_target)
            Actor_loss = α * log_pi(a|s) - Q(s,a)
            where:
                actor_target(state) -> action
                critic_target(state, action) -> Q-value

        :param PER_buffer:
        :return:
        """

        idxs, experiences, sampling_weights = PER_buffer.sample(beta,alpha)

        states = np.array([e.state for e in experiences])
        actions = np.array([e.action for e in experiences])
        rewards = np.array([e.reward for e in experiences])
        next_states = np.array([e.next_state for e in experiences])
        dones = np.array([e.done for e in experiences])

        states = torch.Tensor(states).to(self.device)
        actions = torch.Tensor(actions).to(self.device)
        rewards = torch.Tensor(rewards).to(self.device)
        next_states = torch.Tensor(next_states).to(self.device)
        dones = torch.Tensor(dones).to(self.device)


        with torch.no_grad():

            #current policy에서 next action가져 온다
            next_action,next_log_pi,_ = self.actor.get_action(next_states)

            Q_target1 = self.critic_target(next_states,next_action)
            Q_target2 = self.critic_target2(next_states,next_action)

            # maxmize object function
            target_Q = torch.min(Q_target1,Q_target2) - self.autotune_alpha * next_log_pi

            # Q_targets = r + γ * (min_critic_target(next_state, actor_target(next_state)) - α *log_pi(next_action|next_state))
            target_Q = rewards.flatten()+((1-dones.flatten())*self.gamma * target_Q.view(-1))

        cur_Q1 = self.critic(states,actions).view(-1)
        cur_Q2 = self.critic2(states,actions).view(-1)

        TD_error1 = target_Q - cur_Q1
        TD_error2 = target_Q - cur_Q2
        # priority 선택시 overestimate 최소 위해 최소값을 선택하였는데 평균값으로 실험해봐도 괜찮을것 같다

        #TD_Error = torch.min(TD_error1, TD_error2)
        priority = abs(((TD_error1 + TD_error2)/2.0 + 1e-5).squeeze()).detach().cpu().numpy().flatten()
        PER_buffer.update_priorities(idxs, priority + 1e-6)  # priority must positiv

        _sampling_weights = (torch.Tensor(sampling_weights).view((-1, 1))).to(self.device)

        # MSE loss with importance sampling
        critic_loss = 0.5 * (TD_error1.pow(2) * _sampling_weights).mean()
        critic_loss2 = 0.5 * (TD_error2.pow(2) * _sampling_weights).mean()

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip_norm)
        self.critic_optimizer.step()

        self.critic_optimizer2.zero_grad()
        critic_loss2.backward()
        torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), self.grad_clip_norm)
        self.critic_optimizer2.step()


        if (self.total_it + 1)% self.policy_freq == 0:
            for _ in range(self.policy_freq):
                action_pi,log_pi,_ = self.actor.get_action(states)
                q1 = self.critic(states,action_pi)
                q2 = self.critic2(states,action_pi)
                q_pi = torch.min(q1,q2)

                actor_loss = (((self.autotune_alpha * log_pi) - q_pi)*_sampling_weights).mean()

                # Optimize the actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()


                autotune_alpha_loss = (-self.log_alpha.exp() * (log_pi.detach() + self.target_entropy)).mean()

                self.autotune_alpha_optimizer.zero_grad()
                autotune_alpha_loss.backward()
                #torch.nn.utils.clip_grad_norm_(autotune_alpha_loss, self.grad_clip_norm)

                self.autotune_alpha_optimizer.step()
                self.autotune_alpha = self.log_alpha.exp().item()

            self.soft_update(self.critic, self.critic_target, self.tau)
            self.soft_update(self.critic2, self.critic_target2, self.tau)

            self.total_it += 1




    def soft_update(self, network, target_network, rate):
        for network_params, target_network_params in zip(network.parameters(), target_network.parameters()):
            target_network_params.data.copy_(target_network_params.data * (1.0 - rate) + network_params.data * rate)

    def save(self,filename):

        # save model
        torch.save(self.actor.state_dict(), f"{filename}+_actor")
        torch.save(self.critic.state_dict(), f"{filename}+_critic_1")
        torch.save(self.critic2.state_dict(), f"{filename}+_critic_2")


    def load(self, filename):
        # load model
        self.actor.load_state_dict(torch.load(f"{filename}+_actor"))
        self.actor_target = copy.deepcopy(self.actor)

        self.critic.load_state_dict(torch.load(f"{filename}+_critic_1"))
        self.critic_target = copy.deepcopy(self.critic)

        self.critic2.load_state_dict(torch.load(f"{filename}+_critic_2"))
        self.critic_target2 = copy.deepcopy(self.critic2)
