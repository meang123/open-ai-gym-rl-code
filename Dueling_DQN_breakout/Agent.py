import numpy as np
import torch
from torch import nn
import copy
import torch.optim as optim
from PER import *
from torch.optim.lr_scheduler import MultiStepLR

# dueling dqn network
class Dueling_DQN(nn.Module):

    def __init__(self,nb_action=4):
        super(Dueling_DQN,self).__init__()

        self.relu = nn.ReLU()

        # preprocess에서 gray scale입력이 들어올것이라서 1채널로 시작 한다
        self.conv1 = nn.Conv2d(1,32,kernel_size=(8,8),stride=(4,4))
        self.conv2 = nn.Conv2d(32,64,kernel_size=(4,4),stride=(2,2))
        self.conv3 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1))

        self.flatten = nn.Flatten()


        # (84,84)를 conv3까지 진행 한 결과
        self.advantage1 = nn.Linear(22528,1024) #22528
        #self.advantage2 = nn.Linear(1024, 1024)
        self.advantage3 = nn.Linear(1024, nb_action)

        self.state_value1 = nn.Linear(22528,1024) #3136
        #self.state_value2 = nn.Linear(1024, 1024)
        self.state_value3 = nn.Linear(1024, 1)

    def forward(self,x):


        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.flatten(x)
        state_value = self.relu(self.state_value1(x))

        #state_value = self.relu(self.state_value2(state_value))

        state_value = self.relu(self.state_value3(state_value))

        action_value = self.relu(self.advantage1(x))

        #action_value = self.relu(self.advantage2(action_value))

        action_value = self.relu(self.advantage3(action_value))

        output = state_value + (action_value - action_value.mean()) # Q = V + A

        return output

    # def save_the_model(self,weights_filename = './models/latest.pt'):
    #     torch.save(self.state_dict(), weights_filename)
    #
    #
    # def load_the_model(self, weights_filename='./models/latest.pt'):
    #     try:
    #         self.load_state_dict(torch.load(weights_filename))
    #         print(f"Successfully loaded weights file {weights_filename}")
    #     except:
    #         print(f"No weights file available at {weights_filename}")




# numpy로 처리하고 tensor화 하는 코드 방향으로 작성

class Agent(object):
    def __init__(self,model,opt,device):

        self.device = device

        self.model = model
        self.target_model = copy.deepcopy(model).eval()  # set require_grad = False

        self.model.to(self.device)
        self.target_model.to(self.device)

        self.gamma = opt.gamma
        self.tau = 0.05
        self.batch_size = opt.batch_size
        self.esip_init = opt.epsilon_init
        self.action_dim = opt.action_dim
        self.state_dim = opt.state_dim
        self.grad_clip_norm = 5.0

        self.optimizer = optim.Adam(model.parameters(),lr = opt.lr_init)





        self.sched_critic = MultiStepLR(self.optimizer, milestones=[400], gamma=0.5)

        self.scheds = [self.sched_critic]


    def select_action(self,state,epsilon,deterministic=False):

        """


        :param state:
        :param epsilon:
        :param deterministic:
        :return: actions

        Experience가 nametuple로 설정되어있기 때문에 state를 tensor로 바꿀때 tuple로 묶는다  그리고 차원을 한차원 늘린다

        """
        state_tensor = torch.from_numpy(state).unsqueeze(0).unsqueeze(0).to(self.device)
        state_tensor = state_tensor.float()

        if deterministic: # Greedy policy
            _, actions = self.model(state_tensor).max(dim=1, keepdim=True)
            action = (actions.cpu().item())
            return action


        else:
            # insufficient experince buffer상태면 완전 랜덤하게 탐색 위함
            if not self.has_enough_experience():
                action = np.random.randint(self.action_dim)
            #epsilon greedy policy behavior policy
            else:
                if np.random.random() < epsilon:
                    action = np.random.randint(self.action_dim)

                # greedy policy
                else:
                    #print("\nGreedy policy !!!!!!\n")
                    _, actions = self.model(state_tensor).max(dim=1, keepdim=True)
                    action = (actions.cpu().item())


        return action




    def has_enough_experience(self)->bool:
        """True if buffer hase enough experience to train """

        return len(self.PER_buffer) >= self.PER_buffer.batch_size



    # loop train part in main file
    def train(self,beta,alpha,buffer):

        idxs,experiences,sampling_weights = buffer.sample(beta,alpha)
        # Experience nametuple을 각각 앞에서부터 가져오는 연산 위해 zip(*experience)사용

        #states, actions, rewards, next_states, dones = (torch.Tensor(vs).to(self.device) for vs in zip(*experiences))


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



        states = states.unsqueeze(1)
        next_states = next_states.unsqueeze(1)
        actions = actions.unsqueeze(dim=1).long()
        rewards = rewards.unsqueeze(dim=1)
        dones = dones.unsqueeze(dim=1)

        """
        TD Error 
        y = r+gamma *max target Q(next state,next action)
        
        y-cur Q(s,a)
        
        """


        _, argmax_action = self.model(next_states).max(dim=1, keepdim=True)
        next_q_val = self.target_model(next_states).gather(dim=1,index=argmax_action).to(self.device)
        target_Q = rewards+((1-dones) * self.gamma*next_q_val).to(self.device)
        current_Q = self.model(states).gather(dim=1, index=actions).to(self.device)  # state에 따른 action Q estimate value

        TD_Error = target_Q-current_Q

        priority = (TD_Error.abs().cpu().detach().numpy().flatten())
        buffer.update_priorities(idxs,priority + 1e-6) # priority must positive value

        # compute MSE loss

        _sampling_weights = (torch.Tensor(sampling_weights).view((-1, 1))).to(self.device)

        loss = 0.5 * (TD_Error.pow(2) * _sampling_weights).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip_norm)
        self.optimizer.step()


        # update target network model -> soft update

        for param,target_param in zip(self.model.parameters(),self.target_model.parameters()):
            target_param.data.copy_(self.tau*param.data+(1-self.tau)*target_param.data)







    def save(self, algo, EnvName, steps):
        torch.save(self.model.state_dict(), "./model/cur_{}_{}_{}.pth".format(algo, EnvName, steps))

        torch.save(self.target_model.state_dict(), "./model/target_{}_{}_{}.pth".format(algo, EnvName, steps))

    def load(self, algo, EnvName, steps):
        self.model.load_state_dict(torch.load("./model/cur_{}_{}_{}.pth".format(algo,EnvName,steps)))
        self.target_model.load_state_dict(torch.load("./model/target_{}_{}_{}.pth".format(algo,EnvName,steps)))
