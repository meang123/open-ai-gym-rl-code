import torch

import copy
import torch.optim as optim


class Agent(object):
    def __init__(self,model,opt,device):

        self.device = device

        self.model = model
        self.target_model = copy.deepcopy(model).eval()  # set require_grad = False

        self.model.to(self.device)
        self.target_model.to(self.device)

        self.gamma = opt.gamma
        self.tau = 0.005
        self.batch_size = opt.batch_size
        self.esip_init = opt.epsilon_init
        self.action_dim = opt.action_dim
        self.state_dim = opt.state_dim


        self.optimizer = optim.AdamW(model.parameters(),lr = opt.lr_init)

    def select_action(self,state,deterministic):

        with torch.inference_mode():

            # optimal policy for best action
            # episode를 끝낼때 action계산하기 위해 greedy따르는 정책으로 진행

            Q = self.model(state).detach() # copy
            if deterministic:

                return torch.argmax(Q,dim =-1,keepdim=True).item()
            else:
                # 입실론 greedy 정책으로 action선택
                if torch.rand(1)<self.esip_init:
                    action = torch.randint(self.action_dim,(1,1)).to(self.device) # action에서 취할수있는 행동을 (1,1)사이즈로 랜덤 선택
                    q_a = Q.gather(1,action)
                    return action.item(),q_a

                else:
                    # greedy 따르는 정책을 따라간다

                    action = torch.argmax(Q, dim=-1, keepdim=True).to(self.device)
                    q_a = Q.gather(1,action)
                    return action.item(),q_a


    # loop train part in main file
    def train(self,replay_buffer):

        state,action,reward,next_state,terminated,truncated,index,Normed_IS_weight = replay_buffer.sample(self.batch_size)

        # shape
        # s,a,r,s_next,terminate,truncate : (batch size,dim)
        # index : (batch size,)

        #수식 이용해서 계산 하는 과정 double dqn 적용 한다 Q2 argmax를 Q1 인자에 넣기 때문에 gather를 통해서 indexing할수가 있다
        # Q(s,a) State에 대한 action을 gather을 이용해서 state에 대한 action 값 추출한다

        with torch.inference_mode():
            #next_state = self.preprocess_torch_version(next_state)
            # index로 gathering할수있도록 차원 늘리기
            argmax_action = self.model(next_state).argmax(dim=1).unsqueeze(-1).to(self.device)

            max_q_prime = self.target_model(next_state).gather(1,argmax_action).to(self.device)


            # compute Q target and avoid teminated된 경우 다음 상태를 처리 하지 않도록 해야 함

            Q_target = reward.to(self.device) + (~terminated.to(self.device))*self.gamma*max_q_prime


        # get current Q estimate Q(s,a)
        #state = self.preprocess_torch_version(state)
        current_Q = self.model(state.to(self.device)).gather(1,action.to(self.device)) # state에 따른 action Q estimate value

        # MSE loss with importance sampling
        q_loss = torch.square(Normed_IS_weight.to(self.device)*(Q_target-current_Q)).mean()
        self.model.zero_grad()
        q_loss.backward()
        self.optimizer.step()

        # upgrade priorites of the current batch
        # Batch만큼 priorites 값을 업그레이드 해야한다
        with torch.inference_mode():
            # TD error
            batch_priorities = ((torch.abs(Q_target - current_Q).to(self.device) + 0.01)**replay_buffer.alpha).squeeze(-1)  # (batch size,)
            replay_buffer.TD_prior[index] = batch_priorities # priority에 따라 sampling한 값에 우선 순위(TD error)값 다시 계싼한것을 업데이트 한다


        # update target network model
        for param,target_param in zip(self.model.parameters(),self.target_model.parameters()):
            target_param.data.copy_(self.tau*param.data+(1-self.tau)*target_param.data)


    def save(self, algo, EnvName, steps):
        torch.save(self.model.state_dict(), "./model/{}_{}_{}.pth".format(algo, EnvName, steps))


    def load(self, algo, EnvName, steps):
        self.model.load_state_dict(torch.load("./model/{}_{}_{}.pth".format(algo,EnvName,steps)))
        self.target_model.load_state_dict(torch.load("./model/{}_{}_{}.pth".format(algo,EnvName,steps)))
