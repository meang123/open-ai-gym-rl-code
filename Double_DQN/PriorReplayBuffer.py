"""
prior replay buffer with stochastic sampling and importance sampling


"""
import torch





class Prior_Replay_Buffer():
    def __init__(self,opt,algo_name,device ='cuda:0'): # opt is option argument in main file. device for gpu memory setting buffer content to cpu
        self.device = device
        self.ptr = 0 # buffer index
        self.size = 0 # check untill buffer size

        # buffer size만큼 state,action,reward,terminate,tuncated배열 만든다
        if algo_name=="DDPG":
            self.state = torch.zeros((opt.buffer_size, 1, opt.state_dim), device=self.device)

        elif algo_name=="Double_DQN":
            self.state = torch.zeros((opt.buffer_size,1,opt.state_dim[0]),device=self.device) # action space size is 1 (4)
        else:

            self.state = torch.zeros((opt.buffer_size,1,opt.state_dim[0],opt.state_dim[1]),device=self.device) # (batch,channel,state dim[0],state_dim[1]) 256,1,84,84

        self.action = torch.zeros((opt.buffer_size,1),dtype=torch.int64,device=self.device)
        self.reward =torch.zeros((opt.buffer_size,1),device=self.device)
        self.terminate = torch.zeros((opt.buffer_size,1),dtype=torch.bool,device=self.device)
        self.truncated = torch.zeros((opt.buffer_size,1),dtype=torch.bool,device=self.device)
        self.TD_prior = torch.zeros(opt.buffer_size,dtype=torch.float32,device=self.device)
        self.buffer_size = opt.buffer_size

        self.alpha = opt.alpha # for stochastic sampling
        self.beta = opt.beta_init # for importance sampling
        self.replacement = opt.replacement # 복원 추출 할지 안할지 (T/F)


    # buffer size에 넣는 함수
    def add(self,state,action,reward,terminate,truncated,TD_prior):
        self.state[self.ptr] = state #1 1 84 84
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.terminate[self.ptr]=terminate
        self.truncated[self.ptr]=truncated
        self.TD_prior[self.ptr]=TD_prior

        # buffer index에 추가 하는 역할
        self.ptr = (self.ptr+1)%self.buffer_size # buffer size 넘지 않도록 index관리 한다
                                                # every step 만다 ++1

        self.size = min(self.size+1,self.buffer_size) # size ++1 , 최대 buffer_size가진다


    # buffer size에서 stochastic sampling batch size만큼 sampling한다

    def sample(self,batch_size):

        sampling_prob = self.TD_prior[:self.size-1].clone() # clone TD prior buffer

        #ptr과 size 같이 가는데 size가 큰 경우라면 ptr이 buffer size 넘은거다 즉 정해 놓은 버퍼 사이즈 N을 넘었다는 의미
        # 이때는 오래된것을 비워야 한다 그 부분이 이 코드이다
        if self.ptr < self.size:
            sampling_prob[self.ptr-1] = 0


        index= torch.multinomial(sampling_prob,num_samples=batch_size,replacement=self.replacement) # 복원 추출 true
                                                                                                  # (batchsize,)

        importance_sampling_wieght =(self.size*sampling_prob[index])**(-self.beta)
        normed_importance_sampling_weight = (importance_sampling_wieght/importance_sampling_wieght.max()).unsqueeze(-1) # (batch size,1)


        # state,action,reward,next state,terminate,tuncate,index,normal IS weight
        return self.state[index],self.action[index],self.reward[index],self.state[index+1],self.terminate[index],self.truncated[index],index,normed_importance_sampling_weight


