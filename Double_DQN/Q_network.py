
"""
implement double dqn with cnn and dueling dqn

select Q network through option argument in main file


"""

import torch
from torch import nn

class Double_DQN(nn.Module):
    def __init__(self,action_dim=2):
        super(Double_DQN,self).__init__()


        self.network = nn.Sequential(

            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim),
        )
    def forward(self,x):
        x = torch.Tensor(x).to('cuda')
        # x = x.unsqueeze(0)
        # x = x.unsqueeze(0)
        # x=x.view(1,4,1,1)

        return self.network(x/255)


class Dueling_DQN(nn.Module):

    def __init__(self,nb_action=4):
        super(Dueling_DQN,self).__init__()

        self.relu = nn.ReLU()

        # preprocess에서 gray scale입력이 들어올것이라서 1채널로 시작 한다
        self.conv1 = nn.Conv2d(1,32,kernel_size=(8,8),stride=(4,4))
        self.conv2 = nn.Conv2d(32,64,kernel_size=(4,4),stride=(2,2))
        self.conv3 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1))

        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=0.2)

        # (84,84)를 conv3까지 진행 한 결과
        self.advantage1 = nn.Linear(3136,1024) #22528
        self.advantage2 = nn.Linear(1024, 1024)
        self.advantage3 = nn.Linear(1024, nb_action)

        self.state_value1 = nn.Linear(3136,1024) #3136
        self.state_value2 = nn.Linear(1024, 1024)
        self.state_value3 = nn.Linear(1024, 1)

    def forward(self,x):

        x = torch.Tensor(x).to('cuda') # convert to tensor
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.flatten(x)
        state_value = self.relu(self.state_value1(x))
        #state_value = self.dropout(state_value)
        state_value = self.relu(self.state_value2(state_value))
        #state_value = self.dropout(state_value)
        state_value = self.relu(self.state_value3(state_value))

        action_value = self.relu(self.advantage1(x))
        #action_value = self.dropout(action_value)
        action_value = self.relu(self.advantage2(action_value))
        #action_value = self.dropout(action_value)
        action_value = self.relu(self.advantage3(action_value))

        output = state_value + (action_value - action_value.mean())

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