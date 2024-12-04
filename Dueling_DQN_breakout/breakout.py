import collections
import gym
import numpy as np
import torch
from PIL import Image
"""
https://gymnasium.farama.org/environments/atari/breakout/#breakout

action space : 4


"""
class DQN_Breakout(gym.Wrapper):

    def __init__(self,render_mode ="rgb_array",repeat=4,device="cuda:0"):

        env = gym.make("ALE/Breakout-v5",render_mode=render_mode)
        super(DQN_Breakout,self).__init__(env)
        #self.env = gym.make("BreakoutNoFrameskip-v4", render_mode=render_mode)
        self.device = device
        self.repeat = repeat
        self.frame_buffer = []
        self.image_shape = (84,84) # breakout atari game image shape
        self.lives = env.ale.lives() # 게임속 목숨 정보

    def step(self,action):

        total_reward = 0
        terminate = False

        # 1 movement  : 4 frame

        for i in range(self.repeat):

            observation,reward,terminate,info = self.env.step(action)

            total_reward += reward

            current_live = info['lives'] # 게임속 목숨 정보

            # 여기 코드 진입은 목숨이 줄었다는 의미이다
            # 목숨이 줄었으니 reward도 같이 줄도록 한다
            if current_live < self.lives:
                total_reward = total_reward-1
                self.lives = current_live

            self.frame_buffer.append(observation) # 4개의 observation

            if terminate:
                break

        # 마지막 2개의 fram중에서 max frame만 계산 하겠다
        max_frame = np.max(self.frame_buffer[-2:],axis=0)
        max_frame = self.preprocess(max_frame)




        return max_frame,total_reward,terminate


    # 매번 reset과정 필요 하다 이때 state는 전처리된 결과를 반환 한다
    def reset(self):
        self.frame_buffer=[]
        observation, _ = self.env.reset()

        self.lives = self.env.ale.lives()

        observation = self.preprocess(observation)
        return observation


    def preprocess(self,observation):

        img = Image.fromarray(observation)

        img = img.convert("L")  #convert gray channel
        #img = img.resize(self.image_shape)  # reshape to self.image_shape -> game 화면 크기
        # 160 210
        img = np.array(img)


        img = img /255  # 0-255


        return img #210 160
