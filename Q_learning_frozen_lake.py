"""
copy right
https://github.com/johnnycode8/gym_solutions/blob/main/frozen_lake_q.py

"""

import gym
import numpy as np
import matplotlib.pyplot as plt
import pickle





def run(episodes, is_training=True, render=False):

    env = gym.make('FrozenLake-v1', map_name="8x8", is_slippery=True, render_mode='human' if render else None)
    # is_slippery는 agent의 random성을 높이는 옵션이다 원래 가려는 방향과 다르게 가는것을 의미한다


    if(is_training):
        # Q table 초기화 하는게 중요하다 -> 문제마다 초기화 다름
        q = np.zeros((env.observation_space.n, env.action_space.n)) # init a 64 x 4 array
    else:
        f = open('frozen_lake8x8.pkl', 'rb')
        q = pickle.load(f)
        f.close()

    learning_rate_a = 0.3 # alpha or learning rate
    discount_factor_g = 0.99 # gamma or discount rate. Near 0: more weight/reward placed on immediate state. Near 1: more on future state.
    epsilon = 1         # 1 = 100% random actions
    epsilon_decay_rate = 0.0001        # epsilon decay rate. 1/0.0001 = 10,000
    rng = np.random.default_rng()   # random number generator

    #rewards_per_episode = np.zeros(episodes)
    Gt_reward=[]

    for i in range(episodes):
        state = env.reset()[0]  # states: 0 to 63, 0=top left corner,63=bottom right corner
        terminated = False      # True when fall in hole or reached goal
        truncated = False       # True if episode truncates due to a time limit or a reason that is not defined as part of the task MDP.

        G_episode=0

        while(not terminated and not truncated):

            # behavior policy
            # 입실론 greedy방식으로 탐험 탐색 방식 구현
            if is_training and rng.random() < epsilon:
                action = env.action_space.sample() # actions: 0=left,1=down,2=right,3=up
            else:
                action = np.argmax(q[state,:])

            # 해당 action 취했을때 new state ,reward, terminated, tuncated,_
            new_state,reward,terminated,truncated,_ = env.step(action)

            if is_training:
                # target policy(updating) -> greedy policy -> choose max value
                q[state,action] = q[state,action] + learning_rate_a * (
                    reward + discount_factor_g * np.max(q[new_state,:]) - q[state,action]
                )

            G_episode+=reward

            state = new_state

        epsilon = max(epsilon - epsilon_decay_rate, 0)

        if(epsilon==0):
            learning_rate_a = 0.0001

        Gt_reward.append(G_episode)


    env.close()

    print("reward converge ",sum(Gt_reward)/episodes)
    print("success rate : ",sum(Gt_reward)/len(Gt_reward))

    plt.plot(np.cumsum(Gt_reward)/np.arange(1,episodes+1),'.')


    plt.savefig('frozen_lake8x8.png')

    if is_training:
        f = open("frozen_lake8x8.pkl","wb")
        pickle.dump(q, f)
        f.close()

if __name__ == '__main__':

    run(50000)

    #run(3, is_training=False, render=True)
