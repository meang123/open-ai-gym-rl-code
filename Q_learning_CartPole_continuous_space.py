"""
Demonstration of the Cart Pole OpenAI Gym environment

Author: Aleksandar Haber

copy right : https://aleksandarhaber.com/q-learning-in-python-with-tests-in-cart-pole-openai-gym-environment-reinforcement-learning-tutorial/

"""

import numpy as np
import time
import gym
import matplotlib.pyplot as plt

class Q_learning:
    """
    Inputs :
    env - cartpole env
    alpha,gamma : hyper parameter(learning rate, discount factor)
    epsilon - for epsilon greedy
    episode : total number of episode

    +++++++++++++++++++++++++
    continuous_states - this is a 4 dimensional list that defines the number of grid points

    input like that

    numberOfBinsPosition=30
    numberOfBinsVelocity=30
    numberOfBinsAngle=30
    numberOfBinsAngleVelocity=30
    numberOfBins=[numberOfBinsPosition,numberOfBinsVelocity,numberOfBinsAngle,numberOfBinsAngleVelocity]

    1*4 vector 즉 각각의 state의 continuous space의 값들이 리스트 형식으로 들어가 있다
    즉 position vel angle angule velocity 모두 각각 30개씩 continuous value가 들어있는것이다

    +++++++++++++++++++++++++++++++


    # for state discretization

    to convert continous space to interval range

    state : position, velocity, angle, angle velocity


    # lowerBounds - lower bounds (limits) for discretization, list with 4 entries:
    # lower bounds on cart position, cart velocity, pole angle, and pole angular velocity

    # upperBounds - upper bounds (limits) for discretization, list with 4 entries:
    # upper bounds on cart position, cart velocity, pole angle, and pole angular velocity

    """
    def __init__(self,env,alpha,gamma,epsilon,episodes,continuous_states,lower_bounds,upper_bounds):

        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay_rate = 0.00001
        self.rng = np.random.default_rng() # set default random number generator
        self.action_number = env.action_space.n
        self.episodes = episodes
        self.continuous_states = continuous_states
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds

        # total reward sum list
        self.Gt_reward=[]

        # Q matrix initialize
        self.Q_matrix = np.random.uniform(0,1,size=(continuous_states[0],continuous_states[1],continuous_states[2],continuous_states[3],self.action_number))


    """
    
    coutinuous_ space, lower_bound, upper bound를 이용해서 discretization grid로 만드는 함수
    -> Q matrix에 사용됨 
    
    
    inputs : state list(position,velocity,angle,angle velocity)
    
    output : 4 dim tuple 
    """
    def return_index_state(self,state):
        position = state[0]
        velocity = state[1]
        angle = state[2]
        angular_velocity = state[3]

        position_bin = np.linspace(self.lower_bounds[0],self.upper_bounds[0],self.continuous_states[0])
        velocity_bin=np.linspace(self.lower_bounds[1],self.upper_bounds[1],self.continuous_states[1])
        angle_bin=np.linspace(self.lower_bounds[2],self.upper_bounds[2],self.continuous_states[2])
        angular_velocity_bin =np.linspace(self.lower_bounds[3],self.upper_bounds[3],self.continuous_states[3])

        # np.digitize
        # 값을 정해진 범위 인덱스 반환한다
        # https://numpy.org/doc/stable/reference/generated/numpy.digitize.html

        # for index

        index_position = np.digitize(state[0],position_bin)-1
        index_velocity = np.digitize(state[1], velocity_bin) - 1
        index_angle = np.digitize(state[2], angle_bin) - 1
        index_angular_velocity = np.digitize(state[3], angular_velocity_bin) - 1

        return tuple([index_position,index_velocity,index_angle,index_angular_velocity])

    """
    select action based of current state 
    
    원래 코드에서는 에피소드 개수에 따라서 액션 선택하였다 
    하지만 여기서는 바로 입실론 그리디 적용하였다 
    
    """
    def select_Action(self,state):


        if self.rng.random() < self.epsilon:
            self.epsilon = max(self.epsilon - self.epsilon_decay_rate, 0)
            return np.random.choice(self.action_number)
        else:
            self.epsilon = max(self.epsilon - self.epsilon_decay_rate, 0)
            return np.random.choice(np.where(self.Q_matrix[self.return_index_state(state)]==np.max(self.Q_matrix[self.return_index_state(state)]))[0])



    def simulate_episodes(self):

        for index_episode in range(self.episodes):

            # rewards per episode list -> for tracking convergence
            reward_episode=[]

            # reset env every Episodes

            (stateS,_) = self.env.reset()
            stateS = list(stateS)
            print("Simulating episode {}".format(index_episode))

            # loop untill terminate condition
            terminate_state = False
            while(not terminate_state):

                #return discretized index of the state
                stateS_index = self.return_index_state(stateS)

                #select action

                actionA = self.select_Action(stateS)

                (next_state,reward,terminate_state,_,_) = self.env.step(actionA)

                reward_episode.append(reward)
                next_state = list(next_state)
                next_state_index = self.return_index_state(next_state)

                Q_prime = np.max(self.Q_matrix[next_state_index])

                if not terminate_state:
                    error = reward+self.gamma*Q_prime-self.Q_matrix[stateS_index+(actionA,)]
                    self.Q_matrix[stateS_index+(actionA,)] = self.Q_matrix[stateS_index+(actionA,)]+self.alpha*error

                else:
                    # terminate state에는 Q_prime이 0이니까 이걸 고려한 gradient acsent식을 적용한다
                    # terminate state에도 적용한다 왜냐하면 공식 사이트에서 reward를 terminate state까지 포함하기로 했기 때문

                    error = reward-self.Q_matrix[stateS_index+(actionA,)]
                    self.Q_matrix[stateS_index+(actionA,)] = self.Q_matrix[stateS_index+(actionA,)]+self.alpha*error

                # update current state to next state
                stateS = next_state

            print("Sum of rewards {}".format(np.sum(reward_episode)))
            self.Gt_reward.append(np.sum(reward_episode))



    """
    simulating the final learned optimal policy 
    
    train으로 구한 optimal policy를 simulate한다 즉 test 
    
    output : 
    
        env1 : created cart pole env
        obtainedRewrad : a list of obtained rewards during time step of a single episode
        
    """

    def simulate_learned_strategy(self):
        env1 = gym.make("CartPole-v1",render_mode ='human')
        (currentState, _) = env1.reset()
        env1.render()
        timeSteps = 1000

        obtainedRewards = []  #btained rewards at every time step

        for timeIndex in range(timeSteps):
            print(timeIndex)
            # select greedy actions
            actionInStateS=np.random.choice(np.where(self.Q_matrix[self.return_index_state(currentState)]==np.max(self.Q_matrix[self.return_index_state(currentState)]))[0])
            currentState, reward, terminated, truncated, info =env1.step(actionInStateS)
            obtainedRewards.append(reward)
            time.sleep(0.05)
            if (terminated):
                time.sleep(1)
                break
        return obtainedRewards,env1


    """
    simulate random action 
    
    random policy구하는 함수 - optimal policy와 비교 하기 위한 baseline 
    """
    def simulte_random_strategy(self):

        env2 = gym.make('CartPole-v1')
        (currentState, _) = env2.reset()
        env2.render()
        # number of simulation episodes
        episodeNumber = 100
        # time steps in every episode
        timeSteps = 1000
        # sum of rewards in each episode
        sumRewardsEpisodes = []

        for episodeIndex in range(episodeNumber):
            rewardsSingleEpisode = []
            initial_state = env2.reset()
            print(episodeIndex)
            for timeIndex in range(timeSteps):
                random_action = env2.action_space.sample()
                observation, reward, terminated, truncated, info = env2.step(random_action)
                rewardsSingleEpisode.append(reward)
                if (terminated):
                    break
            sumRewardsEpisodes.append(np.sum(rewardsSingleEpisode))
        return sumRewardsEpisodes, env2


if __name__ == '__main__':
    env = gym.make("CartPole-v1")
    (state, _) = env.reset()

    # Cart pole 공식 문서에서 velocity와 angle velocity의 범위는 무한이라서 그래서 임의로 설정하는거다

    upperBounds = env.observation_space.high
    lowerBounds = env.observation_space.low
    cartVelocityMin = -3
    cartVelocityMax = 3
    poleAngleVelocityMin = -10
    poleAngleVelocityMax = 10
    upperBounds[1] = cartVelocityMax
    upperBounds[3] = poleAngleVelocityMax
    lowerBounds[1] = cartVelocityMin
    lowerBounds[3] = poleAngleVelocityMin

    # continuous value 개수 설정
    numberOfBinsPosition = 30
    numberOfBinsVelocity = 30
    numberOfBinsAngle = 30
    numberOfBinsAngleVelocity = 30
    numberOfBins = [numberOfBinsPosition, numberOfBinsVelocity, numberOfBinsAngle, numberOfBinsAngleVelocity]

    # define the parameters
    alpha = 0.1
    gamma = 1
    epsilon = 0.2
    numberEpisodes = 15000

    # create an object
    Q1 = Q_learning(env, alpha, gamma, epsilon, numberEpisodes, numberOfBins, lowerBounds, upperBounds)

    # run the Q-Learning algorithm
    Q1.simulate_episodes()

    # simulate the learned strategy
    (obtainedRewardsOptimal, env1) = Q1.simulate_learned_strategy()

    plt.figure(figsize=(12, 5))
    # plot the figure and adjust the plot parameters
    plt.plot(Q1.Gt_reward, color='blue', linewidth=1)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.yscale('log')
    plt.show()
    plt.savefig('convergence.png')

    # close the environment
    env1.close()
    # get the sum of rewards
    np.sum(obtainedRewardsOptimal)

    # now simulate a random strategy
    (obtainedRewardsRandom, env2) = Q1.simulte_random_strategy()
    plt.hist(obtainedRewardsRandom)
    plt.xlabel('Sum of rewards')
    plt.ylabel('Percentage')
    plt.savefig('histogram.png')
    plt.show()

