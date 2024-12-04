

"""

** for test **  : evaluate policy then caculate total rewarld GT

**linear schedule** : beta, alpha, epsilon have to convergence at the end of training
so Need adjust parameter scheduler



"""

import gym
import numpy as np

import collections

_field_names = [
    "state",
    "action",
    "reward",
    "next_state",
    "done"
]
Experience = collections.namedtuple("Experience", field_names=_field_names)

def eval_policy(policy, env_name, seed,render=False,eval_episodes=10):

    eval_env = gym.make(env_name)
    #eval_env.seed(seed + 100)

    avg_reward = 0.

    for _ in range(eval_episodes):
        state, _ =eval_env.reset()
        done = False

        while True:
            if(render):
                eval_env.render()

            action = policy.select_action(state)
            state, reward, done, i , _= eval_env.step(action)
            avg_reward += reward
            if done or i:
                break

    avg_reward /= eval_episodes

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    #eval_env.close()
    return avg_reward


class LinearSchedule(object):
    def __init__(self,schedule_timesteps,initial_p,final_p):
        self.schedule_timesteps = schedule_timesteps
        self.initial_p = initial_p
        self.final_p = final_p

    # 조금식 조금식 final_p로 설정한 값으로 가까워 지도록 스케줄링 하는 함수
    def value(self,t):

        fraction = min(float(t)/self.schedule_timesteps,1.0)
        return self.initial_p+fraction*(self.final_p-self.initial_p)

