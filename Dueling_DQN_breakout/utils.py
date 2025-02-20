import collections
import time
import gym
_field_names = [
    "state",
    "action",
    "reward",
    "next_state",
    "done"
]
Experience = collections.namedtuple("Experience", field_names=_field_names)

"""

** for test **  : evaluate policy then caculate total rewarld GT

**linear schedule** : beta, alpha, epsilon have to convergence at the end of training
so Need adjust parameter scheduler



"""

def eval_policy(policy, env_name,render=False,eval_episodes=10):

    if render:
        eval_env = gym.make(env_name, render_mode="human")
    else:
        eval_env = gym.make(env_name)
    #eval_env.seed(seed + 100)

    avg_reward = 0.

    for _ in range(eval_episodes):
        state, _ =eval_env.reset()
        done = False

        while True:
            if(render):
                eval_env.render()
                time.sleep(0.01)

            action = policy.select_action(state,eval_env)
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

class time_base_schedule(object):
    def __init__(self,alpha_max,alpha_min,total_timestep):
        self.C = 0.05
        self.total_timestep = total_timestep
        self.alpha_max = alpha_max
        self.alpha_min = alpha_min

    def value(self,cur_timestep):

        #differ = td_mean-td_std
        progress = cur_timestep / self.total_timestep
        # sigmoid 함수 입력: 진행도가 0.5일 때 중간값이 나오도록 변환
        # 진행도가 낮으면 음수, 높으면 양수가 되어 sigmoid가 0에서 1로 변환됨
        logistic_input = (progress - 0.5) / self.C
        alpha = self.alpha_min+(self.alpha_max-self.alpha_min)*(1/(1+np.exp(-logistic_input)))
        return alpha



