

"""

** for test **  : evaluate policy then caculate total rewarld GT

**linear schedule** : beta, alpha, epsilon have to convergence at the end of training
so Need adjust parameter scheduler



"""

def evaluate_policy(env,model,turns=3):
    scores =0

    for j in range(turns):
        s,info = env.reset()
        terminate = False

        while not terminate:

            # take deterministic action (항상 최고의 행동. 확률 선택 아니다)

            action = model.select_action(s,deterministic=True)
            s_next, reward,terminate,truncate,info = env.step(action)

            terminate = terminate+truncate
            scores+=reward
            s=s_next

        return int(scores/turns)



class LinearSchedule(object):
    def __init__(self,schedule_timesteps,initial_p,final_p):
        self.schedule_timesteps = schedule_timesteps
        self.initial_p = initial_p
        self.final_p = final_p

    # 조금식 조금식 final_p로 설정한 값으로 가까워 지도록 스케줄링 하는 함수
    def value(self,t):

        fraction = min(float(t)/self.schedule_timesteps,1.0)
        return self.initial_p+fraction*(self.final_p-self.initial_p)

