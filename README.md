# Open AI Gym RL code 정리 

강화 학습에 대해서 공부 할때 구현 한 코드 입니다. Frozen lake와 cartpole부터 시작하여 DQN,Dueling dqn, td3,sac 알고리듬까지 
코드 구현 하였습니다. 모든 알고리듬은 replay buffer 사용하는 대신 PER buffer를 사용하도록 코드 수정 하였습니다 


--- 

### Prioritized replay(PER) buffer 

우선 순위를 두어서 sampling합니다 이때 TD error가 큰값에 우선순위를 부여하여 계산합니다 
TD error가 크다는건 다음 상태의 보상과 현 상태의 보상차이가 크다는 의미 이므로 TD error를 우선순위 기준으로 삼는 이유 입니다 

PER buffer 도입할때 크게 2가지를 신경 써야 합니다 

* stochastic sampling prioritization
  * mini batch size만큼 update하기 때문에 TD error가 초반에 크게 나올수있다 따라서 alpha hyperparameter 도입해서 운선순위를 얼마나 사용할것인지 조절하는 파라미터가 도입이 됩니다 
  * Alpha가 1에 가까울수록 TD error 값에 우선순위를 두고 0에 가까워지면 uniform sampling을 합니다 
  
* importance sampling weights
    * 앞서 stochastic sampling 도입 때문에 distribution이 변형되면서 생기는 bias가 발생합니다 이를 해결하기 위해 importance sample weights 고려하여 조정합니다 
    * beta hyperparameter는 훈련이 끝나갈때는 1에 가까워져야 합니다 (선형 스케줄 사용해야 합니다)
  

    

### Double DQN , Dueling DQN breakout atari game

시간 관계상 학습을 목표치까지 학습 하지 못했지만 우상향하는 그래프를 확인할수있습니다

![img.png](img.png)




### SAC 
시간 관계상 학습을 목표치까지 학습 하지 못했지만 우상향하는 그래프를 확인할수있습니다


![img_1.png](img_1.png)


### DDPG and TD3



