# Open AI Gym RL code 정리 

강화 학습에 대해서 공부 할때 구현 한 코드 입니다. Frozen lake와 cartpole부터 시작하여 DQN,Dueling dqn, td3,sac 알고리듬까지 
코드 구현 하였습니다. 모든 알고리듬은 replay buffer 사용하는 대신 PER buffer를 사용하도록 코드 수정 하였습니다 

dueling dqn은 breakout 환경을 대상으로 진행하고 ddpg,td3,sac는 mujoco 환경을 대상을 진행했습니다(walkwe2d,halfcheetha)


sac 환경에 대해서만 PER buffer 에 대해 실험적으로 진행하고 스케줄러 도입한것이 좋다고 판단해서 나머지 알고리듬에 per buffer에 대해서는 스케줄러 도입였습니다 

--- 

## DDPG

### halfcheetha

![ddpg.png](image_data%2Fddpg.png)


--- 

## TD3

### halfcheetha

![td3_halfcheetha.png](image_data%2Ftd3_halfcheetha.png)


--- 

## SAC



1. **fixed alpha PER buffer**

alpha 값을 0.6으로 고정하고 버퍼가 가득 채워졌을때 우선 순위 값이 낮은 값을 우선적으로 대체 하여 버퍼를 채우는 방식으로 실험 진행했습니다 

아래는 walk2d 와 halfcheetha에 대해서 진행했습니다 

### walkwe2d 


![sac_walker2d.png](image_data%2Fsac_walker2d.png)



### halfcheetha 


![sac_halfcheetha_fix_alpha.png](image_data%2Fsac_halfcheetha_fix_alpha.png)


2. **time based scheduler PER buffer**

alpha 값을 고정 값으로 사용하는 대신 시간에 따라 변하도록 스케줄러를 도입하였습니다 처음에는 uniform sampling하도록 진행하고 학습이 끝나갈때는 alpha값을 높혀서 우선순위 높은 값의 샘플링 확률을 
올리도록 스케줄러를 구성하였습니다 그리고 1번 에서 per buffer와 달리 버퍼가 가득 찼을때 우선 순위 낮은거 위주로 교체 하는대신 
순환 버퍼를 사용하여서 우선 순위가 낮은 값도 버퍼내에서 유지하도록하여 다양성을 유지 했습니다


### halfcheetha 


![sac_halfcheetha_schduler_alpha.png](image_data%2Fsac_halfcheetha_schduler_alpha.png)


![sac_halfcheetha_scheduler.png](image_data%2Fsac_halfcheetha_scheduler.png)


![mujoco-2025-02-20-10-34-58.gif](image_data%2Fmujoco-2025-02-20-10-34-58.gif)


1번과 비교했을때 성능향상이 있었습니다 

