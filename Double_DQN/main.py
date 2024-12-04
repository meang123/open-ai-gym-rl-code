import argparse
import os, shutil
from datetime import datetime
import gym
import torch
import numpy as np

from torch.utils.tensorboard import SummaryWriter
from Q_network import Double_DQN,Dueling_DQN
from DQN_Breakout import DQN_Breakout
from PriorReplayBuffer import Prior_Replay_Buffer
from utils import evaluate_policy,LinearSchedule
from Agent import Agent


os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("\nDevice is ",device)
print()

parser = argparse.ArgumentParser()

# env 선택 cart pole (DDQN), breakout (Dueiling)
parser.add_argument('--EnvIdex', type=int, default=0, help='Breakout,CP-v1')

parser.add_argument('--render', type=bool, default=False, help='Render or Not , render human mode for test, rendoer rgb array for train')

parser.add_argument('--write', type=bool, default=True, help='summary T/F')
parser.add_argument('--ModelIdex', type=int, default=20, help='which model to load')
parser.add_argument('--Loadmodel', type=bool, default=False, help='Load pretrained model or Not') # 훈련 마치고 나서는 True로 설정 하기
parser.add_argument('--train_epoch', type=int, default=int(2e5), help='Max training steps') # 3e5
parser.add_argument('--lr_init', type=float, default=1.5e-4, help='Initial Learning rate')
parser.add_argument('--lr_end', type=float, default=6e-5, help='Final Learning rate')
parser.add_argument('--lr_decay_steps', type=float, default=int(3e5), help='Learning rate decay steps -> same train steps')
parser.add_argument('--batch_size', type=int, default=256, help='lenth of sliced trajectory')
parser.add_argument('--epsilon_init', type=float, default=0.9, help='init explore noise')
parser.add_argument('--epsilon_noise_end', type=float, default=0.03, help='final explore noise')
parser.add_argument('--epsilon_decay_steps', type=int, default=int(1e5), help='decay steps of explore noise')
parser.add_argument('--seed', type=int, default=32, help='random seed')
parser.add_argument('--update_every', type=int, default=50, help='training frequency')
parser.add_argument('--save_interval', type=int, default=int(5e4), help='Model saving interval, in steps.')
parser.add_argument('--gamma', type=float, default=0.99, help='Discounted Factor')
parser.add_argument('--warmup', type=int, default=int(3e3), help='steps for random policy to explore') # buffer에 어느정도 쌓여야 한다
parser.add_argument('--buffer_size', type=int, default=int(1e5), help='size of the replay buffer')
parser.add_argument('--alpha', type=float, default=0.6, help='alpha for PER')
parser.add_argument('--beta_init', type=float, default=0.4, help='beta for PER')
parser.add_argument('--eval_interval', type=int, default=int(1e3), help='Model evaluating interval, in steps.')
parser.add_argument('--beta_gain_steps', type=int, default=int(3e5), help='steps of beta from beta_init to 1.0')
parser.add_argument('--replacement', default=True, help='sample method') # 데이터 중복 될수있지만 빠르다
opt = parser.parse_args()
print(opt)

#model = Dueling_DQN()

def main():

    Buffer_ENV=["BreakoutNoFrameskip-v4",'CartPole-v1']

    if(opt.EnvIdex==0):

        algo_name="Dueiling_DQN"
        env = DQN_Breakout(device=device)
        eval_env = DQN_Breakout(device=device,render_mode='human')
        model = Dueling_DQN()
        opt.state_dim = [84, 84]  # env.observation_space.shape[0] (210,160,3)
        opt.action_dim = env.action_space.n


    else:
        algo_name="Double_DQN"
        env = gym.make(Buffer_ENV[opt.EnvIdex])
        eval_env = gym.make(Buffer_ENV[opt.EnvIdex],render_mode = 'human')

        opt.state_dim = env.observation_space.shape
        opt.action_dim = env.action_space.n
        model = Double_DQN(action_dim=opt.action_dim)




    agent = Agent(model,opt,device=device)

# random seed 고정
    torch.manual_seed(opt.seed)
    np.random.seed(opt.seed)

    print("Algorithm name ",algo_name,"\nENV name ",Buffer_ENV[opt.EnvIdex],"\nENV state space ",opt.state_dim,"\nENV action space ",opt.action_dim,"\nRandom seed ",opt.seed)

# setting summaryWriter
    if opt.write:

        writepath = f'runs/LightPrior{algo_name}_{Buffer_ENV[opt.EnvIdex]}' + datetime.now().strftime('%Y-%m-%d')

        if os.path.exists(writepath):

            shutil.rmtree(writepath)
        writer = SummaryWriter(log_dir=writepath)

    # load the model
    if opt.Loadmodel:
        agent.load(algo_name, Buffer_ENV[opt.EnvIdex], opt.ModelIdex)

    # set replay buffer
    buffer = Prior_Replay_Buffer(opt,algo_name=algo_name)


    # set scheduler esiplon,learning rate, beta for IS

    epsilon_scheduler = LinearSchedule(opt.epsilon_decay_steps,opt.epsilon_init,opt.epsilon_noise_end)
    beta_scheduler = LinearSchedule(opt.beta_gain_steps,opt.beta_init,1.0) # beta end is 1.0 : scheduler must go to 1.0
    lr_scheduler = LinearSchedule(opt.lr_decay_steps,opt.lr_init,opt.lr_end)


    # test or train


    if opt.render: # Test
        score = evaluate_policy(eval_env,agent,turns=20)
        print('EnvName:', Buffer_ENV[opt.EnvIdex], 'seed:', opt.seed, 'score:', score)

    else: # Train
        total_steps = 0

        # train epoch
        while total_steps < opt.train_epoch:
            state,info = env.reset()

            # TD error 계산 후 prior 계산 위해 필요
            action,q_a = agent.select_action(state, deterministic=False)



            while True:
                next_state,reward,terminated,truncated,info = env.step(action)

                next_action,next_q_a = agent.select_action(next_state,deterministic=False)

# !!!!!Compute Priority with TD Error!!!!

                priority = (torch.abs(reward+(~terminated)*opt.gamma*next_q_a-q_a)+0.01)**opt.alpha

                buffer.add(state,action,reward,terminated,truncated,priority)


                state,action,q_a = next_state,next_action,next_q_a   #next state (1,1,84,84)

# update every : training frequency
                # train 50 times every 50 epoch rather than 1 training per epoch
                if total_steps >= opt.warmup and total_steps % opt.update_every==0:
                    for j in range(opt.update_every):
                        agent.train(buffer)


# model parameter : lr,epsilon,beta 조정
                    agent.esip_init=epsilon_scheduler.value(total_steps)
                    buffer.beta = beta_scheduler.value(total_steps)

                    for p in agent.optimizer.param_groups:
                        p['lr'] = lr_scheduler.value(total_steps)


# record & log
                if (total_steps) % opt.eval_interval==0:
                    score = evaluate_policy(eval_env, agent,turns=5)
                    if opt.write:
                        writer.add_scalar('ep_r', score, global_step=total_steps)
                        writer.add_scalar('noise', agent.esip_init, global_step=total_steps)
                        writer.add_scalar('beta', buffer.beta, global_step=total_steps)
                        #priorities = buffer.priorities[0: buffer.size].cpu()

                        # writer.add_scalar('p_max', priorities.max(), global_step=total_steps)
                        # writer.add_scalar('p_sum', priorities.sum(), global_step=total_steps)
                    print('EnvName:',Buffer_ENV[opt.EnvIdex],'seed:',opt.seed,'steps: {}k'.format(int(total_steps/1000)),'score:', int(score))

                total_steps += 1
                if (total_steps) % opt.save_interval == 0:
                    agent.save(algo_name, Buffer_ENV[opt.EnvIdex], int(total_steps / 1000))



                if(terminated or truncated):
                    break

    env.close()
    eval_env.close()



if __name__ == '__main__':

    main()
