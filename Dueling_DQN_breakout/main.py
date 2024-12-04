import argparse
import os, shutil
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from breakout import DQN_Breakout
from PER import PER_Buffer
from utils import *
from Agent import *
import gym

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
parser.add_argument('--train_epoch', type=int, default=int(1e5), help='Max training steps') # 3e5
parser.add_argument('--lr_init', type=float, default=1.5e-4, help='Initial Learning rate')
parser.add_argument('--lr_end', type=float, default=6e-5, help='Final Learning rate')
parser.add_argument('--lr_decay_steps', type=float, default=int(1e5), help='Learning rate decay steps -> same train steps')
parser.add_argument('--batch_size', type=int, default=256, help='lenth of sliced trajectory')
parser.add_argument('--epsilon_init', type=float, default=0.9, help='init explore noise')
parser.add_argument('--epsilon_noise_end', type=float, default=0.01, help='final explore noise')
parser.add_argument('--epsilon_decay_steps', type=int, default=int(1e5), help='decay steps of explore noise')
parser.add_argument('--seed', type=int, default=4, help='random seed')

parser.add_argument('--update_every', type=int, default=50, help='training frequency')

parser.add_argument('--save_interval', type=int, default=int(1e4), help='Model saving interval, in steps.')

parser.add_argument('--gamma', type=float, default=0.99, help='Discounted Factor')
parser.add_argument('--warmup', type=int, default=int(3e3), help='steps for random policy to explore') # buffer에 어느정도 쌓여야 한다
parser.add_argument('--buffer_size', type=int, default=int(1e5), help='size of the replay buffer')
parser.add_argument('--alpha', type=float, default=0.6, help='alpha for PER')
parser.add_argument('--beta_init', type=float, default=0.4, help='beta for PER')

parser.add_argument('--eval_interval', type=int, default=int(1e3), help='Model evaluating interval, in steps.')
parser.add_argument('--beta_gain_steps', type=int, default=int(1e5), help='steps of beta from beta_init to 1.0')
parser.add_argument('--replacement', default=True, help='sample method') # 데이터 중복 될수있지만 빠르다
opt = parser.parse_args()
print(opt)

#model = Dueling_DQN()

def main():

    Buffer_ENV=["BreakoutNoFrameskip-v4",'CartPole-v1']



    algo_name="Dueiling_DQN"
    env = DQN_Breakout(device=device)
    eval_env = DQN_Breakout(device=device,render_mode='human')
    model = Dueling_DQN()
    opt.state_dim = env.observation_space.shape #[84, 84]  # env.observation_space.shape[0] (210,160,3)
    opt.action_dim = env.action_space.n



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
    buffer = agent.PER_buffer



    # set scheduler esiplon,learning rate, beta for IS


    EPSILON = opt.epsilon_init
    BETA = opt.beta_init

    epsilon_scheduler = LinearSchedule(opt.epsilon_decay_steps,opt.epsilon_init,opt.epsilon_noise_end)
    beta_scheduler = LinearSchedule(opt.beta_gain_steps,opt.beta_init,1.0) # beta end is 1.0 : scheduler must go to 1.0
    lr_scheduler = LinearSchedule(opt.lr_decay_steps,opt.lr_init,opt.lr_end)


    # test or train


    if opt.render: # Test
        score = evaluate_policy(env,agent,turns=20)
        print('INFERENCE : EnvName:', Buffer_ENV[opt.EnvIdex], 'seed:', opt.seed, 'score:', score)

    else: # Train

        total_steps = 0


        # train epoch
        while total_steps < opt.train_epoch:

            state = env.reset()
            scores = 0
            done = False


            while True:

                total_steps += 1
                action = agent.select_action(state, epsilon=EPSILON, deterministic=False)
                next_state,reward,done = env.step(action)
                experience = Experience(state, action, reward, next_state, done)

                buffer.add(experience)



                if done:
                    #print("\nwhile break\n")
                    break

                #print(f"{total_steps} : {buffer.buffer_length} {agent.has_enough_experience()}")

# update every : training frequency
                # train 50 times every 50 epoch rather than 1 training per epoch
                if agent.has_enough_experience() and total_steps % opt.update_every == 0:
                    print(f"\nTrain!!{total_steps}\n")
                    for j in range(opt.update_every):
                        agent.train(beta=BETA)

# model parameter : lr,epsilon,beta 조정
                    EPSILON = epsilon_scheduler.value(total_steps)
                    BETA = beta_scheduler.value(total_steps)

                    for p in agent.optimizer.param_groups:
                        p['lr'] = lr_scheduler.value(total_steps)

                state = next_state  # next state (1,1,84,84)
                scores += reward

                # record & log
                if (total_steps) % opt.eval_interval == 0:
                    score = evaluate_policy(eval_env, agent,EPSILON,turns=5)
                    if opt.write:

                        writer.add_scalar('score', score, global_step=total_steps)
                        writer.add_scalar('total_reward', scores, global_step=total_steps)
                        writer.add_scalar('epsilon', EPSILON, global_step=total_steps)
                        writer.add_scalar('beta', BETA, global_step=total_steps)
                        #priorities = buffer.priorities[0: buffer.size].cpu()

                        # writer.add_scalar('p_max', priorities.max(), global_step=total_steps)
                        # writer.add_scalar('p_sum', priorities.sum(), global_step=total_steps)
                    print('EnvName:',Buffer_ENV[opt.EnvIdex],'seed:',opt.seed,'steps: {}k'.format(int(total_steps/1000)),'score:', int(score))



                if (total_steps) % opt.save_interval == 0:
                    agent.save(algo_name, Buffer_ENV[opt.EnvIdex], int(total_steps / 1000))

            print(f"\n--------{total_steps} score is {scores}--------\n")




    env.close()
    eval_env.close()



if __name__ == '__main__':

    main()
