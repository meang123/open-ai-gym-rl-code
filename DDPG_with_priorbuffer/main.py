import argparse
import torch
import gym
import numpy as np
import os
import time
from TD_3 import *

from DDPG import DDPG

from priority_replay_buffer import *
from torch.utils.tensorboard import SummaryWriter
from utils import *



os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("\nDevice is ",device)
print()

"""
원래는 1e5로 해야 하는데 시간이 너무 오래 걸려서 1e4까지만 도전해본다 모든 scale을 원래 보다 1/10으로 낮춰서 해보는 결과이다 
"""
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--render', type=bool, default=False,help='Render or Not , render human mode for test, rendoer rgb array for train')
    parser.add_argument("--policy", default="TD3")  # Policy name (TD3, DDPG or OurDDPG)
    parser.add_argument("--env", default='HalfCheetah-v4')  # OpenAI gym environment name
    parser.add_argument("--seed", default=0, type=int)  # Sets Gym, PyTorch and Numpy seeds

    parser.add_argument("--start_timesteps", default=500, type=int)  # Time steps initial random policy is used 25e3
    parser.add_argument("--eval_freq", default=5e4, type=int)  # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1e6, type=int)  # Max time steps to run environment
    parser.add_argument("--expl_noise", default=0.1, type=float)  # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=256, type=int)  # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99, type=float)  # Discount factor
    parser.add_argument("--tau", default=0.005, type=float)  # Target network update rate


    parser.add_argument("--policy_noise", default=0.2)  # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5)  # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)  # Frequency of delayed policy updates
    parser.add_argument("--save_model", action="store_true")  # Save model and optimizer parameters
    parser.add_argument("--load_model", default="")  # Model load file name, "" doesn't load, "default" uses file_name
    parser.add_argument('--gamma', type=float, default=0.99, help='Discounted Factor')

    parser.add_argument("--alpha_min", default=0.0, type=float)  # PER buffer alpha
    parser.add_argument("--alpha_max", default=0.9, type=float)  # PER buffer alpha

    parser.add_argument('--beta_init', type=float, default=0.4, help='beta for PER')
    parser.add_argument('--beta_gain_steps', type=int, default=int(1e6), help='steps of beta from beta_init to 1.0')
    parser.add_argument('--lr_init', type=float, default=3e-4, help='Initial Learning rate')
    parser.add_argument('--lr_end', type=float, default=6e-5, help='Final Learning rate')
    parser.add_argument('--lr_decay_steps', type=float, default=int(1e6), help='Learning rate decay steps')
    parser.add_argument('--write', type=bool, default=True, help='summary T/F')
    parser.add_argument('--save_interval', type=int, default=int(10e4), help='Model saving interval, in steps.')
    parser.add_argument('--Loadmodel', type=bool, default=False,help='Load pretrained model or Not')  # 훈련 마치고 나서는 True로 설정 하기
    parser.add_argument('--buffer_size', type=int, default=int(1e6), help='size of the replay buffer')

    args = parser.parse_args()


    file_name = f"{args.policy}_{args.env}_{args.seed}_2"
    print("---------------------------------------")
    print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
    print("---------------------------------------")

    if not os.path.exists(f"./results/{file_name}"):
        os.makedirs(f"./results/{file_name}")

    if not os.path.exists(f"./models/{file_name}"):
        os.makedirs(f"./models/{file_name}")

    env = gym.make(args.env)

    writer = SummaryWriter(log_dir=f"./results/{file_name}")
    args.summary_writer = writer

    # Set seeds
    #env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    env.action_space.seed(args.seed)

    args.state_dim = env.observation_space.shape[0]
    args.action_dim = env.action_space.shape[0]
    args.max_action = float(env.action_space.high[0])


    print(args)

    if args.policy == "TD3":
        policy = TD_3(args,device)
    elif args.policy =="DDPG":
        policy = DDPG(args,device)

    # load model

    buffer = PER_Buffer(args,device=device)






    BETA = args.beta_init
    ALPHA = args.alpha_min

    beta_scheduler = LinearSchedule(args.beta_gain_steps,args.beta_init,1.0) # beta end is 1.0 : scheduler must go to 1.0
    alpha_scheduler = time_base_schedule(args.alpha_max, args.alpha_min, args.max_timesteps)
    # actor_lr_scheduler = LinearSchedule(args.lr_decay_steps, args.lr_init, args.lr_end)
    # critic_lr_scheduler = LinearSchedule(args.lr_decay_steps, args.lr_init, args.lr_end)

    # load the model
    if args.Loadmodel:
        policy.load(f"./models/{file_name}")


    if args.render and args.Loadmodel: # Test
        score = eval_policy(policy, args.env,render=True)

        print('TEST EnvName:', args.env, 'seed:', args.seed, 'TEST score:', score)

    # train
    else:
        max_step = int(args.max_timesteps)
        total_steps = 0
        while total_steps < max_step:
            state,_= env.reset()
            done = False
            episode_reward = 0


            while True:

                action = policy.select_action(state)

                # Perform action
                next_state, reward, done, i, _ = env.step(action)
                experience = Experience(state, action, reward, next_state, done)



                buffer.add(experience)


                if done or i:
                    print(f"Done {done} tuncated {i}")
                    break

                # train
                if policy.has_enough_experience(buffer) and total_steps > args.start_timesteps:

                    policy.train(BETA, ALPHA, buffer)

                    BETA = beta_scheduler.value(total_steps)
                    # td_mean = np.mean(buffer.buffer[:len(buffer)]["priority"])
                    # td_std = np.std(buffer.buffer[:len(buffer)]["priority"])
                    ALPHA = alpha_scheduler.value(total_steps)

                    if total_steps % 1000 == 0:
                        print('scheduler step')
                        for sched in policy.scheds:
                            sched.step()


                state = next_state
                episode_reward += reward
                total_steps += 1

                # Evaluate episode
                if (total_steps + 1) % args.eval_freq == 0:
                    print("\nEvaluate score\n")
                    score = eval_policy(policy, args.env)
                    if args.write:
                        args.summary_writer.add_scalar('evaled_score', score, global_step=total_steps+1)
                        args.summary_writer.add_scalar('episode_reward', episode_reward, global_step=total_steps+1)
                        #args.summary_writer.add_scalar('critic_lr', critic_lr_scheduler.value(t), global_step=t)
                        #args.summary_writer.add_scalar('actor_lr', actor_lr_scheduler.value(t), global_step=t)
                        args.summary_writer.add_scalar('PER_alpha', ALPHA, global_step=total_steps + 1)
                        args.summary_writer.add_scalar('beta', BETA, global_step=total_steps+1)

                    policy.save(f"./models/{file_name}")
                    print('writer add scalar and save model   ','steps: {}k'.format(int(total_steps / 1000)), 'score:', int(score))



            print(f"\n--------{total_steps} score is {episode_reward}--------\n")

    env.close()

if __name__ == "__main__":
    main()