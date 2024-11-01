# 초기 버퍼 생성
# 워커에서 매 experience 마다 add 해서 버퍼업데
# 센트럴은 5초마다 버퍼를 통해 학습
# 워커에서 일정 스텝만큼 진행하면 센트럴로부터 파라미터 가져와서 업데이트

import ray
import numpy as np
import time
import argparse
from safe_slace.algo import LatentPolicySafetyCriticSlac
import torch

from WrappedGymEnv import WrappedGymEnv
import gym
from safe_slace.trainer import Trainer
import json
from configuration import get_default_config
import carla_rl_env
from configuration import get_default_config

FLAG=True# 삭제해야하는 코드


# 통합을 담당하는 Actor 정의
@ray.remote(num_gpus=1)
class ParameterServer:
    def __init__(self, args, expected_workers=2):
        self.args = args
        self.expected_workers = expected_workers
        
        # 환경 설정
        self.env, self.env_test = self.setup_environment()
        
        # 로그 디렉토리 설정
        self.log_dir = self.setup_log_directory()

        # SLAC 알고리즘 초기화
        self.algo = self.initialize_slac_algorithm()

        # 버퍼 초기화
        self.buffer = self.algo.buffer

        self.trainer = self.initialize_trainer()
        self.trainer.writer.add_text("config", json.dumps(vars(args)), 0) #텐서보드 시각화 위한 코드


        print("ParameterServer initialized successfully.")

    def setup_environment(self):
        # 환경 파라미터 정의
        params = {
            'carla_port': 2000,
            'map_name': 'Town10HD',
            'window_resolution': [1080, 1080],
            'grid_size': [3, 3],
            'sync': True,
            'no_render': False,
            'display_sensor': True,
            'ego_filter': 'vehicle.tesla.model3',
            'num_vehicles': 50,
            'num_pedestrians': 20,
            'enable_route_planner': True,
            'sensors_to_amount': ['left_rgb', 'front_rgb', 'right_rgb', 'top_rgb', 'lidar', 'radar'],
        }

        # 환경 설정
        env = WrappedGymEnv(gym.make(self.args.domain_name, params=params), 
                             action_repeat=self.args.action_repeat, 
                             image_size=64)
        env_test = env  # 테스트 환경을 동일하게 설정
        
        return env, env_test

    def setup_log_directory(self):
        return os.path.join(
            "logs",
            f"{self.args.domain_name}-{self.args.task_name}",
            f'slac-seed{self.args.seed}-{datetime.now().strftime("%Y%m%d-%H%M")}',
        )

    def initialize_slac_algorithm(self):
        return LatentPolicySafetyCriticSlac(
            state_shape=self.env.observation_space.shape,
            ometer_shape=self.env.ometer_space.shape,
            tgt_state_shape=self.env.tgt_state_space.shape,
            action_shape=self.env.action_space.shape,
            action_repeat=self.args.action_repeat,
            device=torch.device("cuda" if self.args.cuda else "cpu"),
            seed=self.args.seed,
            buffer_size=self.args.buffer_size,
            num_sequences=self.args.num_sequences,
            feature_dim=self.args.feature_dim,
            z1_dim=self.args.z1_dim,
            z2_dim=self.args.z2_dim,
            hidden_units=self.args.hidden_units,
            batch_size_latent=self.args.batch_size_latent,
            batch_size_sac=self.args.batch_size_sac,
            lr_sac=self.args.lr_sac,
            lr_latent=self.args.lr_latent,
            start_alpha=self.args.start_alpha,
            start_lagrange=self.args.start_lagrange,
            grad_clip_norm=self.args.grad_clip_norm,
            tau=self.args.tau,
            image_noise=self.args.image_noise,
        )

    def load_model(self):
        self.algo.load_model("logs/tmp")

    def initialize_trainer(self):
        return Trainer(
            num_sequences=self.args.num_sequences,
            env=self.env,
            env_test=self.env_test,
            algo=self.algo,
            log_dir=self.log_dir,
            seed=self.args.seed,
            num_steps=self.args.num_steps,
            initial_learning_steps=self.args.initial_learning_steps,
            initial_collection_steps=self.args.initial_collection_steps,
            collect_with_policy=self.args.collect_with_policy,
            eval_interval=self.args.eval_interval,
            num_eval_episodes=self.args.num_eval_episodes,
            action_repeat=self.args.action_repeat,
            train_steps_per_iter=self.args.train_steps_per_iter,
            env_steps_per_train_step=self.args.env_steps_per_train_step
        )

    def add_experience(self, experience): # 워커의 experience가 저장, woker에서 실행되는 코드
        self.buffer.add(experience)


    #예전코드
    def update_hyperparameters(self, t):
        self.BETA = self.beta_scheduler.value(t)
        td_errors = self.buffer.buffer[:len(self.buffer)]["priority"]
        td_mean = np.mean(td_errors)
        td_std = np.std(td_errors)
        self.BUFFER_ALPHA = self.buffer_alpha_scheduler.value(td_mean, td_std)

    def train_policy(self, t): #
        if len(self.buffer) > self.batch_size and t > self.start_timesteps:

            print(f"policy 업데이트. 경험 개수: {len(self.buffer)}")
            self.policy.train(self.BETA, self.BUFFER_ALPHA, self.buffer)
            self.update_hyperparameters(t)
         
            # Actor lr scheduler
            for p in self.policy.actor_optimizer.param_groups:
                p['lr'] = self.actor_lr_scheduler.value(t)

            # Critic lr scheduler
            for p in self.policy.critic_optimizer.param_groups:
                p['lr'] = self.critic_lr_scheduler.value(t)

            for p in self.policy.critic_optimizer2.param_groups:
                p['lr'] = self.critic_lr_scheduler.value(t)
    
    def get_policy_weights(self):
        weights = {}
        weights['actor'] = {name: param.detach().cpu().numpy() for name, param in self.policy.actor.named_parameters()}
        weights['critic1'] = {name: param.detach().cpu().numpy() for name, param in self.policy.critic.named_parameters()}
        weights['critic2'] = {name: param.detach().cpu().numpy() for name, param in self.policy.critic2.named_parameters()}
        weights['log_alpha'] = self.policy.log_alpha.detach().cpu().numpy()
        return weights