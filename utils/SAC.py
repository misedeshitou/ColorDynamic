import copy
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

from utils.utils_SAC import Double_Q_Net, Policy_Net, ReplayBuffer


class SAC_agent:
    def __init__(self, **kwargs):
        # Init hyperparameters for agent, just like "self.gamma = opt.gamma, self.lambd = opt.lambd, ..."
        self.__dict__.update(kwargs)
        self.tau = 0.005
        self.H_mean = 0
        self.replay_buffer = ReplayBuffer(self.state_dim, self.dvc, max_size=int(1e6))

        self.actor = Policy_Net(self.state_dim, self.action_dim, self.hid_shape).to(
            self.dvc
        )
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)

        self.q_critic = Double_Q_Net(
            self.state_dim, self.action_dim, self.hid_shape
        ).to(self.dvc)
        self.q_critic_optimizer = torch.optim.Adam(
            self.q_critic.parameters(), lr=self.lr
        )
        self.q_critic_target = copy.deepcopy(self.q_critic)
        # 新增计时变量
        self.timer_steps = 0
        self.timer_start = 0.0

        for p in self.q_critic_target.parameters():
            p.requires_grad = False

        if self.adaptive_alpha:
            # We use 0.6 because the recommended 0.98 will cause alpha explosion.
            self.target_entropy = 0.6 * (-np.log(1 / self.action_dim))  # H(discrete)>0
            self.log_alpha = torch.tensor(
                np.log(self.alpha), dtype=float, requires_grad=True, device=self.dvc
            )
            self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=self.lr)

    def select_action(self, state, deterministic):
        with torch.no_grad():
            state = torch.as_tensor(state, dtype=torch.float32, device=self.dvc)
            probs = self.actor(state)  # 假设输出形状为 (N, action_dim)

            if deterministic:
                # 取概率最大的动作
                a = probs.argmax(-1)  # 形状为 (N,)
            else:
                # 按照概率分布采样
                dist = Categorical(probs)
                a = dist.sample()  # 形状为 (N,)

            return a

    def train(self):
        # 1. 启动计时 (在第一次进入 train 时记录开始时间)
        if self.timer_steps == 0:
            self.timer_start = time.time()
        s, a, r, s_next, dw = self.replay_buffer.sample(self.batch_size)
        # --- 防错保险：确保所有张量维度对齐且在正确设备上 ---
        a = a.view(-1, 1).long().to(self.dvc)
        r = r.view(-1, 1).to(self.dvc)
        dw = dw.view(-1, 1).to(self.dvc)
        s = s.to(self.dvc)
        s_next = s_next.to(self.dvc)
        # ----------------------------------------------
        # ------------------------------------------ Train Critic ----------------------------------------#
        """Compute the target soft Q value"""
        with torch.no_grad():
            next_probs = self.actor(s_next)  # [b,a_dim]
            next_log_probs = torch.log(next_probs + 1e-8)  # [b,a_dim]
            next_q1_all, next_q2_all = self.q_critic_target(s_next)  # [b,a_dim]
            min_next_q_all = torch.min(next_q1_all, next_q2_all)
            v_next = torch.sum(
                next_probs * (min_next_q_all - self.alpha * next_log_probs),
                dim=1,
                keepdim=True,
            )  # [b,1]
            target_Q = r + (~dw) * self.gamma * v_next

        """Update soft Q net"""
        q1_all, q2_all = self.q_critic(s)  # [b,a_dim]
        q1, q2 = q1_all.gather(1, a), q2_all.gather(1, a)  # [b,1]
        q_loss = F.mse_loss(q1, target_Q) + F.mse_loss(q2, target_Q)
        self.q_critic_optimizer.zero_grad()
        q_loss.backward()
        self.q_critic_optimizer.step()

        # ------------------------------------------ Train Actor ----------------------------------------#
        probs = self.actor(s)  # [b,a_dim]
        log_probs = torch.log(probs + 1e-8)  # [b,a_dim]
        with torch.no_grad():
            q1_all, q2_all = self.q_critic(s)  # [b,a_dim]
        min_q_all = torch.min(q1_all, q2_all)

        a_loss = torch.sum(
            probs * (self.alpha * log_probs - min_q_all), dim=1, keepdim=False
        )  # [b,]

        self.actor_optimizer.zero_grad()
        a_loss.mean().backward()
        self.actor_optimizer.step()

        # ------------------------------------------ Train Alpha ----------------------------------------#
        if self.adaptive_alpha:
            with torch.no_grad():
                self.H_mean = -torch.sum(probs * log_probs, dim=1).mean()
            alpha_loss = self.log_alpha * (self.H_mean - self.target_entropy)

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp().item()

        # ------------------------------------------ Update Target Net ----------------------------------#
        for param, target_param in zip(
            self.q_critic.parameters(), self.q_critic_target.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

        self.timer_steps += 1
        # --- 训练逻辑结束 ---
        if self.timer_steps == 100:
            end_time = time.time()
            total_time_100 = end_time - self.timer_start
            avg_time_per_step = total_time_100 / 100

            # 计算 200k 次的预估时间
            est_200k_sec = avg_time_per_step * 200000

            print("\n" + "=" * 40)
            print("计时报告 (基于前 100 次训练):")
            print(f"平均单步训练耗时: {avg_time_per_step * 1000:.3f} ms")
            print("预估训练 200k 次所需时间:")
            print(f"  - 分钟: {est_200k_sec / 60:.2f} min")
            print(f"  - 小时: {est_200k_sec / 3600:.2f} h")
            print("=" * 40 + "\n")

    def save(self, timestep):
        torch.save(self.actor.state_dict(), f"./model/sacd_actor_{timestep}.pth")
        torch.save(self.q_critic.state_dict(), f"./model/sacd_critic_{timestep}.pth")

    def load(self, timestep):
        self.actor.load_state_dict(
            torch.load(f"./model/sacd_actor_{timestep}.pth", map_location=self.dvc)
        )
        self.q_critic.load_state_dict(
            torch.load(f"./model/sacd_critic_{timestep}.pth", map_location=self.dvc)
        )
