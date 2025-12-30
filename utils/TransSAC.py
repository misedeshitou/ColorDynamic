import copy
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

from utils.TWQ import TimeWindowQueue_NTD
from utils.utils_TransSAC import Double_Q_Net, Policy_Net, ReplayBuffer


def orthogonal_init(layer, gain=1.414):
    for name, param in layer.named_parameters():
        if "bias" in name:
            nn.init.constant_(param, 0)
        elif "weight" in name:
            nn.init.orthogonal_(param, gain=gain)
    return layer


# todo: Transquer only for eval/play?
class TransSAC_agent:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.tau = 0.005
        self.H_mean = 0

        self.replay_buffer = ReplayBuffer(
            self.state_dim, self.T, self.dvc, max_size=int(1e6)
        )

        self.actor = Policy_Net(self).to(self.dvc)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)

        self.q_critic = Double_Q_Net(self).to(self.dvc)
        self.q_critic_optimizer = torch.optim.Adam(
            self.q_critic.parameters(), lr=self.lr
        )
        self.q_critic_target = copy.deepcopy(self.q_critic)

        # todo:self.queue
        self.queue = TimeWindowQueue_NTD(
            self.N, self.T, self.state_dim, self.dvc, padding=0
        )

        # 计时变量
        self.timer_steps = 0
        self.timer_start = 0.0

        for p in self.q_critic_target.parameters():
            p.requires_grad = False

        if self.adaptive_alpha:
            # We use 0.6 because the recommended 0.98 will cause alpha explosion.
            self.target_entropy = 0.6 * (-np.log(1 / self.action_dim))  # H(discrete)>0
            self.log_alpha = torch.tensor(
                np.log(self.alpha),
                dtype=torch.float32,
                requires_grad=True,
                device=self.dvc,
            )
            self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=self.lr)

    def select_action(self, state, deterministic):
        """
        输入: state (N, D) - 当前单帧状态
        输出: action (N,) - 批量动作
        """
        state = torch.as_tensor(state, dtype=torch.float32, device=self.dvc)
        # todo: 使用时间窗口队列
        # 1. 将单帧数据推入历史队列
        self.queue.append(state)
        # 2. 获取经过排列的时间窗张量 (N, T, D)
        tw_s = self.queue.get()

        with torch.no_grad():
            # input: (N, T, D)
            probs = self.actor(tw_s)

            if deterministic:
                a = probs.argmax(-1)
            else:
                dist = Categorical(probs)
                a = dist.sample()

            return a

    def train(self):
        if self.timer_steps == 0:
            self.timer_start = time.time()

        # 1. 采样：s 和 s_next 现在的形状都是 (B, T, D)
        s, a, r, s_next, dw = self.replay_buffer.sample(self.batch_size)

        # 维度对齐与设备迁移
        a = a.view(-1, 1).long().to(self.dvc)
        r = r.view(-1, 1).to(self.dvc)
        dw = dw.view(-1, 1).to(self.dvc)
        s = s.to(self.dvc)  # (Batch, T, D)
        s_next = s_next.to(self.dvc)  # (Batch, T, D)

        # ------------------------------------------ Train Critic ----------------------------------------#
        """Compute the target soft Q value"""
        with torch.no_grad():
            next_probs = self.actor(s_next)  # [b, a_dim]
            next_log_probs = torch.log(next_probs + 1e-8)
            next_q1_all, next_q2_all = self.q_critic_target(s_next)
            min_next_q_all = torch.min(next_q1_all, next_q2_all)

            # V(s_next) = \sum \pi * (Q - \alpha * log\pi)
            v_next = torch.sum(
                next_probs * (min_next_q_all - self.alpha * next_log_probs),
                dim=1,
                keepdim=True,
            )
            target_Q = r + (~dw) * self.gamma * v_next

        """Update soft Q net"""
        q1_all, q2_all = self.q_critic(s)  # [b, a_dim]
        q1, q2 = q1_all.gather(1, a), q2_all.gather(1, a)
        q_loss = F.mse_loss(q1, target_Q) + F.mse_loss(q2, target_Q)

        self.q_critic_optimizer.zero_grad()
        q_loss.backward()
        self.q_critic_optimizer.step()

        # ------------------------------------------ Train Actor ----------------------------------------#
        probs = self.actor(s)
        log_probs = torch.log(probs + 1e-8)
        with torch.no_grad():
            q1_all, q2_all = self.q_critic(s)
        min_q_all = torch.min(q1_all, q2_all)

        # Actor Loss = \sum \pi * (\alpha * log\pi - Q)
        a_loss = torch.sum(probs * (self.alpha * log_probs - min_q_all), dim=1)

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

        # 计时报告逻辑
        if self.timer_steps == 100:
            end_time = time.time()
            avg_time = (end_time - self.timer_start) / 100
            est_200k_h = (avg_time * 200000) / 3600

            print("\n" + "=" * 30)
            print("Transformer-SAC 性能报告:")
            print(f"单步耗时: {avg_time * 1000:.2f} ms")
            print(f"预估 200k 步总耗时: {est_200k_h:.2f} 小时")
            print("=" * 30 + "\n")

    def save(self, timestep):
        torch.save(self.actor.state_dict(), f"./model/transsac_actor_{timestep}.pth")
        torch.save(
            self.q_critic.state_dict(), f"./model/transsac_critic_{timestep}.pth"
        )

    def load(self, timestep):
        self.actor.load_state_dict(
            torch.load(f"./model/transsac_actor_{timestep}.pth", map_location=self.dvc)
        )
        self.q_critic.load_state_dict(
            torch.load(f"./model/transsac_critic_{timestep}.pth", map_location=self.dvc)
        )
