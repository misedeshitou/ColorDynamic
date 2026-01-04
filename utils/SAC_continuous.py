import copy
import os

import numpy as np
import torch
import torch.nn.functional as F

from utils.utils_SAC_continuous import Actor, Double_Q_Critic


class SAC_continuous:
    def __init__(self, **kwargs):
        # Init hyperparameters for agent, just like "self.gamma = opt.gamma, self.lambd = opt.lambd, ..."
        self.__dict__.update(kwargs)
        self.tau = 0.005

        self.actor = Actor(
            self.state_dim, self.action_dim, (self.net_width, self.net_width)
        ).to(self.dvc)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.a_lr)

        self.q_critic = Double_Q_Critic(
            self.state_dim, self.action_dim, (self.net_width, self.net_width)
        ).to(self.dvc)
        self.q_critic_optimizer = torch.optim.Adam(
            self.q_critic.parameters(), lr=self.c_lr
        )
        self.q_critic_target = copy.deepcopy(self.q_critic)
        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.q_critic_target.parameters():
            p.requires_grad = False

        self.replay_buffer = ReplayBuffer(
            self.state_dim, self.action_dim, max_size=int(1e6), dvc=self.dvc
        )

        if self.adaptive_alpha:
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            self.target_entropy = torch.tensor(
                -self.action_dim, dtype=float, requires_grad=True, device=self.dvc
            )
            # We learn log_alpha instead of alpha to ensure alpha>0
            self.log_alpha = torch.tensor(
                np.log(self.alpha), dtype=float, requires_grad=True, device=self.dvc
            )
            self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=self.c_lr)
            
    # def select_action(self, state, deterministic):
    #     with torch.no_grad():
    #         if state.ndim == 1:
    #             state = torch.as_tensor(state[np.newaxis, :], dtype=torch.float32, device=self.dvc)
    #         else:
    #             state = torch.as_tensor(state, dtype=torch.float32, device=self.dvc)

    #         a, _ = self.actor(state, deterministic, with_logprob=False)

    #     # --- 修复处：显式转到 CPU 并转为 Numpy ---
    #     # 环境（Sparrow）通常不处理 GPU Tensor，所以必须在这里“落地”
    #     return a.detach().cpu().numpy()
    def select_action(self, state, deterministic):
        with torch.no_grad():
            # 1. 确保输入状态在正确的设备上
            if not torch.is_tensor(state):
                state = torch.as_tensor(state, dtype=torch.float32, device=self.dvc)
            
            if state.ndim == 1:
                state = state.unsqueeze(0)
            else:
                state = state.to(self.dvc).float()

            # 2. 网络推理得到动作
            a, _ = self.actor(state, deterministic, with_logprob=False)

        # --- 关键修复：落地到 CPU 并转为 Numpy ---
        # 只有转回 Numpy 才能兼容环境内的物理模型计算
        return a

    def train(
        self,
    ):
        s, a, r, s_next, dw = self.replay_buffer.sample(self.batch_size)

        # ----------------------------- ↓↓↓↓↓ Update Q Net ↓↓↓↓↓ ------------------------------#
        with torch.no_grad():
            a_next, log_pi_a_next = self.actor(
                s_next, deterministic=False, with_logprob=True
            )
            target_Q1, target_Q2 = self.q_critic_target(s_next, a_next)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = r + (~dw) * self.gamma * (
                target_Q - self.alpha * log_pi_a_next
            )  # Dead or Done is tackled by Randombuffer

        # Get current Q estimates
        current_Q1, current_Q2 = self.q_critic(s, a)

        q_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        self.q_critic_optimizer.zero_grad()
        q_loss.backward()
        self.q_critic_optimizer.step()

        # ----------------------------- ↓↓↓↓↓ Update Actor Net ↓↓↓↓↓ ------------------------------#
        # Freeze critic so you don't waste computational effort computing gradients for them when update actor
        for params in self.q_critic.parameters():
            params.requires_grad = False

        a, log_pi_a = self.actor(s, deterministic=False, with_logprob=True)
        current_Q1, current_Q2 = self.q_critic(s, a)
        Q = torch.min(current_Q1, current_Q2)

        a_loss = (self.alpha * log_pi_a - Q).mean()
        self.actor_optimizer.zero_grad()
        a_loss.backward()
        self.actor_optimizer.step()

        for params in self.q_critic.parameters():
            params.requires_grad = True

        # ----------------------------- ↓↓↓↓↓ Update alpha ↓↓↓↓↓ ------------------------------#
        if self.adaptive_alpha:
            # We learn log_alpha instead of alpha to ensure alpha>0
            alpha_loss = -(
                self.log_alpha * (log_pi_a + self.target_entropy).detach()
            ).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self.alpha = self.log_alpha.exp()

        # ----------------------------- ↓↓↓↓↓ Update Target Net ↓↓↓↓↓ ------------------------------#
        for param, target_param in zip(
            self.q_critic.parameters(), self.q_critic_target.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

    def save(self, timestep):
        path = "./model"
        if not os.path.exists(path):
            os.makedirs(path)
        # 建议增加项目前缀，防止被其他实验覆盖
        torch.save(self.actor.state_dict(), f"{path}/sacc_actor_{timestep}.pth")
        torch.save(self.q_critic.state_dict(), f"{path}/sacc_critic_{timestep}.pth")
        print(f"Model saved at timestep {timestep}")

    def load(self, timestep):
        path = "./model"
        try:
            self.actor.load_state_dict(
                torch.load(f"{path}/sacc_actor_{timestep}.pth", map_location=self.dvc)
            )
            self.q_critic.load_state_dict(
                torch.load(f"{path}/sacc_critic_{timestep}.pth", map_location=self.dvc)
            )
            self.q_critic_target = copy.deepcopy(self.q_critic)  # 记得同步 target 网络
            print(f"Model loaded from timestep {timestep}")
        except FileNotFoundError:
            print(f"Error: Checkpoint at timestep {timestep} not found.")


class ReplayBuffer:
    def __init__(self, state_dim, action_dim, max_size, dvc):
        self.max_size = max_size
        self.dvc = dvc
        self.ptr = 0
        self.size = 0

        # 状态维度 (max_size, state_dim)
        self.s = torch.zeros((max_size, state_dim), dtype=torch.float, device=self.dvc)
        # 连续动作维度 (max_size, action_dim)，注意这里是 float
        self.a = torch.zeros((max_size, action_dim), dtype=torch.float, device=self.dvc)
        # 奖励维度 (max_size, 1)
        self.r = torch.zeros((max_size, 1), dtype=torch.float, device=self.dvc)
        # 下一状态维度 (max_size, state_dim)
        self.s_next = torch.zeros(
            (max_size, state_dim), dtype=torch.float, device=self.dvc
        )
        # 掩码维度 (max_size, 1)
        self.dw = torch.zeros((max_size, 1), dtype=torch.bool, device=self.dvc)

    def add_batch(self, s, a, r, s_next, dw):
        """
        批量添加数据 (适配并行环境 N=32)
        s: (N, state_dim)
        a: (N, action_dim)
        r, dw: (N, ) 或 (N, 1)
        """
        n = s.shape[0]  # 获取本次步进的并行数据量 (例如 32)

        # 1. 确保输入是正确的 Tensor 类型且在指定的 dvc 上
        if not torch.is_tensor(s):
            s = torch.as_tensor(s, device=self.dvc)
        if not torch.is_tensor(a):
            a = torch.as_tensor(a, device=self.dvc)
        if not torch.is_tensor(r):
            r = torch.as_tensor(r, device=self.dvc)
        if not torch.is_tensor(s_next):
            s_next = torch.as_tensor(s_next, device=self.dvc)
        if not torch.is_tensor(dw):
            dw = torch.as_tensor(dw, device=self.dvc)

        # 2. 计算环形缓冲区的索引
        # 使用 torch.arange 生成从 ptr 开始的 n 个连续索引，并对 max_size 取模
        idx = torch.arange(self.ptr, self.ptr + n, device=self.dvc) % self.max_size

        # 3. 批量存入数据
        self.s[idx] = s.float()
        self.s_next[idx] = s_next.float()

        # 关键：使用 .view(n, -1) 自动适配 action_dim，且保持为 float
        self.a[idx] = a.view(n, -1).float()
        self.r[idx] = r.view(n, -1).float()
        self.dw[idx] = dw.view(n, -1).bool()

        # 4. 更新指针和当前缓冲区大小
        self.ptr = (self.ptr + n) % self.max_size
        self.size = min(self.size + n, self.max_size)

    def sample(self, batch_size):
        # 随机采样 batch_size 个索引
        ind = torch.randint(0, self.size, device=self.dvc, size=(batch_size,))
        return self.s[ind], self.a[ind], self.r[ind], self.s_next[ind], self.dw[ind]
