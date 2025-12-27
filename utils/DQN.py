import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_net(layer_shape, activation, output_activation):
    """Build networks with For loop"""
    layers = []
    for j in range(len(layer_shape) - 1):
        act = activation if j < len(layer_shape) - 2 else output_activation
        layers += [nn.Linear(layer_shape[j], layer_shape[j + 1]), act()]
    return nn.Sequential(*layers)


class Q_Net(nn.Module):
    def __init__(self, state_dim, action_dim, hid_shape):
        super(Q_Net, self).__init__()
        layers = [state_dim] + list(hid_shape) + [action_dim]
        self.Q = build_net(layers, nn.ReLU, nn.Identity)

    def forward(self, s):
        q = self.Q(s)
        return q


class Duel_Q_Net(nn.Module):
    def __init__(self, state_dim, action_dim, hid_shape):
        super(Duel_Q_Net, self).__init__()
        layers = [state_dim] + list(hid_shape)
        self.hidden = build_net(layers, nn.ReLU, nn.ReLU)
        self.V = nn.Linear(hid_shape[-1], 1)
        self.A = nn.Linear(hid_shape[-1], action_dim)
        # build vectorized envs and actor
        # self.envs = Sparrow(**vars(opt))

    def forward(self, s):
        s = self.hidden(s)
        Adv = self.A(s)
        V = self.V(s)
        Q = V + (
            Adv - torch.mean(Adv, dim=-1, keepdim=True)
        )  # Q(s,a)=V(s)+A(s,a)-mean(A(s,a))
        return Q


class DQN_agent(object):
    def __init__(self, **kwargs):
        # Init hyperparameters for agent, just like "self.gamma = opt.gamma, self.lambd = opt.lambd, ..."
        self.__dict__.update(kwargs)
        self.tau = 0.005
        self.replay_buffer = ReplayBuffer(self.state_dim, self.dvc, max_size=int(1e6))
        if self.Duel:
            self.q_net = Duel_Q_Net(
                self.state_dim, self.action_dim, (self.net_width, self.net_width)
            ).to(self.dvc)
        else:
            self.q_net = Q_Net(
                self.state_dim, self.action_dim, (self.net_width, self.net_width)
            ).to(self.dvc)
        self.q_net_optimizer = torch.optim.Adam(self.q_net.parameters(), lr=self.lr)
        self.q_target = copy.deepcopy(self.q_net)
        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.q_target.parameters():
            p.requires_grad = False

    def select_action(self, state, deterministic):
        with torch.no_grad():
            # 确保 state 是 tensor 且在正确的设备上
            state = torch.as_tensor(state, dtype=torch.float32, device=self.dvc)         
            # 神经网络推理
            q_values = self.q_net(state) # 得到 (N, action_dim)
            
            if deterministic:
                a = q_values.argmax(dim=-1) 
            else:
                # 批量 e-greedy 探索
                if np.random.rand() < self.exp_noise:
                    # 生成 (N,) 的随机张量
                    a = torch.randint(low=0, high=self.action_dim, size=(state.shape[0],), device=self.dvc)
                else:
                    a = q_values.argmax(dim=-1)
                    
        return a 

    def train(self):
        s, a, r, s_next, dw = self.replay_buffer.sample(self.batch_size)

        """Compute the target Q value"""
        with torch.no_grad():
            if self.Double:
                argmax_a = self.q_net(s_next).argmax(dim=1).unsqueeze(-1)
                max_q_next = self.q_target(s_next).gather(1, argmax_a)
            else:
                max_q_next = self.q_target(s_next).max(1)[0].unsqueeze(1)
            target_Q = r + (~dw) * self.gamma * max_q_next  # dw: die or win

        # Get current Q estimates
        current_q = self.q_net(s)
        current_q_a = current_q.gather(1, a)

        q_loss = F.mse_loss(current_q_a, target_Q)
        self.q_net_optimizer.zero_grad()
        q_loss.backward()
        self.q_net_optimizer.step()

        # Update the frozen target models
        for param, target_param in zip(
            self.q_net.parameters(), self.q_target.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

    def save(self, steps):
        torch.save(self.q_net.state_dict(), "./model/{}_{}.pth".format("DQN", steps))

    def load(self, steps):
        self.q_net.load_state_dict(
            torch.load(
                "./model/{}_{}.pth".format("DQN", steps),
                map_location=self.dvc,
            )
        )
        self.q_target.load_state_dict(
            torch.load(
                "./model/{}_{}.pth".format("DQN", steps),
                map_location=self.dvc,
            )
        )


# 通用评估函数
def evaluate_policy(env, agent, turns=3):
    # env.N 是并行的小车数量
    total_scores = torch.zeros(env.N, device=env.dvc)
    
    for j in range(turns):
        s, info = env.reset()
        # 记录哪些环境已经结束
        env_dones = torch.zeros(env.N, dtype=torch.bool, device=env.dvc)
        
        # 只要还有没结束的环境，就继续循环
        while not env_dones.all():
            # 这里返回的是 (N,) 的 GPU Tensor
            a = agent.select_action(s, deterministic=True)
            
            # 执行步进
            s_next, r, dw, tr, info = env.step(a)
            
            # r, dw, tr 此时应该是 (N,) 的 Tensor
            # 只累加那些还没结束的环境的分数
            total_scores += r * (~env_dones)
            
            # 更新结束状态
            env_dones = env_dones | dw | tr
            s = s_next
            
    # 返回所有环境、所有轮次的平均分
    return int(total_scores.mean().item() / turns)

# def evaluate(envs, agent, deterministic, turns):
#     step_collector, total_steps = torch.zeros(opt.N, device=opt.dvc), 0
#     r_collector, total_r = torch.zeros(opt.N, device=opt.dvc), 0
#     arrived, finished = 0, 0

#     agent.queue.clear()
#     s, info = envs.reset()
#     ct = torch.ones(opt.N, device=opt.dvc, dtype=torch.bool)
#     while finished < turns:
#         """单步state -> 时序窗口state:"""
#         agent.queue.append(s)  # 将s加入时序窗口队列
#         TW_s = agent.queue.get()  # 取出队列所有数据及
#         a = agent.select_action(TW_s, deterministic)
#         s, r, dw, tr, info = envs.step(a)

#         """解析dones, wins, deads, truncateds, consistents信号："""
#         agent.queue.padding_with_done(~ct)  # 根据上一时刻的ct去padding
#         dones = dw + tr
#         wins = r == envs.AWARD
#         dead_and_tr = dones ^ wins  # dones-wins = deads and truncateds
#         ct = ~dones

#         """统计回合步数："""
#         step_collector += 1
#         total_steps += step_collector[wins].sum()  # 到达,总步数加上真实步数
#         total_steps += (
#             envs.max_ep_steps * dead_and_tr
#         ).sum()  # 未到达,总步数加上回合最大步数
#         step_collector[dones] = 0

#         """统计总奖励："""
#         r_collector += r
#         total_r += r_collector[dones].sum()
#         r_collector[dones] = 0

#         """统计到达率："""
#         finished += dones.sum()
#         arrived += wins.sum()

#     return (
#         int(total_steps.item() / finished.item()),
#         round(total_r.item() / finished.item(), 2),
#         round(arrived.item() / finished.item(), 2),
#     )


class ReplayBuffer(object):
    def __init__(self, state_dim, dvc, max_size=int(1e6)):
        self.max_size = max_size
        self.dvc = dvc
        self.ptr = 0
        self.size = 0

        self.s = torch.zeros((max_size, state_dim), dtype=torch.float, device=self.dvc)
        self.a = torch.zeros((max_size, 1), dtype=torch.long, device=self.dvc)
        self.r = torch.zeros((max_size, 1), dtype=torch.float, device=self.dvc)
        self.s_next = torch.zeros(
            (max_size, state_dim), dtype=torch.float, device=self.dvc
        )
        self.dw = torch.zeros((max_size, 1), dtype=torch.bool, device=self.dvc)

    def add_batch(self, s, a, r, s_next, dw):
        """
        批量添加数据
        s: (batch_size, state_dim)
        a, r, dw: (batch_size, ) 或 (batch_size, 1)
        """
        n = s.shape[0]  # 获取这次进来的数据量

        # 确保数据格式是 Tensor 且在正确的设备上
        if not torch.is_tensor(s):
            s = torch.from_numpy(s).to(self.dvc)
        if not torch.is_tensor(s_next):
            s_next = torch.from_numpy(s_next).to(self.dvc)

        # 处理索引越界（环形缓冲逻辑）
        idx = torch.arange(self.ptr, self.ptr + n) % self.max_size

        self.s[idx] = s.float()
        self.a[idx] = torch.as_tensor(a, device=self.dvc).view(-1, 1).long()
        self.r[idx] = torch.as_tensor(r, device=self.dvc).view(-1, 1).float()
        self.s_next[idx] = s_next.float()
        self.dw[idx] = torch.as_tensor(dw, device=self.dvc).view(-1, 1).bool()

        self.ptr = (self.ptr + n) % self.max_size
        self.size = min(self.size + n, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)
        return (self.s[ind], self.a[ind], self.r[ind], self.s_next[ind], self.dw[ind])
