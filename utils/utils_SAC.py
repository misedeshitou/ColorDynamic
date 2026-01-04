import torch
import torch.nn as nn
import torch.nn.functional as F


def build_net(layer_shape, hid_activation, output_activation):
    """build net with for loop"""
    layers = []
    for j in range(len(layer_shape) - 1):
        act = hid_activation if j < len(layer_shape) - 2 else output_activation
        layers += [nn.Linear(layer_shape[j], layer_shape[j + 1]), act()]
    return nn.Sequential(*layers)


class Double_Q_Net(nn.Module):
    def __init__(self, state_dim, action_dim, hid_shape):
        super(Double_Q_Net, self).__init__()
        layers = [state_dim] + list(hid_shape) + [action_dim]

        self.Q1 = build_net(layers, nn.ReLU, nn.Identity)
        self.Q2 = build_net(layers, nn.ReLU, nn.Identity)

    def forward(self, s):
        q1 = self.Q1(s)
        q2 = self.Q2(s)
        return q1, q2


class Policy_Net(nn.Module):
    def __init__(self, state_dim, action_dim, hid_shape):
        super(Policy_Net, self).__init__()
        layers = [state_dim] + list(hid_shape) + [action_dim]
        self.P = build_net(layers, nn.ReLU, nn.Identity)

    def forward(self, s):
        logits = self.P(s)
        probs = F.softmax(logits, dim=1)
        return probs


class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, action_type, dvc, max_size=int(1e6)):
        self.max_size = max_size
        self.action_type = action_type
        self.dvc = dvc
        self.ptr = 0
        self.size = 0

        self.s = torch.zeros((max_size, state_dim), dtype=torch.float, device=self.dvc)
        if action_type == 'Discrete':
            self.a = torch.zeros((max_size, 1), dtype=torch.long, device=self.dvc)
        else:
            self.a = torch.zeros((max_size, action_dim), device=self.dvc)
        self.r = torch.zeros((max_size, 1), dtype=torch.float, device=self.dvc)
        self.s_next = torch.zeros(
            (max_size, state_dim), dtype=torch.float, device=self.dvc
        )
        self.dw = torch.zeros((max_size, 1), dtype=torch.bool, device=self.dvc)

    def add_batch(self, s, a, r, s_next, dw):
        n = s.shape[0] 

        # 确保数据在正确的设备上
        if not torch.is_tensor(s): s = torch.from_numpy(s).to(self.dvc)
        if not torch.is_tensor(s_next): s_next = torch.from_numpy(s_next).to(self.dvc)

        idx = torch.arange(self.ptr, self.ptr + n, device=self.dvc) % self.max_size

        self.s[idx] = s.float()
        self.s_next[idx] = s_next.float()
        
        # --- 核心修改处 ---
        # 1. 使用 view(n, -1) 确保第一维始终匹配 Batch Size
        # 2. 根据动作类型决定是用 long 还是 float
        a_tensor = torch.as_tensor(a, device=self.dvc).view(n, -1)
        if self.action_type == "Discrete":
            self.a[idx] = a_tensor.long()
        else:
            self.a[idx] = a_tensor.float() # 连续动作必须是浮点数
            
        # 奖励和完成标志也建议明确指定 view(n, -1)
        self.r[idx] = torch.as_tensor(r, device=self.dvc).view(n, -1).float()
        self.dw[idx] = torch.as_tensor(dw, device=self.dvc).view(n, -1).bool()
        # ----------------

        self.ptr = (self.ptr + n) % self.max_size
        self.size = min(self.size + n, self.max_size)

    def sample(self, batch_size):
        ind = torch.randint(0, self.size, device=self.dvc, size=(batch_size,))
        return self.s[ind], self.a[ind], self.r[ind], self.s_next[ind], self.dw[ind]


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
