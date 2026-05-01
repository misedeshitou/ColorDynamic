import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.PoE import PositionalEncoding_NTD


def build_net(layer_shape, hid_activation, output_activation):
    """build net with for loop"""
    layers = []
    for j in range(len(layer_shape) - 1):
        act = hid_activation if j < len(layer_shape) - 2 else output_activation
        layers += [nn.Linear(layer_shape[j], layer_shape[j + 1]), act()]
    return nn.Sequential(*layers)


class Double_Q_Net(nn.Module):
    # def __init__(self, state_dim, action_dim, hid_shape):
    #     super(Double_Q_Net, self).__init__()
    #     layers = [state_dim] + list(hid_shape) + [action_dim]

    #     self.Q1 = build_net(layers, nn.ReLU, nn.Identity)
    #     self.Q2 = build_net(layers, nn.ReLU, nn.Identity)
    def __init__(self, opt):
        super(Double_Q_Net, self).__init__()
        self.d = opt.state_dim - 8  # Lidar 特征维度

        # 1. PositionalEncoding_NTD
        self.pe = PositionalEncoding_NTD(maxlen=opt.T, emb_size=self.d)

        # 2. Transformer Encoder (Q1 和 Q2 共享结构但不共享参数)
        # todo: encoder_layer; num_layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d, nhead=opt.H, batch_first=True, dim_feedforward=opt.net_width
        )
        self.trans_q1_encoder = nn.TransformerEncoder(encoder_layer, num_layers=opt.L)
        self.trans_q2_encoder = nn.TransformerEncoder(encoder_layer, num_layers=opt.L)

        # 3. 输入维度 = Transformer 聚合雷达特征 (self.d) + 当前状态完整 (state_dim)
        # todo：orthogonal_init
        in_dim = self.d + opt.state_dim
        layers = [in_dim] + list(opt.hid_shape) + [opt.action_dim]

        self.Q1 = build_net(layers, nn.ReLU, nn.Identity)
        self.Q2 = build_net(layers, nn.ReLU, nn.Identity)

    # def forward(self, s):
    #     q1 = self.Q1(s)
    #     q2 = self.Q2(s)
    #     return q1, q2
    def forward(self, TW_s):
        """TW_s.shape = (B,T,D)"""
        temporal_ld = TW_s[:, :, 8:]  # s[0:7] is robot state, s[8:] is lidar results
        temporal_ld = self.pe(temporal_ld)  # (N,T,d)

        temporal_ld_feat1 = self.trans_q1_encoder(temporal_ld)  # (N,T,d)
        # 指定维度求平均
        temporal_ld_feat1 = temporal_ld_feat1.mean(dim=1)  # (N,T,d) ->  (N,d)

        temporal_ld_feat2 = self.trans_q2_encoder(temporal_ld)  # (N,T,d)
        temporal_ld_feat2 = temporal_ld_feat2.mean(dim=1)  # (N,T,d) ->  (N,d)

        # Q1 分支
        aug_s1 = torch.cat((temporal_ld_feat1, TW_s[:, 0, :]), dim=-1)  # (N,d+S_dim)
        q1 = self.Q1(aug_s1)

        # Q2 分支
        aug_s2 = torch.cat((temporal_ld_feat2, TW_s[:, 0, :]), dim=-1)  # (N,d+S_dim)
        q2 = self.Q2(aug_s2)

        return q1, q2


class Policy_Net(nn.Module):
    # def __init__(self, state_dim, action_dim, hid_shape):
    #     super(Policy_Net, self).__init__()
    #     layers = [state_dim] + list(hid_shape) + [action_dim]
    #     self.P = build_net(layers, nn.ReLU, nn.Identity)

    def __init__(self, opt):
        super(Policy_Net, self).__init__()
        self.d = opt.state_dim - 8

        # 1. PositionalEncoding_NTD
        # todo: self.d
        self.pe = PositionalEncoding_NTD(maxlen=opt.T, emb_size=self.d)
        # 2. Transformer Encoder
        # todo: dropout
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d,
            nhead=opt.H,
            dropout=0,  # 每个神经元始终处于激活状态
            dim_feedforward=opt.net_width,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=opt.L
        )

        # 3. 输出头
        in_dim = self.d + opt.state_dim
        layers = [in_dim] + list(opt.hid_shape) + [opt.action_dim]
        self.P = build_net(layers, nn.ReLU, nn.Identity)

    def forward(self, TW_s):
        # TW_s shape: (B, T, state_dim)
        temporal_ld = TW_s[:, :, 8:]
        temporal_ld = self.pe(temporal_ld)
        temporal_ld = self.transformer_encoder(temporal_ld)
        temporal_ld = temporal_ld.mean(dim=1)  # 聚合时序特征

        # 拼接当前帧 (t=0时刻)
        aug_s = torch.cat((temporal_ld, TW_s[:, 0, :]), dim=-1)

        logits = self.P(aug_s)
        return F.softmax(logits, dim=-1)  # 返回动作概率分布


# todo: replay buffer for TransSAC
# return-> (B,T,D)
class ReplayBuffer(object):
    def __init__(self, state_dim, T, dvc, max_size=int(1e6)):
        self.max_size = max_size
        self.dvc = dvc
        self.T = T
        self.ptr = 0
        self.size = 0

        # 存储原始数据，采样时动态合成序列
        self.s = torch.zeros((max_size, state_dim), device=self.dvc)
        self.a = torch.zeros((max_size, 1), dtype=torch.long, device=self.dvc)
        self.r = torch.zeros((max_size, 1), device=self.dvc)
        self.dw = torch.zeros((max_size, 1), dtype=torch.bool, device=self.dvc)

    def add_batch(self, s, a, r, s_next, dw):
        n = s.shape[0]
        idx = torch.arange(self.ptr, self.ptr + n, device=self.dvc) % self.max_size

        self.s[idx] = torch.as_tensor(s, device=self.dvc).float()
        self.a[idx] = torch.as_tensor(a, device=self.dvc).view(-1, 1).long()
        self.r[idx] = torch.as_tensor(r, device=self.dvc).view(-1, 1).float()
        self.dw[idx] = torch.as_tensor(dw, device=self.dvc).view(-1, 1).bool()

        self.ptr = (self.ptr + n) % self.max_size
        self.size = min(self.size + n, self.max_size)

    def sample(self, batch_size):
        # 随机采样 batch_size 个索引作为结束点
        # 确保 ind-T+1 > 0
        ind = torch.randint(self.T, self.size, size=(batch_size,), device=self.dvc)

        # 向量化构建时间窗，避免 Python 循环带来的 CPU 开销
        # [t, t-1, ..., t-T+1]
        offsets = torch.arange(self.T, device=self.dvc)
        tw_idx = ind.unsqueeze(1) - offsets.unsqueeze(0)
        tw_s = self.s[tw_idx]

        # [t+1, t, ..., t-T+2]
        next_ind = (ind + 1) % self.size
        tw_next_idx = next_ind.unsqueeze(1) - offsets.unsqueeze(0)
        tw_s_next = self.s[tw_next_idx]

        return tw_s, self.a[ind], self.r[ind], tw_s_next, self.dw[ind]


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
