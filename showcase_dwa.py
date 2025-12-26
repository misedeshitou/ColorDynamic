import argparse

import torch

from DWA.dwa_controller import DWAController
from Sparrow_V2 import Sparrow, str2bool

# fmt: off
'''Hyperparameter Setting for Sparrow'''
parser = argparse.ArgumentParser()
parser.add_argument('--dvc', type=str, default='cpu', help='running device of Sparrow: cuda / cpu')
parser.add_argument('--render_mode', type=str, default='human', help=" None or 'human' ")
parser.add_argument('--action_type', type=str, default='Continuous', help='Action type: Discrete / Continuous')
parser.add_argument('--window_size', type=int, default=800, help='size of the map')
parser.add_argument('--D', type=int, default=400, help='maximal local planning distance')
parser.add_argument('--N', type=int, default=1, help='number of vectorized environments')
parser.add_argument('--O', type=int, default=0, help='number of dynamics obstacles in each environment')
parser.add_argument('--RdON', type=str2bool, default=False, help='whether to randomize the Number of dynamic obstacles')
parser.add_argument('--ScOV', type=str2bool, default=False, help='whether to scale the maximal velocity of dynamic obstacles')
parser.add_argument('--RdOV', type=str2bool, default=True, help='whether to randomize the Velocity of dynamic obstacles')
parser.add_argument('--RdOT', type=str2bool, default=True, help='whether to randomize the Type of dynamic obstacles')
parser.add_argument('--RdOR', type=str2bool, default=True, help='whether to randomize the Radius of obstacles')
parser.add_argument('--Obs_R', type=int, default=20, help='maximal obstacle radius, cm')
parser.add_argument('--Obs_V', type=int, default=30, help='maximal obstacle velocity, cm/s')
parser.add_argument('--MapObs', type=str, default=None, help="name of map file, e.g. 'map.png' or None")
parser.add_argument('--ld_a_range', type=int, default=270, help='max scanning angle of lidar (degree)')
parser.add_argument('--ld_d_range', type=int, default=300, help='max scanning distance of lidar (cm)')
parser.add_argument('--ld_num', type=int, default=27, help='number of lidar streams in each world')
parser.add_argument('--ld_GN', type=int, default=3, help='how many lidar streams are grouped for one group')
parser.add_argument('--ri', type=int, default=0, help='render index: the index of world that be rendered')
parser.add_argument('--basic_ctrl_interval', type=float, default=0.1, help='control interval (s), 0.1 means 10 Hz control frequency')
parser.add_argument('--ctrl_delay', type=int, default=0, help='control delay, in basic_ctrl_interval, 0 means no control delay')
parser.add_argument('--K', type=tuple, default=(0.55,0.6), help='K_linear, K_angular')
parser.add_argument('--draw_auxiliary', type=str2bool, default=False, help='whether to draw auxiliary infos')
parser.add_argument('--render_speed', type=str, default='fast', help='fast / slow / real')
parser.add_argument('--max_ep_steps', type=int, default=500, help='maximum episodic steps')
parser.add_argument('--noise', type=str2bool, default=False, help='whether to add noise to the observations')
parser.add_argument('--DR', type=str2bool, default=True, help='whether to use Domain Randomization')
parser.add_argument('--DR_freq', type=int, default=int(3.2e3), help='frequency of Domain Randomization, in total steps')
parser.add_argument('--compile', type=str2bool, default=False, help='whether to use torch.compile to boost simulation speed')
parser.add_argument('--planner', type=bool, default=False, help='whether to use rl_planner')
parser.add_argument('--random',type=bool, default=False, help='whether to use random action test')
opt = parser.parse_args()
opt.dvc = torch.device(opt.dvc)
# fmt: on

# Build env
env = Sparrow(**vars(opt))  # for test
controller = DWAController()


def main():
    env.reset()
    s, info = env.reset()
    # Play
    while True:
        if opt.random:
            random_action_test()
        else:
            dwa_control(env, controller, env_idx=0)
    return


def dwa_control(env, controller, env_idx=0):
    """
    使用向量化 DWA 控制器的新调用方式
    env_idx: 指定当前处理的是第几个并行环境 (0 ~ N-1)
    """
    # 1. 获取目标点并转换为 Tensor (需与 controller 的 device 一致)
    # 假设 target_point 形状为 (N, 2)
    goal_tensor = env.target_point[env_idx].to(controller.dvc).float()

    # 2. 获取当前车辆状态 (N, 5) -> 取出第 env_idx 个
    # 状态格式: [x, y, yaw, v, w]
    x = env.car_state[env_idx].cpu().numpy()

    # 3. 调用向量化 Plan
    # self.vec_dynamic_obs_P_shaped(N, O * P, 2, 1)
    u, best_traj = controller.plan(
        x, goal_tensor, env.vec_static_obs_P_shaped, env_idx=env_idx
    )
    #    u, best_traj = controller.plan(
    #     x, goal_tensor, env.vec_dynamic_obs_P_shaped, env_idx=env_idx
    # )

    # 4. 动作归一化 (DWA 输出的是物理速度，环境通常需要 -1 ~ 1)
    v_norm = u[0] / env.v_linear_max
    w_norm = u[1] / env.v_angular_max

    # 5. 组装 Action Tensor (N, 2)
    batch_action = torch.zeros((env.N, 2), device=env.dvc, dtype=torch.float32)
    batch_action[env_idx] = torch.tensor([v_norm, w_norm], device=env.dvc)

    env.step(batch_action)

    return u, best_traj


def random_action_test():
    # 采取随机动作测试环境
    # discrete action
    # a = torch.randint(low=0, high=env.action_dim, size=(env.N,), device=env.dvc)
    # print("random control", a.numpy())
    # s_next, r, terminated, truncated, info = env.step(a)

    # continuous action
    # env.N 是小车的数量，env.dvc 是设备(cpu或cuda)
    N = env.N
    device = env.dvc
    # 生成一个 (N, 2) 的随机张量，值在 [-1, 1] 之间
    # torch.rand 生成 [0, 1]，通过 *2 - 1 映射到 [-1, 1]
    random_action = torch.rand((N, 2), device=device) * 2 - 1

    print("random continuous control:\n", random_action.cpu().numpy())

    # 执行环境步进
    env.step(random_action)
    return


if __name__ == "__main__":
    main()
