import argparse

import torch

from Sparrow_V2 import Sparrow, str2bool
from utils.DQN import DQN_agent, evaluate_policy

# fmt: off
parser = argparse.ArgumentParser()
'''Hyperparameter Setting for DQN'''
parser.add_argument('--Env_dvc', type=str, default='cuda:0', help='running device for Sparrow Env')
parser.add_argument('--EnvIdex', type=int, default=0, help='CP-v1, LLd-v2')
parser.add_argument('--write', type=str2bool, default=True, help='Use SummaryWriter to record the training')
parser.add_argument('--render', type=str2bool, default=False, help='Render or Not')
parser.add_argument('--Loadmodel', type=str2bool, default=False, help='Load pretrained model or Not')

parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--Max_train_steps', type=int, default=int(1e6), help='Max training steps')
parser.add_argument('--save_interval', type=int, default=int(50e3), help='Model saving interval, in steps.')
parser.add_argument('--eval_interval', type=int, default=int(2e3), help='Model evaluating interval, in steps.')
parser.add_argument('--random_steps', type=int, default=int(3e3), help='steps for random policy to explore')
parser.add_argument('--update_every', type=int, default=50, help='training frequency')

parser.add_argument('--gamma', type=float, default=0.99, help='Discounted Factor')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--batch_size', type=int, default=256, help='lenth of sliced trajectory')
parser.add_argument('--exp_noise', type=float, default=0.2, help='explore noise')
parser.add_argument('--noise_decay', type=float, default=0.99, help='decay rate of explore noise')
parser.add_argument('--Double', type=str2bool, default=True, help='Whether to use Double Q-learning')
parser.add_argument('--Duel', type=str2bool, default=True, help='Whether to use Duel networks')

parser.add_argument('--ModelIdex', type=int, default=1000, help='which model(e.g. DQN_1000.pth) to load')
parser.add_argument('--net_width', type=int, default=200, help='Linear net width')

'''Hyperparameter Setting for Sparrow'''
parser.add_argument('--dvc', type=str, default='cuda', help='running device of Sparrow: cuda / cpu')
parser.add_argument('--action_type', type=str, default='Discrete', help='Action type: Discrete / Continuous')
parser.add_argument('--window_size', type=int, default=800, help='size of the map')
parser.add_argument('--D', type=int, default=400, help='maximal local planning distance:366*1.414')
parser.add_argument('--N', type=int, default=1, help='number of vectorized environments')
parser.add_argument('--O', type=int, default=15, help='number of obstacles in each environment')
parser.add_argument('--RdON', type=str2bool, default=False, help='whether to randomize the Number of dynamic obstacles')
parser.add_argument('--ScOV', type=str2bool, default=False, help='whether to scale the maximal velocity of dynamic obstacles')
parser.add_argument('--RdOV', type=str2bool, default=True, help='whether to randomize the Velocity of dynamic obstacles')
parser.add_argument('--RdOT', type=str2bool, default=True, help='whether to randomize the Type of dynamic obstacles')
parser.add_argument('--RdOR', type=str2bool, default=True, help='whether to randomize the Radius of obstacles')
parser.add_argument('--Obs_R', type=int, default=14, help='maximal obstacle radius, cm')
parser.add_argument('--Obs_V', type=int, default=30, help='maximal obstacle velocity, cm/s')
parser.add_argument('--MapObs', type=str, default=None, help="name of map file, e.g. 'map.png' or None")
parser.add_argument('--ld_a_range', type=int, default=360, help='max scanning angle of lidar (degree)')
parser.add_argument('--ld_d_range', type=int, default=300, help='max scanning distance of lidar (cm)')
parser.add_argument('--ld_num', type=int, default=72, help='number of lidar streams in each world')
parser.add_argument('--ld_GN', type=int, default=3, help='how many lidar streams are grouped for one group')
parser.add_argument('--ri', type=int, default=0, help='render index: the index of world that be rendered')
parser.add_argument('--basic_ctrl_interval', type=float, default=0.1, help='control interval (s), 0.1 means 10 Hz control frequency')
parser.add_argument('--ctrl_delay', type=int, default=0, help='control delay, in basic_ctrl_interval, 0 means no control delay')
parser.add_argument('--K', type=tuple, default=(0.55,0.6), help='K_linear, K_angular')
parser.add_argument('--draw_auxiliary', type=str2bool, default=False, help='whether to draw auxiliary infos')
parser.add_argument('--render_speed', type=str, default='fast', help='fast / slow / real')
parser.add_argument('--max_ep_steps', type=int, default=500, help='maximum episodic steps')
parser.add_argument('--noise', type=str2bool, default=True, help='whether to add noise to the observations')
parser.add_argument('--DR', type=str2bool, default=True, help='whether to use Domain Randomization')
parser.add_argument('--DR_freq', type=int, default=int(3.2e3), help='frequency of Domain Randomization, in total steps')
parser.add_argument('--compile', type=str2bool, default=False, help='whether to use torch.compile to boost simulation speed')
opt = parser.parse_args()
opt.render_mode = 'human'
opt.dvc = torch.device(opt.dvc)


# fmt: on
def main():
    # Build env
    env = Sparrow(**vars(opt))  # for test
    opt.state_dim = env.state_dim
    opt.action_dim = env.action_dim

    # Init agent
    agent = DQN_agent(**vars(opt))
    agent.load(opt.ModelIdex)

    # Play
    while True:
        scores = evaluate_policy(env, agent, turns=100)
        print(f"ArrivalRate:{scores[2]}, Reward:{scores[1]}, Steps: {scores[0]}\n")


# def evaluate(envs, agent, deterministic, turns):
#     step_collector, total_steps = torch.zeros(opt.N, device=opt.dvc), 0
#     r_collector, total_r = torch.zeros(opt.N, device=opt.dvc), 0
#     arrived, finished = 0, 0

#     # agent.queue.clear()
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


if __name__ == "__main__":
    main()
