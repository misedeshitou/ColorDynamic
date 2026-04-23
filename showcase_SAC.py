import argparse

import torch

from Sparrow_V2 import Sparrow, str2bool
from utils.SAC import SAC_agent

# fmt: off
parser = argparse.ArgumentParser()
'''Hyperparameter Setting for SAC'''
parser.add_argument('--write', type=str2bool, default=False, help='Use SummaryWriter to record the training')
parser.add_argument('--render', type=str2bool, default=False, help='Render or Not')
parser.add_argument('--Loadmodel', type=str2bool, default=False, help='Load pretrained model or Not')
parser.add_argument('--ModelIdex', type=int, default=7000, help='which model to load')

parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--Max_train_steps', type=int, default=4e5, help='Max training steps')
parser.add_argument('--save_interval', type=int, default=1e5, help='Model saving interval, in steps.')
parser.add_argument('--eval_interval', type=int, default=2e3, help='Model evaluating interval, in steps.')
parser.add_argument('--random_steps', type=int, default=1e4, help='steps for random policy to explore')
parser.add_argument('--update_every', type=int, default=50, help='training frequency')

parser.add_argument('--gamma', type=float, default=0.99, help='Discounted Factor')
parser.add_argument('--hid_shape', type=list, default=[200,200], help='Hidden net shape')
parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--alpha', type=float, default=0.2, help='init alpha')
parser.add_argument('--adaptive_alpha', type=str2bool, default=True, help='Use adaptive alpha turning')

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
    agent = SAC_agent(**vars(opt))
    agent.load(opt.ModelIdex)

    # Play
    while True:
        test_ep_steps, test_ep_r, test_arrival_rate = evaluate(
            env, agent, deterministic=True, turns=100
        )
        print(
            f"ArrivalRate:{test_arrival_rate}, Reward:{test_ep_r}, Steps: {test_ep_steps}\n"
        )


def evaluate(envs, agent, deterministic, turns):
    step_collector, total_steps = torch.zeros(opt.N, device=opt.dvc), 0
    r_collector, total_r = torch.zeros(opt.N, device=opt.dvc), 0
    arrived, finished = 0, 0

    s, info = envs.reset()
    while finished < turns:
        a = agent.select_action(s, deterministic)
        s, r, dw, tr, info = envs.step(a)

        dones = dw + tr
        wins = r == envs.AWARD
        dead_and_tr = dones ^ wins  # dones-wins = deads and truncateds

        """统计回合步数："""
        step_collector += 1
        total_steps += step_collector[wins].sum()  # 到达,总步数加上真实步数
        total_steps += (
            envs.max_ep_steps * dead_and_tr
        ).sum()  # 未到达,总步数加上回合最大步数
        step_collector[dones] = 0

        """统计总奖励："""
        r_collector += r
        total_r += r_collector[dones].sum()
        r_collector[dones] = 0

        """统计到达率："""
        finished += dones.sum()
        arrived += wins.sum()

    return (
        int(total_steps.item() / finished.item()),
        round(total_r.item() / finished.item(), 2),
        round(arrived.item() / finished.item(), 2),
    )


if __name__ == "__main__":
    main()
