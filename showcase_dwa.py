import argparse

import torch

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
parser.add_argument('--O', type=int, default=0, help='number of obstacles in each environment')
parser.add_argument('--RdON', type=str2bool, default=False, help='whether to randomize the Number of dynamic obstacles')
parser.add_argument('--ScOV', type=str2bool, default=False, help='whether to scale the maximal velocity of dynamic obstacles')
parser.add_argument('--RdOV', type=str2bool, default=True, help='whether to randomize the Velocity of dynamic obstacles')
parser.add_argument('--RdOT', type=str2bool, default=True, help='whether to randomize the Type of dynamic obstacles')
parser.add_argument('--RdOR', type=str2bool, default=True, help='whether to randomize the Radius of obstacles')
parser.add_argument('--Obs_R', type=int, default=0, help='maximal obstacle radius, cm')
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
opt = parser.parse_args()
opt.dvc = torch.device(opt.dvc)
# fmt: on

# Build env
env = Sparrow(**vars(opt))  # for test


def main():
    env.reset()
    s, info = env.reset()
    get_goal_position()
    # Play
    while True:
        a = torch.randint(low=0, high=env.action_dim, size=(env.N,), device=env.dvc)
        s_next, r, terminated, truncated, info = env.step(a)


def get_goal_position():
    target_position = env.target_point[0].cpu().numpy()
    return


if __name__ == "__main__":
    main()
