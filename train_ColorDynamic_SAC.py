import argparse

# import gymnasium as gym
import os

import torch

from Sparrow_V2 import Sparrow, str2bool
from utils.SAC import SAC_agent

# fmt: off
'''Hyperparameter Setting for DRL'''
parser = argparse.ArgumentParser()
# parser.add_argument('--EnvIdex', type=int, default=0, help='CP-v1, LLd-v2')
parser.add_argument('--write', type=str2bool, default=False, help='Use SummaryWriter to record the training')
parser.add_argument('--render', type=str2bool, default=False, help='Render or Not')
parser.add_argument('--Loadmodel', type=str2bool, default=False, help='Load pretrained model or Not')
parser.add_argument('--ModelIdex', type=int, default=2000, help='which model to load')

parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--Max_train_steps', type=int, default=1e8, help='Max training steps')
parser.add_argument('--save_interval', type=int, default=1, help='Model saving interval, in steps.')
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
parser.add_argument('--action_type', type=str, default='Continuous', help='Action type: Discrete / Continuous')
parser.add_argument('--window_size', type=int, default=800, help='size of the training map')
parser.add_argument('--D', type=int, default=400, help='maximal local planning distance')
parser.add_argument('--N', type=int, default=32, help='number of vectorized environments')
parser.add_argument('--O', type=int, default=15, help='number of obstacles in each environment')
parser.add_argument('--RdON', type=str2bool, default=False, help='whether to randomize the Number of dynamic obstacles')
parser.add_argument('--ScOV', type=str2bool, default=False, help='whether to scale the maximal velocity of dynamic obstacles')
parser.add_argument('--RdOV', type=str2bool, default=False, help='whether to randomize the Velocity of dynamic obstacles')
parser.add_argument('--RdOT', type=str2bool, default=True, help='whether to randomize the Type of dynamic obstacles')
parser.add_argument('--RdOR', type=str2bool, default=True, help='whether to randomize the Radius of obstacles')
parser.add_argument('--Obs_R', type=int, default=14, help='maximal obstacle radius, cm')
parser.add_argument('--Obs_V', type=int, default=50, help='maximal obstacle velocity, cm/s')
parser.add_argument('--MapObs', type=str, default=None, help="name of map file, e.g. 'map.png' or None")
parser.add_argument('--ld_a_range', type=int, default=360, help='max scanning angle of lidar (degree)')
parser.add_argument('--ld_d_range', type=int, default=300, help='max scanning distance of lidar (cm)')
parser.add_argument('--ld_num', type=int, default=72, help='number of lidar streams in each world')
parser.add_argument('--ld_GN', type=int, default=3, help='how many lidar streams are grouped for each group')
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
parser.add_argument('--compile', type=str2bool, default=True, help='whether to use torch.compile to boost simulation speed')
opt = parser.parse_args()
opt.render_mode = None # dont render when training
opt.buffersize = min(int(1E6), opt.Max_train_steps)
# opt.reset_freq = int(opt.reset_freq / opt.N)  # Tsteps -> Vsteps

opt.dvc = torch.device(opt.dvc)
# discrete action
# opt.state_dim = 8+int(opt.ld_num/opt.ld_GN)
# opt.action_dim = 7
#continuous action
opt.state_dim = 8 + int(opt.ld_num/opt.ld_GN)
# print("opt.state_dim:", opt.state_dim)
opt.action_dim = 2
# fmt: on
# print(opt)
# Create Env
env = Sparrow(**vars(opt))  # for train
eval_env = Sparrow(**vars(opt))  # for eval
agent = SAC_agent(**vars(opt))


def random_action_test(discrete=True):
    # 采取随机动作测试环境
    if discrete:
        random_action = torch.randint(
            low=0, high=env.action_dim, size=(env.N,), device=env.dvc
        )
        # s_next, r, terminated, truncated, info = env.step(random_action)

    # continuous action
    else:
        N = env.N
        device = env.dvc
        # 生成一个 (N, 2) 的随机张量，值在 [-1, 1] 之间
        # torch.rand 生成 [0, 1]，通过 *2 - 1 映射到 [-1, 1]
        random_action = torch.rand((N, 2), device=device) * 2 - 1

    # 执行环境步进
    s_next, r, terminated, truncated, info = env.step(random_action)
    return random_action


def main():
    # Seed Everything
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False
    print("Random Seed: {}".format(opt.seed))

    # Build model
    if not os.path.exists("model"):
        os.mkdir("model")

    total_steps = 0
    while total_steps < opt.Max_train_steps:
        s, info = env.reset()
        # done = False
        done = torch.zeros(env.N, dtype=torch.bool, device=env.dvc)

        """Interact & trian"""
        while not done.all():
            # e-greedy exploration
            if total_steps < opt.random_steps:
                a = random_action_test(discrete=False)  # continuous action
            else:
                a = agent.select_action(s, deterministic=False)
            s_next, r, dw, tr, info = env.step(a)  # dw: dead&win; tr: truncated
            done = dw | tr

            agent.replay_buffer.add_batch(s, a, r, s_next, dw)
            s = s_next

            """update if its time"""
            # train 50 times every 50 steps rather than 1 training per step. Better!
            if total_steps >= opt.random_steps and total_steps % opt.update_every == 0:
                for j in range(opt.update_every):
                    agent.train()

            """record & log"""
            total_steps += 1

            """save model"""
            if total_steps % opt.save_interval == 0:
                agent.save(int(total_steps / 1000))
    env.close()
    eval_env.close()
    print("Training Finished.")


if __name__ == "__main__":
    main()
