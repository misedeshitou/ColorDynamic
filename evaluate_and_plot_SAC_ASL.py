import argparse
import os
import re
import shutil
from datetime import datetime

import torch
from torch.utils.tensorboard import SummaryWriter

from Sparrow_V2 import Sparrow, str2bool
from utils.SAC import SAC_agent

# fmt: off
parser = argparse.ArgumentParser()

"""Evaluation setting: model_eval_turns = C * N"""
parser.add_argument('--C', type=int, default=10, help='number of reset times')
parser.add_argument('--N', type=int, default=10, help='number of vectorized environments')
parser.add_argument('--deterministic', type=str2bool, default=True, help='whether to use deterministic policy when evaluating')

"""SAC checkpoint setting"""
parser.add_argument('--model_root', type=str, default='model/old_SAC', help='root folder of SAC checkpoints')
parser.add_argument('--run_dir', type=str, default='', help='timestamp subfolder under model_root (empty means latest)')

"""Hyperparameter Setting for SAC (for network build only)"""
parser.add_argument('--gamma', type=float, default=0.99, help='Discounted Factor')
parser.add_argument('--hid_shape', type=list, default=[200, 200], help='Hidden net shape')
parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--alpha', type=float, default=0.2, help='init alpha')
parser.add_argument('--adaptive_alpha', type=str2bool, default=True, help='Use adaptive alpha turning')

"""Hyperparameter Setting for Sparrow"""
parser.add_argument('--dvc', type=str, default='cuda', help='running device of Sparrow: cuda / cpu')
parser.add_argument('--action_type', type=str, default='Discrete', help='Action type: Discrete / Continuous')
parser.add_argument('--window_size', type=int, default=800, help='size of the map')
parser.add_argument('--D', type=int, default=400, help='maximal local planning distance:366*1.414')
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
parser.add_argument('--ld_GN', type=int, default=3, help='how many lidar streams are grouped for one group')
parser.add_argument('--ri', type=int, default=0, help='render index: the index of world that be rendered')
parser.add_argument('--basic_ctrl_interval', type=float, default=0.1, help='control interval (s), 0.1 means 10 Hz control frequency')
parser.add_argument('--ctrl_delay', type=int, default=0, help='control delay, in basic_ctrl_interval, 0 means no control delay')
parser.add_argument('--K', type=tuple, default=(0.55, 0.6), help='K_linear, K_angular')
parser.add_argument('--draw_auxiliary', type=str2bool, default=False, help='whether to draw auxiliary infos')
parser.add_argument('--render_speed', type=str, default='fast', help='fast / slow / real')
parser.add_argument('--max_ep_steps', type=int, default=500, help='maximum episodic steps')
parser.add_argument('--noise', type=str2bool, default=True, help='whether to add noise to the observations')
parser.add_argument('--DR', type=str2bool, default=True, help='whether to use Domain Randomization')
parser.add_argument('--DR_freq', type=int, default=int(3.2e3), help='frequency of Domain Randomization, in total steps')
parser.add_argument('--compile', type=str2bool, default=True, help='whether to use torch.compile to boost simulation speed')
opt = parser.parse_args()
opt.render_mode = None
opt.dvc = torch.device(opt.dvc)
# fmt: on


def resolve_ckpt_dir(model_root, run_dir):
    if run_dir:
        return os.path.join(model_root, run_dir)

    if not os.path.isdir(model_root):
        return model_root

    # if actor checkpoints are directly under model_root, use it directly
    direct_actor = [
        f for f in os.listdir(model_root) if re.match(r"^sacd_actor_\d+\.pth$", f)
    ]
    if direct_actor:
        return model_root

    subdirs = [
        os.path.join(model_root, d)
        for d in os.listdir(model_root)
        if os.path.isdir(os.path.join(model_root, d))
    ]
    if not subdirs:
        return model_root

    subdirs.sort()
    return subdirs[-1]


def list_actor_ckpts(ckpt_dir):
    if not os.path.isdir(ckpt_dir):
        return []

    pairs = []
    for name in os.listdir(ckpt_dir):
        m = re.match(r"^sacd_actor_(\d+)\.pth$", name)
        if m:
            pairs.append((int(m.group(1)), os.path.join(ckpt_dir, name)))
    pairs.sort(key=lambda x: x[0])
    return pairs


def main():
    eval_envs = Sparrow(**vars(opt))
    opt.state_dim = eval_envs.state_dim
    opt.action_dim = eval_envs.action_dim

    agent = SAC_agent(**vars(opt))

    ckpt_dir = resolve_ckpt_dir(opt.model_root, opt.run_dir)
    ckpts = list_actor_ckpts(ckpt_dir)
    if len(ckpts) == 0:
        raise FileNotFoundError(f"No SAC actor checkpoint found under: {ckpt_dir}")

    print(f"Evaluating SAC-ASL checkpoints from: {ckpt_dir}")

    timenow = str(datetime.now())[0:-10]
    timenow = " " + timenow[0:13] + "_" + timenow[-2::]
    writepath = f"runs/SAC_ASL_Eval-C{opt.C}-N{opt.N}-" + timenow
    if os.path.exists(writepath):
        shutil.rmtree(writepath)
    writer = SummaryWriter(log_dir=writepath)

    results = []
    for model_idx, ckpt_path in ckpts:
        agent.actor.load_state_dict(
            torch.load(ckpt_path, map_location=opt.dvc, weights_only=True)
        )

        ep_steps, ep_r, arrival_rate = 0, 0, 0
        for _ in range(opt.C):
            temp_ep_steps, temp_ep_r, temp_arrival_rate = (
                vectorized_model_evaluation_sac(
                    eval_envs, agent, deterministic=opt.deterministic
                )
            )
            ep_steps += temp_ep_steps
            ep_r += temp_ep_r
            arrival_rate += temp_arrival_rate

        ep_steps /= opt.C
        ep_r /= opt.C
        arrival_rate /= opt.C

        writer.add_scalar("ep_steps", ep_steps, global_step=model_idx)
        writer.add_scalar("ep_r", ep_r, global_step=model_idx)
        writer.add_scalar("arrival_rate", arrival_rate, global_step=model_idx)

        normed_ep_steps = round(
            (opt.max_ep_steps - ep_steps) / (opt.max_ep_steps - 70), 3
        )
        normed_ep_r = round(ep_r / 220, 3)
        arrival_rate = round(arrival_rate, 3)
        normed_total_score = round(
            (normed_ep_steps + normed_ep_r + arrival_rate) / 3, 3
        )

        model_name = os.path.basename(ckpt_path)
        results.append(
            (model_name, normed_total_score, arrival_rate, normed_ep_steps, normed_ep_r)
        )

        print(f"model: {model_name}")
        print(
            f"Episodic Steps: {int(ep_steps)}, Episodic Rewards: {int(ep_r)}, Arrival Rate: {arrival_rate}"
        )
        print(
            f"Total Score: {normed_total_score}, Arrival Rate: {arrival_rate}, Step Score: {normed_ep_steps}, Reward Score: {normed_ep_r}"
        )
        print(
            "----------------------------------------------------------------------------------------------"
        )

    write_rank("SAC_ASL_TotalRank.txt", reversed(sorted(results, key=lambda x: x[1])))
    write_rank("SAC_ASL_ArrivalRank.txt", reversed(sorted(results, key=lambda x: x[2])))
    write_rank("SAC_ASL_StepRank.txt", reversed(sorted(results, key=lambda x: x[3])))
    write_rank("SAC_ASL_RewardRank.txt", reversed(sorted(results, key=lambda x: x[4])))

    writer.close()
    eval_envs.close()


def vectorized_model_evaluation_sac(envs, agent, deterministic=True):
    step_collector, total_steps = torch.zeros(opt.N, device=opt.dvc), 0
    r_collector, total_r = torch.zeros(opt.N, device=opt.dvc), 0
    arrived_vec = torch.zeros(opt.N, dtype=torch.bool, device=opt.dvc)
    finished_vec = torch.zeros(opt.N, dtype=torch.bool, device=opt.dvc)
    finished = 0

    s, info = envs.reset()
    while not finished_vec.all():
        a = agent.select_action(s, deterministic)
        s, r, dw, tr, info = envs.step(a)

        dones = dw | tr
        wins = r == envs.AWARD
        dead_and_tr = dones ^ wins

        step_collector += 1
        total_steps += step_collector[wins].sum()
        total_steps += (envs.max_ep_steps * dead_and_tr).sum()
        step_collector[dones] = 0

        r_collector += r
        total_r += r_collector[dones].sum()
        r_collector[dones] = 0

        arrived_vec += (~finished_vec) & wins
        finished_vec += dones
        finished += dones.sum()

    return (
        total_steps.item() / finished.item(),
        total_r.item() / finished.item(),
        arrived_vec.sum().item() / opt.N,
    )


def write_rank(filename, data):
    path = "Evaluation_result"
    if not os.path.exists(path):
        os.mkdir(path)
    with open(path + "/" + filename, "w") as f:
        column_width = 20
        header = ["Index", "Total Score", "Arrival Score", "Step Score", "Reward Score"]
        f.write("|".join(f"{h:<{column_width}}" for h in header) + "\n")
        for item in data:
            line = "|".join(f"{str(element):<{column_width}}" for element in item)
            f.write(line + "\n")


if __name__ == "__main__":
    main()
