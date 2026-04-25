import argparse

# import gymnasium as gym
import os
import time
from datetime import datetime

import torch
from torch.utils.tensorboard import SummaryWriter

from Sparrow_V2 import Sparrow, str2bool
from utils.SAC import SAC_agent

# fmt: off
'''Hyperparameter Setting for DRL'''
parser = argparse.ArgumentParser()
# parser.add_argument('--EnvIdex', type=int, default=0, help='CP-v1, LLd-v2')
parser.add_argument('--write', type=str2bool, default=True, help='Use SummaryWriter to record the training')
parser.add_argument('--render', type=str2bool, default=False, help='Render or Not')
parser.add_argument('--Loadmodel', type=str2bool, default=False, help='Load pretrained model or Not')
parser.add_argument('--ModelIdex', type=int, default=2000, help='which model to load')

parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--max_train_steps', '--Max_train_steps', dest='max_train_steps', type=int, default=1e8, help='Max training steps')
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
opt.run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
opt.run_dir = os.path.join('runs', 'SAC', opt.run_timestamp)
opt.model_dir = os.path.join('model', 'SAC', opt.run_timestamp)
os.makedirs(opt.run_dir, exist_ok=True)
os.makedirs(opt.model_dir, exist_ok=True)
opt.render_mode = None # dont render when training
opt.buffersize = min(int(1E6), opt.max_train_steps)
# opt.reset_freq = int(opt.reset_freq / opt.N)  # Tsteps -> Vsteps

opt.dvc = torch.device(opt.dvc)
opt.state_dim = 8+int(opt.ld_num/opt.ld_GN)
opt.action_dim = 7
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


def evaluate(envs, agent, deterministic=True, turns=20):
    step_collector, total_steps = torch.zeros(envs.N, device=envs.dvc), 0
    r_collector, total_r = torch.zeros(envs.N, device=envs.dvc), 0
    arrived, finished = 0, 0

    s, info = envs.reset()
    while finished < turns:
        a = agent.select_action(s, deterministic)
        s, r, dw, tr, info = envs.step(a)

        dones = dw + tr
        wins = r == envs.AWARD
        dead_and_tr = dones ^ wins

        step_collector += 1
        total_steps += step_collector[wins].sum()
        total_steps += (envs.max_ep_steps * dead_and_tr).sum()
        step_collector[dones] = 0

        r_collector += r
        total_r += r_collector[dones].sum()
        r_collector[dones] = 0

        finished += int(dones.sum().item())
        arrived += int(wins.sum().item())

    return (
        int(total_steps.item() / finished),
        round(total_r.item() / finished, 2),
        round(arrived / finished, 2),
    )


def main():
    # Seed Everything
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print("Random Seed: {}".format(opt.seed))

    # Build model
    writer = None
    if opt.write:
        writer = SummaryWriter(log_dir=opt.run_dir)
        writer.add_text("config", str(vars(opt)))
        sample_state = torch.zeros((2, opt.state_dim), device=opt.dvc)
        try:
            writer.add_graph(agent.actor, sample_state)
        except Exception as e:
            print(f"[TensorBoard] add_graph(actor) failed: {e}")
        try:
            writer.add_graph(agent.q_critic, sample_state)
        except Exception as e:
            print(f"[TensorBoard] add_graph(q_critic) failed: {e}")

    print(f"[SAC] logs -> {opt.run_dir}")
    print(f"[SAC] models -> {opt.model_dir}")

    total_steps = 0
    bstep = 0
    perf_start_after_warmup = None
    perf_last_print_steps = -1
    while total_steps < opt.max_train_steps:
        s, info = env.reset()
        # done = False
        done = torch.zeros(env.N, dtype=torch.bool, device=env.dvc)

        """Interact & trian"""
        while not done.all():
            # e-greedy exploration
            if total_steps < opt.random_steps:
                a = random_action_test()
            else:
                a = agent.select_action(s, deterministic=False)
            s_next, r, dw, tr, info = env.step(a)  # dw: dead&win; tr: truncated
            done = dw | tr

            agent.replay_buffer.add_batch(s, a, r, s_next, dw)
            s = s_next

            """update if its time"""
            # train 50 times every 50 steps rather than 1 training per step. Better!
            if total_steps >= opt.random_steps and total_steps % opt.update_every == 0:
                if perf_start_after_warmup is None:
                    perf_start_after_warmup = time.time()

                train_info = None
                for j in range(opt.update_every):
                    train_info = agent.train()
                    bstep += 1

                if writer is not None and train_info is not None:
                    elapsed_after_warmup = max(
                        time.time() - perf_start_after_warmup, 1e-6
                    )
                    data_steps = max(total_steps - opt.random_steps, 1)
                    sps = data_steps / elapsed_after_warmup
                    bps = bstep / elapsed_after_warmup
                    utd = (bstep * opt.batch_size) / data_steps
                    replay_usage = agent.replay_buffer.size / max(opt.buffersize, 1)

                    writer.add_scalar("Loss/Q", train_info["q_loss"], total_steps)
                    writer.add_scalar(
                        "Loss/Actor", train_info["actor_loss"], total_steps
                    )
                    writer.add_scalar("Alpha/value", train_info["alpha"], total_steps)
                    writer.add_scalar(
                        "Policy/Entropy", train_info["entropy"], total_steps
                    )
                    writer.add_scalar("Perf/SPS", sps, total_steps)
                    writer.add_scalar("Perf/BPS", bps, total_steps)
                    writer.add_scalar("Perf/UTD", utd, total_steps)
                    writer.add_scalar("Perf/ReplayUsage", replay_usage, total_steps)
                    writer.add_scalar("Perf/PolicyLagSteps", 0, total_steps)

                if (
                    total_steps - perf_last_print_steps >= 5000
                    or perf_last_print_steps < 0
                ):
                    elapsed_after_warmup = max(
                        time.time() - perf_start_after_warmup, 1e-6
                    )
                    data_steps = max(total_steps - opt.random_steps, 1)
                    sps = data_steps / elapsed_after_warmup
                    bps = bstep / elapsed_after_warmup
                    utd = (bstep * opt.batch_size) / data_steps
                    replay_usage = agent.replay_buffer.size / max(opt.buffersize, 1)
                    print(
                        f"(SAC Single) steps={total_steps / 1e3:.1f}k | SPS={sps:.1f} | "
                        f"BPS={bps:.1f} | UTD={utd:.2f} | Replay={replay_usage * 100:.1f}%"
                    )
                    perf_last_print_steps = total_steps

            """record & log"""
            if writer is not None and total_steps % 100 == 0:
                writer.add_scalar("Train/RewardMean", r.mean().item(), total_steps)
                writer.add_scalar(
                    "Train/DoneRate", done.float().mean().item(), total_steps
                )
                writer.add_scalar("Buffer/Size", agent.replay_buffer.size, total_steps)

            if total_steps > 0 and total_steps % opt.eval_interval == 0:
                test_ep_steps, test_ep_r, test_arrival_rate = evaluate(
                    eval_env, agent, deterministic=True, turns=20
                )
                print(
                    f"Eval@{total_steps}: ArrivalRate:{test_arrival_rate}, Reward:{test_ep_r}, Steps:{test_ep_steps}"
                )
                if writer is not None:
                    writer.add_scalar(
                        "Eval/ArrivalRate", test_arrival_rate, total_steps
                    )
                    writer.add_scalar("Eval/Reward", test_ep_r, total_steps)
                    writer.add_scalar("Eval/Steps", test_ep_steps, total_steps)
                    # compatibility tags for quick filtering
                    writer.add_scalar("arrival_rate", test_arrival_rate, total_steps)
                    writer.add_scalar("ep_r", test_ep_r, total_steps)
                    writer.add_scalar("ep_steps", test_ep_steps, total_steps)

            total_steps += 1

            """save model"""
            if total_steps % opt.save_interval == 0:
                agent.save(int(total_steps / 1000))

    if writer is not None:
        writer.close()
    env.close()
    eval_env.close()
    print("Training Finished.")


if __name__ == "__main__":
    main()
