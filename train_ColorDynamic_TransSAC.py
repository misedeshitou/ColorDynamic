import argparse
import copy
import os
import re
import sys
from datetime import datetime

import torch
from torch.utils.tensorboard import SummaryWriter

# torch.set_num_threads(4) # 甚至可以尝试 1 或 2
# from utils.utils_TransSAC import evaluate_policy
from Sparrow_V2 import Sparrow, str2bool
from utils.TransSAC import TransSAC_agent


def random_action_discrete(env):
    return torch.randint(low=0, high=env.action_dim, size=(env.N,), device=env.dvc)


def _get_latest_transsac_kstep(model_dir):
    if not os.path.isdir(model_dir):
        return None

    pattern = re.compile(r"^transsac_(?:ckpt|actor)_(\d+)\.pth$")
    latest_kstep = None
    for name in os.listdir(model_dir):
        matched = pattern.match(name)
        if matched is None:
            continue

        kstep = int(matched.group(1))
        if latest_kstep is None or kstep > latest_kstep:
            latest_kstep = kstep
    return latest_kstep


def _resolve_resume_dirs(project_root, resume_from):
    if resume_from is None or str(resume_from).strip() == "":
        return None

    text = str(resume_from).strip().rstrip("/")
    if os.path.isabs(text):
        run_dir = text
    elif text.startswith("runs/"):
        run_dir = os.path.join(project_root, text)
    elif text.startswith("TransSAC/"):
        run_dir = os.path.join(project_root, "runs", text)
    else:
        run_dir = os.path.join(project_root, "runs", "TransSAC", text)

    run_dir = os.path.normpath(run_dir)
    run_timestamp = os.path.basename(run_dir)
    model_dir = os.path.normpath(
        os.path.join(project_root, "model", "TransSAC", run_timestamp)
    )
    return run_dir, model_dir, run_timestamp


def _normalize_cli_argv(argv):
    """兼容误写参数：--ModelIndex-1 -> --ModelIndex -1"""
    normalized = [argv[0]]
    pattern = re.compile(r"^--ModelIndex(-?\d+)$")

    for token in argv[1:]:
        matched = pattern.match(token)
        if matched is not None:
            normalized.extend(["--ModelIndex", matched.group(1)])
        else:
            normalized.append(token)

    return normalized


def evaluate(envs, agent, deterministic=True, turns=20):
    queue_state = copy.deepcopy(agent.queue.__dict__)
    step_collector, total_steps = torch.zeros(envs.N, device=envs.dvc), 0
    r_collector, total_r = torch.zeros(envs.N, device=envs.dvc), 0
    arrived, finished = 0, 0

    try:
        agent.queue.clear()
        s, info = envs.reset()
        while finished < turns:
            a = agent.select_action(s, deterministic)
            s, r, dw, tr, info = envs.step(a)

            dones = dw + tr
            wins = r == envs.AWARD
            dead_and_tr = dones ^ wins

            if dones.any():
                agent.queue.padding_with_done(dones)

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
    finally:
        agent.queue.__dict__.clear()
        agent.queue.__dict__.update(queue_state)


# fmt: off
if __name__ == '__main__':
    '''Hyperparameter Setting for DRL'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--write', type=str2bool, default=True, help='Use SummaryWriter to record the training')
    parser.add_argument('--render', type=str2bool, default=False, help='Render or Not')
    parser.add_argument('--Loadmodel', type=str2bool, default=False, help='Load pretrained model or Not')
    parser.add_argument('--ModelIndex', type=int, default=-1, help='which model to load (k steps), -1 means latest')
    parser.add_argument('--resume_from', type=str, default=None, help='Resume from timestamp/path, e.g. 20260424_130245 or runs/TransSAC/20260424_130245')
    parser.add_argument('--resume_total_steps', type=int, default=None, help='Manually set resumed total steps (overrides auto infer)')
    parser.add_argument('--load_model_dir', type=str, default=None, help='Checkpoint directory used only for loading model')

    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--max_train_steps', type=int, default=5e7, help='Max training steps')
    parser.add_argument('--save_interval', type=int, default=5e4, help='Model saving interval, in steps.')
    parser.add_argument('--eval_interval', type=int, default=5e3, help='Model evaluating interval, in steps.')
    parser.add_argument('--eval_enable', type=str2bool, default=False, help='Enable periodic evaluation during training')
    parser.add_argument('--random_steps', type=int, default=1e4, help='steps for random policy to explore')
    parser.add_argument('--update_every', type=int, default=10, help='training frequency')
    parser.add_argument('--train_repeat', type=int, default=4, help='Extra train repeats per update trigger (increase to raise GPU utilization)')
    parser.add_argument('--fast_mode', type=str2bool, default=True, help='Use faster (non-deterministic) backend settings for higher throughput')

    parser.add_argument('--gamma', type=float, default=0.99, help='Discounted Factor')
    parser.add_argument('--net_width', type=int, default=32, help='Linear net width')
    parser.add_argument('--hid_shape', type=list, default=[200,200], help='Hidden net shape')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=512, help='batch size')
    parser.add_argument('--alpha', type=float, default=0.2, help='init alpha')
    parser.add_argument('--adaptive_alpha', type=str2bool, default=True, help='Use adaptive alpha turning')

    # Transqer configuration
    parser.add_argument('--T', type=int, default=10, help='length of time window')
    parser.add_argument('--H', type=int, default=4, help='Number of Head')
    parser.add_argument('--L', type=int, default=2, help='Number of Transformer Encoder Layers')

    '''Hyperparameter Setting for Sparrow'''
    parser.add_argument('--dvc', type=str, default='cuda', help='running device of Sparrow: cuda / cpu')
    parser.add_argument('--action_type', type=str, default='Discrete', help='Action type: Discrete / Continuous')
    parser.add_argument('--window_size', type=int, default=800, help='size of the training map')
    parser.add_argument('--D', type=int, default=400, help='maximal local planning distance')
    parser.add_argument('--N', type=int, default=64, help='number of vectorized environments')
    parser.add_argument('--O', type=int, default=5, help='number of obstacles in each environment')
    parser.add_argument('--RdON', type=str2bool, default=False, help='whether to randomize the Number of dynamic obstacles')
    parser.add_argument('--ScOV', type=str2bool, default=False, help='whether to scale the maximal velocity of dynamic obstacles')
    parser.add_argument('--RdOV', type=str2bool, default=False, help='whether to randomize the Velocity of dynamic obstacles')
    parser.add_argument('--RdOT', type=str2bool, default=True, help='whether to randomize the Type of dynamic obstacles')
    parser.add_argument('--RdOR', type=str2bool, default=True, help='whether to randomize the Radius of obstacles')
    parser.add_argument('--Obs_R', type=int, default=14, help='maximal obstacle radius, cm')
    parser.add_argument('--Obs_V', type=int, default=10, help='maximal obstacle velocity, cm/s')
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
    normalized_argv = _normalize_cli_argv(sys.argv)
    opt = parser.parse_args(normalized_argv[1:])
    if opt.action_type != 'Discrete':
        raise ValueError("TransSAC currently only supports discrete actions. Please set --action_type Discrete.")
    project_root = os.path.dirname(os.path.abspath(__file__))

    resume_dirs = _resolve_resume_dirs(project_root, opt.resume_from)
    if resume_dirs is None:
        opt.run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        opt.run_dir = os.path.join(project_root, "runs", "TransSAC", opt.run_timestamp)
        opt.model_dir = os.path.join(project_root, "model", "TransSAC", opt.run_timestamp)
    else:
        opt.run_dir, opt.model_dir, opt.run_timestamp = resume_dirs

    os.makedirs(opt.run_dir, exist_ok=True)
    os.makedirs(opt.model_dir, exist_ok=True)

    # Backward compatibility: old checkpoints are saved under project_root/model
    if opt.load_model_dir is not None and str(opt.load_model_dir).strip() != "":
        opt.load_model_dir = os.path.normpath(
            opt.load_model_dir
            if os.path.isabs(opt.load_model_dir)
            else os.path.join(project_root, opt.load_model_dir)
        )
    elif opt.resume_from is not None:
        opt.load_model_dir = opt.model_dir
    else:
        opt.load_model_dir = os.path.join(project_root, "model")

    latest_kstep = None
    if opt.Loadmodel and opt.ModelIndex < 0:
        latest_kstep = _get_latest_transsac_kstep(opt.load_model_dir)
        if latest_kstep is None:
            raise FileNotFoundError(
                f"No TransSAC checkpoint found in load_model_dir: {opt.load_model_dir}"
            )
        opt.ModelIndex = latest_kstep

    if opt.resume_from is not None and not opt.Loadmodel:
        latest_kstep = _get_latest_transsac_kstep(opt.load_model_dir)
        if latest_kstep is not None:
            opt.Loadmodel = True
            opt.ModelIndex = latest_kstep

    opt.initial_total_steps = 0
    if opt.Loadmodel:
        opt.initial_total_steps = int(opt.ModelIndex) * 1000
    if opt.resume_total_steps is not None:
        opt.initial_total_steps = int(opt.resume_total_steps)


    opt.render_mode = None # dont render when training
    opt.buffersize = min(int(1E6), opt.max_train_steps)
    # opt.reset_freq = int(opt.reset_freq / opt.N)  # Tsteps -> Vsteps

    opt.dvc = torch.device(opt.dvc)
    opt.state_dim = 8+int(opt.ld_num/opt.ld_GN)
    opt.action_dim = 7
# fmt: on

   # Seed Everything
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    if opt.fast_mode:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    else:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Create Env & Agent
    env = Sparrow(**vars(opt))
    eval_env = Sparrow(**vars(opt)) if opt.eval_enable else None
    opt.action_dim = env.action_dim
    agent = TransSAC_agent(**vars(opt))

    # 限制 CPU 线程数，防止多线程空转抢夺 CPU 导致 100% 满载
    torch.set_num_threads(4)

    print("Random Seed: {}".format(opt.seed))
    print(f"[TransSAC] logs -> {opt.run_dir}")
    print(f"[TransSAC] models -> {opt.model_dir}")
    print(f"[TransSAC] fast_mode={opt.fast_mode}, eval_enable={opt.eval_enable}, train_repeat={opt.train_repeat}")
    
    if opt.Loadmodel:
        print(f"[TransSAC] resume ckpt -> {os.path.join(opt.load_model_dir, f'transsac_ckpt_{opt.ModelIndex}.pth')}")
        agent.load(opt.ModelIndex, model_dir=opt.load_model_dir)
        opt.initial_total_steps = int(getattr(agent, "loaded_total_steps", opt.initial_total_steps))

    agent.total_steps = int(opt.initial_total_steps)
        
    print(f"[TransSAC] start total_steps -> {opt.initial_total_steps}")

    # ================= 恢复被遗漏的 writer 初始化 =================
    writer = None
    if opt.write:
        writer_kwargs = {"log_dir": opt.run_dir}
        if opt.initial_total_steps > 0:
            writer_kwargs["purge_step"] = opt.initial_total_steps
        writer = SummaryWriter(**writer_kwargs)
        writer.add_text("config", str(vars(opt)))
    # ==============================================================

    total_steps = int(opt.initial_total_steps)
    
    s, info = env.reset()
    
    # 重置环境时清空智能体的时间窗记忆
    agent.queue.clear() 

    # ================= 展平的主循环，去除 while not done.all() 死循环 =================
    while total_steps < opt.max_train_steps:
        # e-greedy exploration
        if total_steps < opt.random_steps:
            a = random_action_discrete(env)
            agent.queue.append(s) # 维护队列
        else:
            a = agent.select_action(s, deterministic=False)
            
        s_next, r, dw, tr, info = env.step(a)
        
        # --- 判断哪些环境结束了，清除这些环境对应的历史记录 ---
        dones = dw | tr
        if dones.any():
            agent.queue.padding_with_done(dones)

        # (Buffer 存的是单帧，采样时会自动溯源 T 步)
        agent.replay_buffer.add_batch(s, a, r, s_next, dw)
        s = s_next

        """update if its time"""
        # train multiple times every update_every steps
        if total_steps >= opt.random_steps and total_steps % opt.update_every == 0:
            train_info = None
            update_loops = max(1, int(opt.update_every * opt.train_repeat))
            for _ in range(update_loops):
                train_info = agent.train()

            if writer is not None and train_info is not None:
                writer.add_scalar("Loss/Q", train_info["q_loss"], total_steps)
                writer.add_scalar("Loss/Actor", train_info["actor_loss"], total_steps)
                writer.add_scalar("Alpha/value", train_info["alpha"], total_steps)
                writer.add_scalar("Policy/Entropy", train_info["entropy"], total_steps)

        if (
            opt.eval_enable
            and eval_env is not None
            and total_steps > 0
            and total_steps % opt.eval_interval == 0
        ):
            test_ep_steps, test_ep_r, test_arrival_rate = evaluate(
                eval_env, agent, deterministic=True, turns=20
            )
            print(
                f"Eval@{total_steps}: ArrivalRate:{test_arrival_rate}, Reward:{test_ep_r}, Steps:{test_ep_steps}"
            )
            if writer is not None:
                writer.add_scalar("Eval/ArrivalRate", test_arrival_rate, total_steps)
                writer.add_scalar("Eval/Reward", test_ep_r, total_steps)

        total_steps += 1
        agent.total_steps = total_steps

        """save model"""
        if total_steps % opt.save_interval == 0:
            agent.save(int(total_steps / 1000))

    if writer is not None:
        writer.close()

    env.close()
    if eval_env is not None:
        eval_env.close()
    print("Training Finished.")
