import argparse
import os
import re
from datetime import datetime
from multiprocessing.managers import BaseManager

import torch
import torch.multiprocessing as mp

from ASL.sac_actor import sac_actor_process
from ASL.sac_learner import sac_learner_process
from ASL.sac_sharer import shared_data_sac
from Sparrow_V2 import str2bool


def _get_latest_actor_kstep(model_dir):
    if not os.path.isdir(model_dir):
        return None

    pattern = re.compile(r"^sacd_actor_(\d+)\.pth$")
    latest_kstep = None
    for name in os.listdir(model_dir):
        m = pattern.match(name)
        if m is None:
            continue
        kstep = int(m.group(1))
        if latest_kstep is None or kstep > latest_kstep:
            latest_kstep = kstep
    return latest_kstep


def _get_max_step_from_events(run_dir):
    if not os.path.isdir(run_dir):
        return None

    event_files = [
        os.path.join(run_dir, name)
        for name in os.listdir(run_dir)
        if name.startswith("events.out.tfevents")
    ]
    if len(event_files) == 0:
        return None

    try:
        from tensorboard.backend.event_processing import event_accumulator
    except Exception:
        return None

    max_step = -1
    for f in sorted(event_files):
        try:
            ea = event_accumulator.EventAccumulator(f)
            ea.Reload()
            for tag in ea.Tags().get("scalars", []):
                scalars = ea.Scalars(tag)
                if len(scalars) > 0:
                    max_step = max(max_step, int(scalars[-1].step))
        except Exception:
            continue

    return max_step if max_step >= 0 else None


def _resolve_resume_dirs(resume_from):
    if resume_from is None or str(resume_from).strip() == "":
        return None

    text = str(resume_from).strip().rstrip("/")
    if os.path.isabs(text):
        run_dir = text
    elif text.startswith("runs/"):
        run_dir = text
    elif text.startswith("SAC_ASL/"):
        run_dir = os.path.join("runs", text)
    else:
        run_dir = os.path.join("runs", "SAC_ASL", text)

    run_dir = os.path.normpath(run_dir)
    timestamp = os.path.basename(run_dir)
    model_dir = os.path.normpath(os.path.join("model", "SAC_ASL", timestamp))
    return run_dir, model_dir, timestamp


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # device configuration
    parser.add_argument(
        "--A_dvc", type=str, default="cuda:0", help="running device for SAC Actor"
    )
    parser.add_argument(
        "--B_dvc",
        type=str,
        default="cuda:0",
        help="running device for shared replay buffer",
    )
    parser.add_argument(
        "--L_dvc", type=str, default="cuda:0", help="running device for SAC Learner"
    )

    # SAC training configuration
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument(
        "--max_train_steps",
        "--Max_train_steps",
        dest="max_train_steps",
        type=int,
        default=int(1e8),
        help="Max training total steps",
    )
    parser.add_argument(
        "--random_steps",
        type=int,
        default=int(1e4),
        help="steps for random policy to explore",
    )
    parser.add_argument(
        "--update_every", type=int, default=50, help="training frequency"
    )
    parser.add_argument(
        "--upload_freq",
        type=int,
        default=100,
        help="learner upload actor frequency, in Bstep",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default="ColorDynamic_SAC_ASL",
        help="Experiment name",
    )
    parser.add_argument(
        "--reset_freq",
        type=int,
        default=int(32e3),
        help="training env reset frequency (curriculum learning), in total steps",
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=int(1e5),
        help="Model saving interval, in total steps",
    )
    parser.add_argument(
        "--eval_interval",
        type=int,
        default=int(2e3),
        help="Model evaluating interval, in total steps",
    )
    parser.add_argument(
        "--eval_turns",
        type=int,
        default=20,
        help="number of finished episodes used in each evaluation",
    )
    parser.add_argument(
        "--download_check_interval",
        type=int,
        default=320,
        help="actor checks new model every X total steps",
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Resume from timestamp/path, e.g. 20260424_130245 or runs/SAC_ASL/20260424_130245",
    )
    parser.add_argument(
        "--resume_total_steps",
        type=int,
        default=None,
        help="Manually set resumed total steps (overrides auto infer)",
    )

    parser.add_argument("--gamma", type=float, default=0.99, help="Discounted Factor")
    parser.add_argument(
        "--hid_shape", type=list, default=[200, 200], help="Hidden net shape"
    )
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=256, help="batch size")
    parser.add_argument("--alpha", type=float, default=0.2, help="init alpha")
    parser.add_argument(
        "--adaptive_alpha",
        type=str2bool,
        default=True,
        help="Use adaptive alpha tuning",
    )
    parser.add_argument(
        "--write",
        type=str2bool,
        default=True,
        help="Use SummaryWriter to record the training",
    )

    # Sparrow env configuration
    parser.add_argument(
        "--dvc", type=str, default="cuda", help="running device of Sparrow: cuda / cpu"
    )
    parser.add_argument(
        "--action_type",
        type=str,
        default="Discrete",
        help="Action type: Discrete / Continuous",
    )
    parser.add_argument(
        "--window_size", type=int, default=800, help="size of the training map"
    )
    parser.add_argument(
        "--D", type=int, default=400, help="maximal local planning distance"
    )
    parser.add_argument(
        "--N", type=int, default=32, help="number of vectorized environments"
    )
    parser.add_argument(
        "--O", type=int, default=15, help="number of obstacles in each environment"
    )
    parser.add_argument(
        "--RdON",
        type=str2bool,
        default=False,
        help="whether to randomize the Number of dynamic obstacles",
    )
    parser.add_argument(
        "--ScOV",
        type=str2bool,
        default=False,
        help="whether to scale the maximal velocity of dynamic obstacles",
    )
    parser.add_argument(
        "--RdOV",
        type=str2bool,
        default=False,
        help="whether to randomize the Velocity of dynamic obstacles",
    )
    parser.add_argument(
        "--RdOT",
        type=str2bool,
        default=True,
        help="whether to randomize the Type of dynamic obstacles",
    )
    parser.add_argument(
        "--RdOR",
        type=str2bool,
        default=True,
        help="whether to randomize the Radius of obstacles",
    )
    parser.add_argument(
        "--Obs_R", type=int, default=14, help="maximal obstacle radius, cm"
    )
    parser.add_argument(
        "--Obs_V", type=int, default=50, help="maximal obstacle velocity, cm/s"
    )
    parser.add_argument(
        "--MapObs",
        type=str,
        default=None,
        help="name of map file, e.g. 'map.png' or None",
    )
    parser.add_argument(
        "--ld_a_range",
        type=int,
        default=360,
        help="max scanning angle of lidar (degree)",
    )
    parser.add_argument(
        "--ld_d_range",
        type=int,
        default=300,
        help="max scanning distance of lidar (cm)",
    )
    parser.add_argument(
        "--ld_num", type=int, default=72, help="number of lidar streams in each world"
    )
    parser.add_argument(
        "--ld_GN",
        type=int,
        default=3,
        help="how many lidar streams are grouped for each group",
    )
    parser.add_argument(
        "--ri",
        type=int,
        default=0,
        help="render index: the index of world that be rendered",
    )
    parser.add_argument(
        "--basic_ctrl_interval",
        type=float,
        default=0.1,
        help="control interval (s), 0.1 means 10 Hz control frequency",
    )
    parser.add_argument(
        "--ctrl_delay",
        type=int,
        default=0,
        help="control delay, in basic_ctrl_interval, 0 means no control delay",
    )
    parser.add_argument(
        "--K", type=tuple, default=(0.55, 0.6), help="K_linear, K_angular"
    )
    parser.add_argument(
        "--draw_auxiliary",
        type=str2bool,
        default=False,
        help="whether to draw auxiliary infos",
    )
    parser.add_argument(
        "--render_speed", type=str, default="fast", help="fast / slow / real"
    )
    parser.add_argument(
        "--max_ep_steps", type=int, default=500, help="maximum episodic steps"
    )
    parser.add_argument(
        "--noise",
        type=str2bool,
        default=True,
        help="whether to add noise to the observations",
    )
    parser.add_argument(
        "--DR", type=str2bool, default=True, help="whether to use Domain Randomization"
    )
    parser.add_argument(
        "--DR_freq",
        type=int,
        default=int(3.2e3),
        help="frequency of Domain Randomization, in total steps",
    )
    parser.add_argument(
        "--compile",
        type=str2bool,
        default=True,
        help="whether to use torch.compile to boost simulation speed",
    )

    opt = parser.parse_args()

    resume_dirs = _resolve_resume_dirs(opt.resume_from)
    opt.resume_actor_ckpt = None
    opt.resume_actor_kstep = None
    opt.initial_total_steps = 0

    if resume_dirs is None:
        opt.run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        opt.run_dir = os.path.join("runs", "SAC_ASL", opt.run_timestamp)
        opt.model_dir = os.path.join("model", "SAC_ASL", opt.run_timestamp)
    else:
        opt.run_dir, opt.model_dir, opt.run_timestamp = resume_dirs
        latest_actor_kstep = _get_latest_actor_kstep(opt.model_dir)
        if latest_actor_kstep is not None:
            opt.resume_actor_kstep = latest_actor_kstep
            opt.resume_actor_ckpt = os.path.join(
                opt.model_dir, f"sacd_actor_{latest_actor_kstep}.pth"
            )
            opt.initial_total_steps = latest_actor_kstep * 1000
        else:
            inferred_step = _get_max_step_from_events(opt.run_dir)
            if inferred_step is not None:
                opt.initial_total_steps = inferred_step

    if opt.resume_total_steps is not None:
        opt.initial_total_steps = int(opt.resume_total_steps)

    os.makedirs(opt.run_dir, exist_ok=True)
    os.makedirs(opt.model_dir, exist_ok=True)

    opt.render_mode = None
    opt.buffersize = min(int(1e6), opt.max_train_steps)
    opt.reset_freq = int(opt.reset_freq / opt.N)  # Tsteps -> Vsteps

    opt.dvc = torch.device(opt.dvc)
    opt.state_dim = 8 + int(opt.ld_num / opt.ld_GN)
    opt.action_dim = 7

    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)

    print(f"[SAC_ASL] logs -> {opt.run_dir}")
    print(f"[SAC_ASL] models -> {opt.model_dir}")
    if opt.resume_from is not None:
        print(
            f"[SAC_ASL] resume -> total_steps={opt.initial_total_steps}, actor_ckpt={opt.resume_actor_ckpt}"
        )

    BaseManager.register("shared_data_sac", callable=shared_data_sac)
    ShareManager = BaseManager()
    ShareManager.start()
    opt.shared_data = ShareManager.shared_data_sac(opt)
    opt.shared_data.set_total_steps(opt.initial_total_steps)

    processes = []
    processes.append(mp.Process(target=sac_actor_process, args=(opt,)))
    processes[-1].start()

    processes.append(mp.Process(target=sac_learner_process, args=(opt,)))
    processes[-1].start()

    for p in processes:
        p.join()
