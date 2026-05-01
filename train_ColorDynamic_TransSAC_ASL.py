import argparse
import os
import re
from datetime import datetime
from multiprocessing.managers import BaseManager

import torch
import torch.multiprocessing as mp

from ASL.transsac_actor import transsac_actor_process
from ASL.transsac_learner import transsac_learner_process
from ASL.transsac_sharer import shared_data_transsac
from Sparrow_V2 import str2bool

# fmt: off
if __name__ == '__main__':
    '''Hyperparameter Setting for DRL'''
    parser = argparse.ArgumentParser()
    # running devices configuration
    parser.add_argument('--Env_dvc', type=str, default='cuda:0', help='running device for Sparrow Env')
    parser.add_argument('--A_dvc', type=str, default='cuda:0', help='running device for Actor')
    parser.add_argument('--B_dvc', type=str, default='cuda:0', help='running device for Buffer of Sharer')
    parser.add_argument('--L_dvc', type=str, default='cuda:0', help='running device for Learner')

    # training strategy configuration
    parser.add_argument('--exp_name', type=str, default='ColorDynamic_TransSAC', help='Experiment name')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--max_train_steps', type=int, default=int(2E6), help='Max training total steps')
    parser.add_argument('--random_steps', type=int, default=int(1E4), help='steps for random policy exploration')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discounted Factor')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--alpha', type=float, default=0.2, help='init alpha')
    parser.add_argument('--adaptive_alpha', type=str2bool, default=True, help='Use adaptive alpha turning')
    parser.add_argument('--update_every', type=int, default=10, help='training frequency')
    parser.add_argument('--train_repeat', type=int, default=8, help='Extra train repeats per update trigger')
    parser.add_argument('--upload_freq', type=int, default=int(500), help='actor download freq, in batch steps')
    parser.add_argument('--save_interval', type=int, default=int(5e4), help='Model save frequency, in batch steps')
    parser.add_argument('--eval_interval', type=int, default=int(5e3), help='Model evaluation frequency')
    parser.add_argument('--reset_freq', type=int, default=int(3.2e3), help='training env reset frequency (curriculum learning), in total steps')

    # Transformer configuration
    parser.add_argument('--net_width', type=int, default=32, help='Linear net width')
    parser.add_argument('--hid_shape', type=list, default=[200, 200], help='Hidden layer shape')
    parser.add_argument('--T', type=int, default=10, help='length of time window')
    parser.add_argument('--H', type=int, default=4, help='Number of Head')
    parser.add_argument('--L', type=int, default=2, help='Number of Transformer Encoder Layers')
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Resume from timestamp/path, e.g. 20260424_130245 or runs/TransSAC_ASL/20260424_130245",
    )
    parser.add_argument(
        "--resume_total_steps",
        type=int,
        default=None,
        help="Manually set resumed total steps (overrides auto infer)",
    )

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
    
    opt = parser.parse_args()
    resume_dirs = None
    opt.resume_actor_ckpt = None
    opt.resume_actor_kstep = None
    opt.initial_total_steps = 0

    def _get_latest_actor_kstep(model_dir):
        if not os.path.isdir(model_dir):
            return None
        pattern = re.compile(r"^transsac_ckpt_(\d+)\.pth$")
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
        elif text.startswith("TransSAC_ASL/"):
            run_dir = os.path.join("runs", text)
        else:
            run_dir = os.path.join("runs", "TransSAC_ASL", text)
        run_dir = os.path.normpath(run_dir)
        timestamp = os.path.basename(run_dir)
        model_dir = os.path.normpath(os.path.join("model", "TransSAC_ASL", timestamp))
        return run_dir, model_dir, timestamp

    resume_dirs = _resolve_resume_dirs(getattr(opt, "resume_from", None))

    opt.render_mode = None  # dont render when training
    opt.buffersize = min(int(1E6), opt.max_train_steps)
    opt.reset_freq = int(opt.reset_freq / opt.N)  # Tsteps -> Vsteps

    opt.dvc = torch.device(opt.Env_dvc)
    opt.state_dim = 8 + int(opt.ld_num / opt.ld_GN)
    opt.action_dim = 7
    
    # Setup run directory (support resume)
    if resume_dirs is None:
        opt.run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        opt.run_dir = os.path.join("runs", "TransSAC_ASL", opt.run_timestamp)
        opt.model_dir = os.path.join("model", "TransSAC_ASL", opt.run_timestamp)
    else:
        opt.run_dir, opt.model_dir, opt.run_timestamp = resume_dirs
        latest_actor_kstep = _get_latest_actor_kstep(opt.model_dir)
        if latest_actor_kstep is not None:
            opt.resume_actor_kstep = latest_actor_kstep
            opt.resume_actor_ckpt = os.path.join(opt.model_dir, f"transsac_ckpt_{latest_actor_kstep}.pth")
            opt.initial_total_steps = latest_actor_kstep * 1000
        else:
            inferred_step = _get_max_step_from_events(opt.run_dir)
            if inferred_step is not None:
                opt.initial_total_steps = inferred_step

    if opt.resume_total_steps is not None:
        opt.initial_total_steps = int(opt.resume_total_steps)

    os.makedirs(opt.run_dir, exist_ok=True)
    os.makedirs(opt.model_dir, exist_ok=True)
# fmt: on

    # Set seed
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)

    print("Random Seed: {}".format(opt.seed))
    print(f"[TransSAC ASL] logs -> {opt.run_dir}")
    print(f"[TransSAC ASL] models -> {opt.model_dir}")

    # Register shared_data class for multiprocessing
    BaseManager.register("shared_data_transsac", callable=shared_data_transsac)
    ShareManager = BaseManager()
    ShareManager.start()
    opt.shared_data = ShareManager.shared_data_transsac(opt)

    processes = []

    # Learner process
    print("[Main] Starting Learner process...")
    processes.append(mp.Process(target=transsac_learner_process, args=(opt,)))
    processes[-1].start()

    # Actor process
    print("[Main] Starting Actor process...")
    processes.append(mp.Process(target=transsac_actor_process, args=(opt,)))
    processes[-1].start()

    # Wait for all processes
    for p in processes:
        p.join()

    print("[Main] Training finished!")
