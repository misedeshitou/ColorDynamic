import argparse
from multiprocessing.managers import BaseManager

import torch
import torch.multiprocessing as mp

from ASL.sac_actor import sac_actor_process
from ASL.sac_learner import sac_learner_process
from ASL.sac_sharer import shared_data_sac
from Sparrow_V2 import str2bool

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
        "--save_interval",
        type=int,
        default=int(1e5),
        help="Model saving interval, in total steps",
    )
    parser.add_argument(
        "--download_check_interval",
        type=int,
        default=320,
        help="actor checks new model every X total steps",
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

    opt.render_mode = None
    opt.buffersize = min(int(1e6), opt.max_train_steps)

    opt.dvc = torch.device(opt.dvc)
    opt.state_dim = 8 + int(opt.ld_num / opt.ld_GN)
    opt.action_dim = 7

    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)

    BaseManager.register("shared_data_sac", callable=shared_data_sac)
    ShareManager = BaseManager()
    ShareManager.start()
    opt.shared_data = ShareManager.shared_data_sac(opt)

    processes = []
    processes.append(mp.Process(target=sac_actor_process, args=(opt,)))
    processes[-1].start()

    processes.append(mp.Process(target=sac_learner_process, args=(opt,)))
    processes[-1].start()

    for p in processes:
        p.join()
