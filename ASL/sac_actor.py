import time

import torch
from torch.distributions.categorical import Categorical

from Sparrow_V2 import Sparrow
from utils.utils_SAC import Policy_Net


def sac_actor_process(opt):
    actor = SACActor(opt)
    actor.run()


class SACActor:
    def __init__(self, opt):
        self.A_dvc = torch.device(opt.A_dvc)
        self.shared_data = opt.shared_data

        self.O = opt.O
        self.N = opt.N
        self.reset_freq = opt.reset_freq
        self.exp_name = opt.exp_name
        self.max_train_steps = opt.max_train_steps
        self.random_steps = opt.random_steps
        self.download_check_interval = opt.download_check_interval

        self.state_dim = opt.state_dim
        self.action_dim = opt.action_dim
        self.hid_shape = opt.hid_shape

        self.envs = Sparrow(**vars(opt))
        self.actor = Policy_Net(self.state_dim, self.action_dim, self.hid_shape).to(
            self.A_dvc
        )
        for p in self.actor.parameters():
            p.requires_grad = False

        self.total_steps = self.shared_data.get_total_steps()
        self.t_start = time.time()
        if self.total_steps > 0:
            print(f"(SAC Actor) Resume total steps: {self.total_steps}")
        print("SAC Actor Started!")

    def run(self):
        s, info = self.envs.reset()
        ep_r = 0.0

        while self.total_steps < self.max_train_steps:
            # baby-step curriculum learning (same idea as train_ColorDynamic.py)
            if (
                self.total_steps > 0
                and self.total_steps % (self.reset_freq * self.N) == 0
            ):
                self.envs.O = (
                    int(
                        self.O
                        * min(0.15, self.total_steps / self.max_train_steps)
                        / 0.15
                    )
                    + 1
                )
                print(
                    f"(SAC Actor) {self.exp_name}, Total steps: {round(self.total_steps / 1e3, 2)}k; "
                    f"Obstacle Numbers: {self.envs.O}"
                )
                s, info = self.envs.reset()
                ep_r = 0.0
                continue

            if self.total_steps < self.random_steps:
                a = torch.randint(0, self.action_dim, (self.N,), device=self.A_dvc)
            else:
                a = self.select_action(s)

            s_next, r, dw, tr, info = self.envs.step(a)
            self.shared_data.add(s, a, r, s_next, dw)

            s = s_next
            self.total_steps += self.N
            self.shared_data.set_total_steps(self.total_steps)

            if (
                self.total_steps > self.random_steps
                and self.total_steps % self.download_check_interval == 0
            ):
                if self.shared_data.get_should_download():
                    self.shared_data.set_should_download(False)
                    self.download_model()

            ep_r += float(r[0].item())
            if bool(dw[0] or tr[0]):
                print("-" * 90)
                print(
                    f"(SAC Actor) Total steps: {round(self.total_steps / 1e3, 2)}k, ep_r: {round(ep_r, 1)}"
                )
                time_consumed = time.time() - self.t_start
                print(f"(SAC Actor) Consumed Time: {round(time_consumed / 3600, 2)}h")
                print("-" * 90 + "\n")
                ep_r = 0.0

        self.envs.close()
        print("---------------- SAC Actor Finished ----------------")

    def select_action(self, s):
        with torch.no_grad():
            probs = self.actor(s)
            dist = Categorical(probs)
            return dist.sample()

    def download_model(self):
        param = self.shared_data.get_actor_param()
        if param is not None:
            self.actor.load_state_dict(param)
            for p in self.actor.parameters():
                p.requires_grad = False
