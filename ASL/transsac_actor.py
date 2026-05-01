import time

import torch

from Sparrow_V2 import Sparrow
from utils.TransSAC import TransSAC_agent


def transsac_actor_process(opt):
    actor = TransSACActor(opt)
    actor.run()


class TransSACActor:
    def __init__(self, opt):
        self.A_dvc = torch.device(opt.A_dvc)
        self.shared_data = opt.shared_data

        self.O = opt.O
        self.N = opt.N
        self.reset_freq = opt.reset_freq
        self.exp_name = opt.exp_name
        self.max_train_steps = opt.max_train_steps
        self.random_steps = opt.random_steps

        self.state_dim = opt.state_dim
        self.action_dim = opt.action_dim

        self.envs = Sparrow(**vars(opt))

        # Create TransSAC agent (with queue)
        self.agent = TransSAC_agent(**vars(opt))
        for p in self.agent.actor.parameters():
            p.requires_grad = False

        self.total_steps = self.shared_data.get_total_steps()
        self.t_start = time.time()
        if self.total_steps > 0:
            print(f"(TransSAC Actor) Resume total steps: {self.total_steps}")
        print("TransSAC Actor Started!")

    def run(self):
        s, info = self.envs.reset()
        self.agent.queue.clear()
        ep_r = 0.0

        while self.total_steps < self.max_train_steps:
            # baby-step curriculum learning (前15%进行难度递增)
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
                    f"(TransSAC Actor) {self.exp_name}, Total steps: {round(self.total_steps / 1e3, 2)}k; "
                    f"Obstacle Numbers: {self.envs.O}"
                )
                s, info = self.envs.reset()
                self.agent.queue.clear()
                ep_r = 0.0
                self.total_steps += self.N  # Ensure total_steps is incremented
                continue

            """向sharer更新total steps"""
            self.shared_data.set_total_steps(self.total_steps)

            # Append state to time window queue
            self.agent.queue.append(s)
            tw_s = self.agent.queue.get()

            if self.total_steps < self.random_steps:
                a = torch.randint(0, self.action_dim, (self.N,), device=self.A_dvc)
            else:
                a = self.agent.select_action(s, deterministic=False)

            s_next, r, dw, tr, info = self.envs.step(a)

            # Add transition to shared buffer (with time window state)
            dones = dw | tr
            self.shared_data.add(tw_s, a, r, dones)

            # Handle done environments
            if dones.any():
                self.agent.queue.padding_with_done(dones)

            s = s_next
            self.total_steps += self.N

            """Download model parameters from shared_data"""
            if self.total_steps % (10 * self.N) == 0:
                if self.shared_data.get_should_download():
                    self.shared_data.set_should_download(False)
                    self.download_model()

            """打印回合累计奖励"""
            ep_r += r[0].item() if torch.is_tensor(r[0]) else r[0]
            if dw[0] or tr[0]:
                print(
                    f"(TransSAC Actor) {self.exp_name}, Total steps: {round(self.total_steps / 1e3, 2)}k, ep_r: {round(ep_r, 1)}"
                )
                ep_r = 0.0

        print("---------------- TransSAC Actor Finished ----------------")

    def download_model(self):
        """从shared_data下载actor参数"""
        actor_param = self.shared_data.get_actor_param()
        if actor_param is not None:
            self.agent.actor.load_state_dict(actor_param)
