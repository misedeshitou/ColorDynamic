import os
import time
from copy import deepcopy
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from Sparrow_V2 import Sparrow
from utils.utils_SAC import Double_Q_Net, Policy_Net


def sac_learner_process(opt):
    learner = SACLearner(opt)
    learner.run()


class SACLearner:
    def __init__(self, opt):
        self.L_dvc = torch.device(opt.L_dvc)
        self.shared_data = opt.shared_data

        self.max_train_steps = opt.max_train_steps
        self.random_steps = opt.random_steps
        self.batch_size = opt.batch_size
        self.gamma = opt.gamma
        self.lr = opt.lr
        self.action_dim = opt.action_dim
        self.update_every = opt.update_every
        self.upload_freq = opt.upload_freq
        self.save_interval = opt.save_interval
        self.eval_interval = opt.eval_interval
        self.eval_turns = opt.eval_turns
        self.adaptive_alpha = opt.adaptive_alpha
        self.buffer_capacity = int(opt.buffersize)

        self.actor = Policy_Net(opt.state_dim, opt.action_dim, opt.hid_shape).to(
            self.L_dvc
        )
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)

        self.q_critic = Double_Q_Net(opt.state_dim, opt.action_dim, opt.hid_shape).to(
            self.L_dvc
        )
        self.q_critic_optimizer = torch.optim.Adam(
            self.q_critic.parameters(), lr=self.lr
        )
        self.q_critic_target = deepcopy(self.q_critic)
        for p in self.q_critic_target.parameters():
            p.requires_grad = False

        self.tau = 0.005
        self.alpha = opt.alpha
        self.H_mean = 0
        if self.adaptive_alpha:
            self.target_entropy = 0.6 * (-np.log(1 / self.action_dim))
            self.log_alpha = torch.tensor(
                np.log(self.alpha), dtype=float, requires_grad=True, device=self.L_dvc
            )
            self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=self.lr)

        self.Bstep = 0
        self.start_time = time.time()
        self.start_time_after_warmup = None
        self.last_upload_total_steps = 0

        if not os.path.exists("model"):
            os.mkdir("model")

        self.writer = None
        if opt.write:
            run_name = (
                f"SAC-ASL-C{opt.O}-N{opt.N}-{datetime.now().strftime('%Y-%m-%d %H_%M')}"
            )
            self.writer = SummaryWriter(log_dir=os.path.join("runs", run_name))
            self.writer.add_text("config", str(vars(opt)))

        self.eval_env = Sparrow(**vars(opt))

        # first upload so actor can use non-random policy later
        self.upload_actor()

        print("SAC Learner Started!")

    def run(self):
        last_trained_total_steps = -1
        last_perf_log_steps = -1
        last_eval_total_steps = 0

        while True:
            total_steps = self.shared_data.get_total_steps()
            buffer_size = self.shared_data.get_buffer_size()

            if total_steps >= self.max_train_steps:
                break

            if total_steps < self.random_steps or buffer_size < self.batch_size:
                time.sleep(0.1)
                continue

            if self.start_time_after_warmup is None:
                self.start_time_after_warmup = time.time()

            # mimic original single-process rhythm: every update_every env-steps, do update_every gradient steps
            if (
                total_steps != last_trained_total_steps
                and total_steps % self.update_every == 0
            ):
                train_info = None
                for _ in range(self.update_every):
                    train_info = self.train_step()
                    self.Bstep += 1

                    if self.Bstep % self.upload_freq == 0:
                        self.upload_actor()
                        self.last_upload_total_steps = total_steps
                        self.shared_data.set_should_download(True)

                if self.writer is not None and train_info is not None:
                    elapsed_after_warmup = max(
                        time.time() - self.start_time_after_warmup, 1e-6
                    )
                    data_steps = max(total_steps - self.random_steps, 1)
                    sps = data_steps / elapsed_after_warmup
                    bps = self.Bstep / elapsed_after_warmup
                    utd = (self.Bstep * self.batch_size) / data_steps
                    replay_usage = buffer_size / max(self.buffer_capacity, 1)

                    self.writer.add_scalar("Loss/Q", train_info["q_loss"], total_steps)
                    self.writer.add_scalar(
                        "Loss/Actor", train_info["actor_loss"], total_steps
                    )
                    self.writer.add_scalar(
                        "Alpha/value", train_info["alpha"], total_steps
                    )
                    self.writer.add_scalar(
                        "Policy/Entropy", train_info["entropy"], total_steps
                    )
                    self.writer.add_scalar("Buffer/Size", buffer_size, total_steps)
                    self.writer.add_scalar("Perf/SPS", sps, total_steps)
                    self.writer.add_scalar("Perf/BPS", bps, total_steps)
                    self.writer.add_scalar("Perf/UTD", utd, total_steps)
                    self.writer.add_scalar(
                        "Perf/ReplayUsage", replay_usage, total_steps
                    )
                    self.writer.add_scalar(
                        "Perf/PolicyLagSteps",
                        total_steps - self.last_upload_total_steps,
                        total_steps,
                    )

                if total_steps - last_perf_log_steps >= 5000 or last_perf_log_steps < 0:
                    elapsed_after_warmup = max(
                        time.time() - self.start_time_after_warmup, 1e-6
                    )
                    data_steps = max(total_steps - self.random_steps, 1)
                    sps = data_steps / elapsed_after_warmup
                    bps = self.Bstep / elapsed_after_warmup
                    utd = (self.Bstep * self.batch_size) / data_steps
                    replay_usage = buffer_size / max(self.buffer_capacity, 1)
                    print(
                        f"(SAC Learner) steps={total_steps / 1e3:.1f}k | SPS={sps:.1f} | "
                        f"BPS={bps:.1f} | UTD={utd:.2f} | Replay={replay_usage * 100:.1f}%"
                    )
                    last_perf_log_steps = total_steps

                if self.save_interval > 0 and total_steps % self.save_interval == 0:
                    self.save(total_steps)

                if (
                    self.eval_interval > 0
                    and total_steps - last_eval_total_steps >= self.eval_interval
                ):
                    test_ep_steps, test_ep_r, test_arrival_rate = self.evaluate(
                        deterministic=True, turns=self.eval_turns
                    )
                    print(
                        f"Eval@{total_steps}: ArrivalRate:{test_arrival_rate}, Reward:{test_ep_r}, Steps:{test_ep_steps}"
                    )
                    if self.writer is not None:
                        self.writer.add_scalar(
                            "Eval/ArrivalRate", test_arrival_rate, total_steps
                        )
                        self.writer.add_scalar("Eval/Reward", test_ep_r, total_steps)
                        self.writer.add_scalar("Eval/Steps", test_ep_steps, total_steps)
                        # compatibility tags for quick filtering
                        self.writer.add_scalar(
                            "arrival_rate", test_arrival_rate, total_steps
                        )
                        self.writer.add_scalar("ep_r", test_ep_r, total_steps)
                        self.writer.add_scalar("ep_steps", test_ep_steps, total_steps)
                    last_eval_total_steps = total_steps

                last_trained_total_steps = total_steps
            else:
                time.sleep(0.01)

        self.save(total_steps)
        if self.writer is not None:
            self.writer.close()
        self.eval_env.close()
        print("---------------- SAC Learner Finished ----------------")

    def evaluate(self, deterministic=True, turns=20):
        envs = self.eval_env
        step_collector, total_steps = torch.zeros(envs.N, device=envs.dvc), 0
        r_collector, total_r = torch.zeros(envs.N, device=envs.dvc), 0
        arrived, finished = 0, 0

        s, info = envs.reset()
        while finished < turns:
            a = self.select_action(s, deterministic).to(envs.dvc)
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

            finished += int(dones.sum().item())
            arrived += int(wins.sum().item())

        return (
            int(total_steps.item() / finished),
            round(total_r.item() / finished, 2),
            round(arrived / finished, 2),
        )

    def select_action(self, s, deterministic):
        with torch.no_grad():
            probs = self.actor(s)
            if deterministic:
                return probs.argmax(dim=-1)
            return torch.multinomial(probs, num_samples=1).squeeze(1)

    def train_step(self):
        batch = self.shared_data.sample(self.batch_size)
        if batch is None:
            return None

        s, a, r, s_next, dw = batch
        a = a.view(-1, 1).long()
        r = r.view(-1, 1)
        dw = dw.view(-1, 1)

        with torch.no_grad():
            next_probs = self.actor(s_next)
            next_log_probs = torch.log(next_probs + 1e-8)
            next_q1_all, next_q2_all = self.q_critic_target(s_next)
            min_next_q_all = torch.min(next_q1_all, next_q2_all)
            v_next = torch.sum(
                next_probs * (min_next_q_all - self.alpha * next_log_probs),
                dim=1,
                keepdim=True,
            )
            target_Q = r + (~dw) * self.gamma * v_next

        q1_all, q2_all = self.q_critic(s)
        q1, q2 = q1_all.gather(1, a), q2_all.gather(1, a)
        q_loss = F.mse_loss(q1, target_Q) + F.mse_loss(q2, target_Q)
        self.q_critic_optimizer.zero_grad()
        q_loss.backward()
        self.q_critic_optimizer.step()

        probs = self.actor(s)
        log_probs = torch.log(probs + 1e-8)
        with torch.no_grad():
            q1_all, q2_all = self.q_critic(s)
        min_q_all = torch.min(q1_all, q2_all)

        a_loss = torch.sum(
            probs * (self.alpha * log_probs - min_q_all), dim=1, keepdim=False
        )

        self.actor_optimizer.zero_grad()
        a_loss_mean = a_loss.mean()
        a_loss_mean.backward()
        self.actor_optimizer.step()

        if self.adaptive_alpha:
            with torch.no_grad():
                self.H_mean = -torch.sum(probs * log_probs, dim=1).mean()
            alpha_loss = self.log_alpha * (self.H_mean - self.target_entropy)
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self.alpha = self.log_alpha.exp().item()

        for param, target_param in zip(
            self.q_critic.parameters(), self.q_critic_target.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

        return {
            "q_loss": q_loss.item(),
            "actor_loss": a_loss_mean.item(),
            "alpha": float(self.alpha),
            "entropy": float(self.H_mean.item())
            if torch.is_tensor(self.H_mean)
            else float(self.H_mean),
        }

    def upload_actor(self):
        self.shared_data.set_actor_param(deepcopy(self.actor).cpu().state_dict())

    def save(self, total_steps):
        model_idx = int(total_steps / 1000)
        torch.save(self.actor.state_dict(), f"./model/sacd_actor_{model_idx}.pth")
        torch.save(self.q_critic.state_dict(), f"./model/sacd_critic_{model_idx}.pth")
