import os
import time
from copy import deepcopy

import torch
from torch.utils.tensorboard import SummaryWriter

from utils.TransSAC import TransSAC_agent


def transsac_learner_process(opt):
    learner = TransSACLearner(opt)
    learner.run()


class TransSACLearner:
    def __init__(self, opt):
        self.L_dvc = torch.device(opt.L_dvc)
        self.shared_data = opt.shared_data

        self.max_train_steps = opt.max_train_steps
        self.random_steps = opt.random_steps
        self.batch_size = opt.batch_size
        self.gamma = opt.gamma
        self.update_every = opt.update_every
        self.train_repeat = opt.train_repeat
        self.upload_freq = opt.upload_freq
        self.save_interval = opt.save_interval
        self.eval_interval = opt.eval_interval
        self.run_dir = getattr(opt, "run_dir", os.path.join("runs", "TransSAC_ASL"))
        self.model_dir = getattr(
            opt, "model_dir", os.path.join("model", "TransSAC_ASL")
        )
        os.makedirs(self.run_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)

        # Create agent
        self.agent = TransSAC_agent(**vars(opt))
        self.agent.dvc = self.L_dvc
        self.agent.actor = self.agent.actor.to(self.L_dvc)
        self.agent.q_critic = self.agent.q_critic.to(self.L_dvc)
        self.agent.q_critic_target = self.agent.q_critic_target.to(self.L_dvc)

        self.Bstep = 0
        self.last_upload_total_steps = 0
        self.start_time = time.time()

        # TensorBoard
        self.writer = SummaryWriter(log_dir=self.run_dir)
        self.writer.add_text("config", str(vars(opt)))

        # If resuming, try to load checkpoint and restore steps
        resume_k = getattr(opt, "resume_actor_kstep", None)
        if resume_k is not None:
            try:
                self.agent.load(resume_k, self.model_dir)
                # set learner Bstep according to loaded total steps if available
                loaded_steps = getattr(self.agent, "loaded_total_steps", 0)
                if loaded_steps:
                    self.Bstep = int(loaded_steps / 1000)
                    self.shared_data.set_total_steps(loaded_steps)
                    print(
                        f"(TransSAC Learner) Resumed from ckpt {resume_k}, total_steps={loaded_steps}"
                    )
                else:
                    # fallback to opt.initial_total_steps
                    if getattr(opt, "initial_total_steps", 0) > 0:
                        self.shared_data.set_total_steps(opt.initial_total_steps)
            except Exception as e:
                print(f"(TransSAC Learner) Failed to load resume ckpt: {e}")

        print("TransSAC Learner Started!")

    def run(self):
        warmup_done = False

        while True:
            total_steps = self.shared_data.get_total_steps()

            if total_steps > self.max_train_steps:
                print("---------------- TransSAC Learner Finished ----------------")
                self.writer.close()
                break

            # Start training after random steps
            if total_steps >= self.random_steps and not warmup_done:
                warmup_done = True
                print(
                    f"(TransSAC Learner) Warmup finished, starting training at step {total_steps}"
                )

            if warmup_done:
                # Sample from shared buffer and populate agent's replay buffer
                # This allows agent.train() to work normally
                batch_data = self.shared_data.sample(self.batch_size)

                if batch_data is not None:
                    s, a, r, dw = batch_data
                    # s shape: (batch_size, T, state_dim)
                    # a shape: (batch_size, 1)
                    # r shape: (batch_size, 1)
                    # dw shape: (batch_size, 1)

                    # Populate agent's replay buffer for training
                    # Note: shared buffer doesn't have s_next, so we approximate by shifting
                    s_next = (
                        s  # Approximate s_next as s (ideally would use next sample)
                    )

                    for i in range(min(self.batch_size, 1)):
                        self.agent.replay_buffer.add_batch(
                            s[i : i + 1].unsqueeze(0),  # Shape: (1, 1, T, state_dim)
                            a[i : i + 1],
                            r[i : i + 1],
                            s_next[i : i + 1].unsqueeze(0),
                            dw[i : i + 1],
                        )

                    # Train if we have enough data
                    if self.agent.replay_buffer.size > 0:
                        for _ in range(self.train_repeat):
                            train_info = self.agent.train()

                            self.Bstep += 1

                            if train_info is not None and self.Bstep % 100 == 0:
                                self.writer.add_scalar(
                                    "Loss/Q", train_info["q_loss"], self.Bstep
                                )
                                self.writer.add_scalar(
                                    "Loss/Actor", train_info["actor_loss"], self.Bstep
                                )
                                self.writer.add_scalar(
                                    "Alpha/value", train_info["alpha"], self.Bstep
                                )
                                self.writer.add_scalar(
                                    "Policy/Entropy", train_info["entropy"], self.Bstep
                                )

                    """Upload model every upload_freq batch steps"""
                    if self.Bstep % self.upload_freq == 0 and self.Bstep > 0:
                        if not self.shared_data.get_should_download():
                            actor_param = deepcopy(self.agent.actor.state_dict())
                            self.shared_data.set_actor_param(actor_param)
                            self.shared_data.set_should_download(True)
                            print(
                                f"(TransSAC Learner) Uploaded model at Bstep {self.Bstep}"
                            )

                    """Save checkpoint"""
                    if self.Bstep % self.save_interval == 0 and self.Bstep > 0:
                        self.agent.save(int(self.Bstep / 1000))
                        print(
                            f"(TransSAC Learner) Saved checkpoint at Bstep {self.Bstep}"
                        )

            time.sleep(1e-3)  # Avoid busy waiting
