import time

import torch


class shared_data_sac:
    def __init__(self, opt):
        self.A_dvc = torch.device(opt.A_dvc)
        self.B_dvc = torch.device(opt.B_dvc)
        self.L_dvc = torch.device(opt.L_dvc)

        self.max_size = int(opt.buffersize / opt.N)
        self.state_dim = opt.state_dim
        self.N = opt.N
        self.ptr = 0
        self.size = 0
        self.full = False

        # shared replay buffer (vectorized by env axis N)
        self.s = torch.zeros((self.max_size, opt.N, opt.state_dim), device=self.B_dvc)
        self.a = torch.zeros(
            (self.max_size, opt.N, 1), dtype=torch.int64, device=self.B_dvc
        )
        self.r = torch.zeros((self.max_size, opt.N, 1), device=self.B_dvc)
        self.s_next = torch.zeros(
            (self.max_size, opt.N, opt.state_dim), device=self.B_dvc
        )
        self.dw = torch.zeros(
            (self.max_size, opt.N, 1), dtype=torch.bool, device=self.B_dvc
        )

        # shared control fields
        self.actor_param = None
        self.total_steps = 0
        self.should_download = False

        # locks
        self.get_lock_time = 2e-4
        self.set_lock_time = 1e-4
        self.busy = [False, False]  # [actor_param, replay_buffer]

        print("SAC Sharer Started!")

    def add(self, s, a, r, s_next, dw):
        self.set_lock(self.add_core, 1, (s, a, r, s_next, dw))

    def add_core(self, trans):
        s, a, r, s_next, dw = trans

        if self.A_dvc != self.B_dvc:
            s = s.to(self.B_dvc)
            a = a.to(self.B_dvc)
            r = r.to(self.B_dvc)
            s_next = s_next.to(self.B_dvc)
            dw = dw.to(self.B_dvc)

        self.s[self.ptr] = s
        self.a[self.ptr] = a.unsqueeze(-1)
        self.r[self.ptr] = r.unsqueeze(-1)
        self.s_next[self.ptr] = s_next
        self.dw[self.ptr] = dw.unsqueeze(-1)

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
        if self.size == self.max_size:
            self.full = True

    def sample(self, batch_size):
        return self.get_lock(self.sample_core, 1, batch_size)

    def sample_core(self, batch_size):
        high = self.size if self.full else self.ptr
        if high <= 1:
            return None

        ind = torch.randint(low=0, high=high, size=(batch_size,), device=self.B_dvc)
        env_ind = torch.randint(
            low=0, high=self.N, size=(batch_size,), device=self.B_dvc
        )

        if self.B_dvc != self.L_dvc:
            return (
                self.s[ind, env_ind].to(self.L_dvc),
                self.a[ind, env_ind].to(self.L_dvc),
                self.r[ind, env_ind].to(self.L_dvc),
                self.s_next[ind, env_ind].to(self.L_dvc),
                self.dw[ind, env_ind].to(self.L_dvc),
            )

        return (
            self.s[ind, env_ind],
            self.a[ind, env_ind],
            self.r[ind, env_ind],
            self.s_next[ind, env_ind],
            self.dw[ind, env_ind],
        )

    def get_buffer_size(self):
        return self.size * self.N

    def get_actor_param(self):
        return self.get_lock(self.get_actor_param_core, 0)

    def get_actor_param_core(self):
        return self.actor_param

    def set_actor_param(self, actor_param):
        self.set_lock(self.set_actor_param_core, 0, actor_param)

    def set_actor_param_core(self, actor_param):
        self.actor_param = actor_param

    def get_total_steps(self):
        return self.total_steps

    def set_total_steps(self, total_steps):
        self.total_steps = total_steps

    def get_should_download(self):
        return self.should_download

    def set_should_download(self, bol):
        self.should_download = bol

    def get_lock(self, get_func, idx, *args):
        while True:
            if self.busy[idx]:
                time.sleep(self.get_lock_time)
            else:
                time.sleep(self.get_lock_time)
                if not self.busy[idx]:
                    self.busy[idx] = True
                    data = get_func(*args)
                    self.busy[idx] = False
                    return data

    def set_lock(self, set_func, idx, *args):
        while True:
            if self.busy[idx]:
                time.sleep(self.set_lock_time)
            else:
                time.sleep(self.set_lock_time)
                if not self.busy[idx]:
                    self.busy[idx] = True
                    set_func(*args)
                    self.busy[idx] = False
                    break
