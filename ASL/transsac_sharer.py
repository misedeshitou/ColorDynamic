import time

import torch


class shared_data_transsac:
    """Shared data manager for TransSAC actor-learner architecture"""
    
    def __init__(self, opt):
        self.A_dvc = torch.device(opt.A_dvc)
        self.B_dvc = torch.device(opt.B_dvc)
        self.L_dvc = torch.device(opt.L_dvc)

        self.max_size = int(opt.buffersize / opt.N)
        self.state_dim = opt.state_dim
        self.T = opt.T  # Time window length for TransSAC
        self.N = opt.N
        self.ptr = 0
        self.size = 0
        self.full = False

        # shared replay buffer for TransSAC (stores time-windowed states)
        # Shape: (max_size, N, T, state_dim) for time-windowed states
        self.s = torch.zeros((self.max_size, opt.N, opt.T, opt.state_dim), device=self.B_dvc)
        self.a = torch.zeros(
            (self.max_size, opt.N, 1), dtype=torch.int64, device=self.B_dvc
        )
        self.r = torch.zeros((self.max_size, opt.N, 1), device=self.B_dvc)
        self.dw = torch.zeros(
            (self.max_size, opt.N, 1), dtype=torch.bool, device=self.B_dvc
        )

        # shared control fields
        self.actor_param = None
        self.total_steps = 0
        self.should_download = False

        # locks for thread-safe operations
        self.get_lock_time = 2e-4
        self.set_lock_time = 1e-4
        self.busy = [False, False]  # [actor_param, replay_buffer]

        print("TransSAC Sharer Started!")

    def add(self, tw_s, a, r, dw):
        """Add transition to buffer (tw_s is already time-windowed)"""
        self.set_lock(self.add_core, 1, (tw_s, a, r, dw))

    def add_core(self, trans):
        tw_s, a, r, dw = trans

        if self.A_dvc != self.B_dvc:
            tw_s = tw_s.to(self.B_dvc)
            a = a.to(self.B_dvc)
            r = r.to(self.B_dvc)
            dw = dw.to(self.B_dvc)

        self.s[self.ptr] = tw_s
        self.a[self.ptr] = a.unsqueeze(-1) if a.dim() == 1 else a
        self.r[self.ptr] = r.unsqueeze(-1) if r.dim() == 1 else r
        self.dw[self.ptr] = dw.unsqueeze(-1) if dw.dim() == 1 else dw

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
        if self.size == self.max_size:
            self.full = True

    def sample(self, batch_size):
        """Sample a batch from the replay buffer"""
        return self.get_lock(self.sample_core, 1, batch_size)

    def sample_core(self, batch_size):
        high = self.size if self.full else self.ptr
        if high <= 1:
            return None

        ind = torch.randint(low=0, high=high, size=(batch_size,), device=self.B_dvc)
        env_ind = torch.randint(
            low=0, high=self.N, size=(batch_size,), device=self.B_dvc
        )

        s_batch = self.s[ind, env_ind]  # (batch_size, T, state_dim)
        a_batch = self.a[ind, env_ind]  # (batch_size, 1)
        r_batch = self.r[ind, env_ind]  # (batch_size, 1)
        dw_batch = self.dw[ind, env_ind]  # (batch_size, 1)

        if self.B_dvc != self.L_dvc:
            return (
                s_batch.to(self.L_dvc),
                a_batch.to(self.L_dvc),
                r_batch.to(self.L_dvc),
                dw_batch.to(self.L_dvc),
            )

        return s_batch, a_batch, r_batch, dw_batch

    def get_buffer_size(self):
        return self.size * self.N

    # Actor parameter management
    def set_actor_param(self, param_dict):
        """Set actor parameters for actor to download"""
        self.set_lock(self._set_actor_param_core, 0, param_dict)

    def _set_actor_param_core(self, param_dict):
        self.actor_param = param_dict

    def get_actor_param(self):
        """Get actor parameters for actor to download"""
        return self.get_lock(self._get_actor_param_core, 0, None)

    def _get_actor_param_core(self, _):
        return self.actor_param

    # Total steps management
    def set_total_steps(self, steps):
        self.total_steps = steps

    def get_total_steps(self):
        return self.total_steps

    # Download flag management
    def set_should_download(self, flag):
        self.should_download = flag

    def get_should_download(self):
        return self.should_download

    # Lock mechanism for thread-safe access
    def set_lock(self, func, lock_idx, args):
        """Use lock for set operations"""
        while self.busy[lock_idx]:
            time.sleep(self.set_lock_time)
        self.busy[lock_idx] = True
        func(args)
        self.busy[lock_idx] = False

    def get_lock(self, func, lock_idx, args):
        """Use lock for get operations"""
        while self.busy[lock_idx]:
            time.sleep(self.get_lock_time)
        self.busy[lock_idx] = True
        result = func(args)
        self.busy[lock_idx] = False
        return result
