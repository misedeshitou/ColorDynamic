import numpy as np
import torch


class DWAController:
    def __init__(self, config=None, device="cuda"):
        self.config = config if config else self._DefaultConfig()
        self.dvc = device

        self._prepare_velocity_samples()

    class _DefaultConfig:
        def __init__(self):
            self.max_speed = 50.0  # cm/s
            self.min_speed = -50.0
            self.max_yawrate = 2.0
            self.max_accel = 5.0
            self.max_dyawrate = 2.0
            self.v_reso = 0.1  # m/s 速度分辨率
            self.yawrate_reso = 0.2  # rad/s 角速度分辨率
            self.dt = 0.1  # 采样周期
            self.predict_time = 1.0  # 预测时间
            self.goal_cost_gain = 0.6
            self.speed_cost_gain = 0.1
            self.obstacle_cost_gain = 0.3
            self.robot_radius = 5.0
            self.window_size = 800  # 地图尺寸800cm x 800cm
            self.safe_dist = 20.0  # 预警半径 cm

    def _prepare_velocity_samples(self):
        v_range = torch.arange(
            self.config.min_speed,
            self.config.max_speed,
            self.config.v_reso,
            device=self.dvc,
        )
        w_range = torch.arange(
            -self.config.max_yawrate,
            self.config.max_yawrate,
            self.config.yawrate_reso,
            device=self.dvc,
        )

        # 生成网格组合 (Num_Samples, 2) -> [v, w]
        grid_v, grid_w = torch.meshgrid(v_range, w_range, indexing="ij")
        self.all_samples = torch.stack([grid_v.flatten(), grid_w.flatten()], dim=1)

    def plan(self, x, goal_tensor, vec_static_obs_P_shaped, env_idx=0):
        # 1. 数据准备
        current_obs = vec_static_obs_P_shaped[env_idx].squeeze(-1).float().to(self.dvc)
        goal_tensor = goal_tensor.to(self.dvc)
        x_state = torch.tensor(x, device=self.dvc).float()

        # 2. 动态窗口 (Dynamic Window) 筛选
        dw = [
            max(self.config.min_speed, x[3] - self.config.max_accel * self.config.dt),
            min(self.config.max_speed, x[3] + self.config.max_accel * self.config.dt),
            max(
                -self.config.max_yawrate,
                x[4] - self.config.max_dyawrate * self.config.dt,
            ),
            min(
                self.config.max_yawrate,
                x[4] + self.config.max_dyawrate * self.config.dt,
            ),
        ]
        v_mask = (self.all_samples[:, 0] >= dw[0]) & (self.all_samples[:, 0] <= dw[1])
        w_mask = (self.all_samples[:, 1] >= dw[2]) & (self.all_samples[:, 1] <= dw[3])
        samples = self.all_samples[v_mask & w_mask]

        if samples.shape[0] == 0:
            return np.array([0.0, 0.0]), x_state[:2].cpu().numpy()

        # 3. 圆弧积分法代替直线积分预测轨迹
        eval_dt = 0.05
        steps = int(self.config.predict_time / eval_dt)
        t_range = torch.arange(1, steps + 1, device=self.dvc) * eval_dt

        v, w = samples[:, 0:1], samples[:, 1:2]
        print(v, w)
        w_safe = torch.where(w.abs() < 1e-5, torch.ones_like(w) * 1e-5, w)
        current_yaw = x_state[2]

        #  x 修正: 积分 v*cos(yaw)
        traj_x = x_state[0] + (v / w_safe) * (
            torch.sin(current_yaw + w_safe * t_range) - torch.sin(current_yaw)
        )

        # y 修正: 积分 v*(-sin(yaw))
        traj_y = x_state[1] + (v / w_safe) * (
            torch.cos(current_yaw + w_safe * t_range) - torch.cos(current_yaw)
        )

        traj_yaw = current_yaw + w_safe * t_range
        all_trajectories = torch.stack([traj_x, traj_y], dim=-1)  # (K, Steps, 2)

        # 4. 代价计算 (无量纲化)
        # (a) 目标代价: 归一化到 [0, 1]
        max_map_dist = (self.config.window_size**2 * 2) ** 0.5
        dist_to_goal = torch.norm(all_trajectories[:, -1, :] - goal_tensor, dim=1)
        norm_goal_cost = dist_to_goal / max_map_dist

        # (b) 速度代价: 鼓励接近最大速度 [0, 1]
        norm_speed_cost = (
            self.config.max_speed - samples[:, 0]
        ) / self.config.max_speed

        # (c) 障碍物代价: 指数惩罚 + 硬碰撞保护
        if current_obs.shape[0] == 0:
            norm_obs_cost = torch.zeros(samples.shape[0], device=self.dvc)
        else:
            flat_trajs = all_trajectories.view(-1, 2)
            dist_matrix = torch.cdist(flat_trajs, current_obs)
            min_dists_per_step = torch.min(dist_matrix, dim=1)[0]
            min_dists_per_traj = torch.min(
                min_dists_per_step.view(samples.shape[0], steps), dim=1
            )[0]

            # 核心保护逻辑
            net_dist = min_dists_per_traj - self.config.robot_radius

            # 指数衰减 (safe_dist/3 确保在 safe_dist 处代价约 0.05)
            norm_obs_cost = torch.exp(
                -torch.clamp(net_dist, min=0) / (self.config.safe_dist / 3.0)
            )

            # --- 穿模与硬碰撞保护 ---
            # 如果净距离小于 2cm（即将撞上或已穿模），强制代价设为极大值，压过 goal_cost
            norm_obs_cost[net_dist <= 15.0] = 5.0

        # 5. 决策
        final_costs = (
            self.config.goal_cost_gain * norm_goal_cost
            + self.config.speed_cost_gain * norm_speed_cost
            + self.config.obstacle_cost_gain * norm_obs_cost
        )
        print(
            self.config.goal_cost_gain * norm_goal_cost,
            self.config.speed_cost_gain * norm_speed_cost,
            self.config.obstacle_cost_gain * norm_obs_cost,
        )
        # 找到最优动作索引
        best_idx = torch.argmin(final_costs)

        # 修改判断条件：看这个最优动作的障碍物代价是否触发了“硬保护”
        # 注意：5.0 是你赋的值，0.3 是权重
        if norm_obs_cost[best_idx] >= 5.0:
            best_u = np.array([0.0, 0.0])  # 发现最优解也会撞，急停
        else:
            best_u = samples[best_idx].cpu().numpy()

        return best_u, all_trajectories[best_idx].cpu().numpy()
