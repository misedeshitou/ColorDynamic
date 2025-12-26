import torch


class DWAController:
    def __init__(self, config=None, device="cuda"):
        self.config = config if config else self._DefaultConfig()
        self.dvc = device

        self._prepare_velocity_samples()

    class _DefaultConfig:
        def __init__(self):
            self.max_speed = 50.0
            self.min_speed = -50.0
            self.max_yawrate = 2.0
            self.max_accel = 2.0
            self.max_dyawrate = 2.0
            self.v_reso = 0.1
            self.yawrate_reso = 0.2
            self.dt = 0.1
            self.predict_time = 3.0
            self.goal_cost_gain = 1.0
            self.speed_cost_gain = 1.0
            self.obstacle_cost_gain = 1.0
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
        """
        x: 当前状态 [x, y, yaw, v, w]
        goal_tensor: (2,) 目标点 Tensor
        ob_tensor: (N, O*P, 2, 1) 原始障碍物数据
        """
        # 确保障碍物在正确的设备上
        current_obs = vec_static_obs_P_shaped[env_idx].squeeze(-1).float().to(self.dvc)
        print(current_obs.shape)
        goal_tensor = goal_tensor.to(self.dvc)

        x_state = torch.tensor(x, device=self.dvc).float()

        # 计算动态窗口 (Dynamic Window)
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

        # 筛选符合 DW 的速度样本
        v_mask = (self.all_samples[:, 0] >= dw[0]) & (self.all_samples[:, 0] <= dw[1])
        w_mask = (self.all_samples[:, 1] >= dw[2]) & (self.all_samples[:, 1] <= dw[3])
        samples = self.all_samples[v_mask & w_mask]  # (K, 2) K 为样本数

        # 批量预测轨迹 (Batch Trajectory Prediction)
        # 轨迹形状: (K, Steps, 3) 其中 3 是 [x, y, yaw]
        steps = int(self.config.predict_time / self.config.dt)
        t_range = torch.arange(1, steps + 1, device=self.dvc) * self.config.dt

        # 提取速度和角速度
        v = samples[:, 0:1]  # (K, 1)
        w = samples[:, 1:2]  # (K, 1)

        w_safe = torch.where(w.abs() < 1e-5, torch.ones_like(w) * 1e-5, w)

        current_yaw = x_state[2]

        # x 修正: 积分 v*cos(yaw)
        traj_x = x_state[0] + (v / w_safe) * (
            torch.sin(current_yaw + w_safe * t_range) - torch.sin(current_yaw)
        )

        # y 修正: 积分 v*(-sin(yaw))
        traj_y = x_state[1] + (v / w_safe) * (
            torch.cos(current_yaw + w_safe * t_range) - torch.cos(current_yaw)
        )

        traj_yaw = current_yaw + w_safe * t_range
        # -----------------------

        # 合并为 (K, Steps, 2) 用于代价计算
        all_trajectories = torch.stack([traj_x, traj_y], dim=-1)

        # 目标代价归一化：当前距离 / 最大可能距离 (对角线长度)
        max_dist = (self.config.window_size**2 + self.config.window_size**2) ** 0.5
        dist_to_goal = torch.norm(all_trajectories[:, -1, :] - goal_tensor, dim=1)
        norm_goal_cost = dist_to_goal / max_dist

        # 速度代价归一化：(最高速 - 当前速) / 最高速
        norm_speed_cost = (
            self.config.max_speed - samples[:, 0]
        ) / self.config.max_speed

        # 障碍物代价归一化：使用负指数衰减惩罚函数
        # 当 dist -> robot_radius 时，cost -> 1
        # 当 dist 很大时，cost -> 0
        # 检查是否有障碍物点
        if current_obs.shape[0] == 0:
            # 如果没有障碍物，代价全为 0，最小距离设为一个很大的值
            min_dists_per_traj = torch.ones(samples.shape[0], device=self.dvc) * 1e6
            print("No obstacles detected.")
        else:
            flat_trajs = all_trajectories.view(-1, 2)
            # 计算距离矩阵 (K*Steps, M)
            dist_matrix = torch.cdist(flat_trajs, current_obs)
            # 找到每条轨迹的最小距离 (K*Steps,) -> (K, Steps) -> (K,)
            min_dists_per_step = torch.min(dist_matrix, dim=1)[0]
            min_dists_per_traj = torch.min(
                min_dists_per_step.view(samples.shape[0], steps), dim=1
            )[0]
        norm_obs_cost = torch.exp(
            -(min_dists_per_traj - self.config.robot_radius)
            / (self.config.safe_dist / 3.0)
        )

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
        best_idx = torch.argmin(final_costs)

        best_u = samples[best_idx].cpu().numpy()
        # 返回最优轨迹用于可视化
        best_trajectory = all_trajectories[best_idx].cpu().numpy()

        return best_u, best_trajectory
