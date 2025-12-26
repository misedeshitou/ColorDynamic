import math

import numpy as np


class DWAController:
    def __init__(self, config=None):
        # 如果没有传入配置，则使用默认配置
        self.config = config if config else self._DefaultConfig()
        self.ob = np.empty((0, 2))  # 障碍物
        self.goal = np.array([0, 0])  # 目标点

    class _DefaultConfig:
        """默认参数类"""

        def __init__(self):
            self.max_speed = 5  # [m/s]  # 最大速度
            self.min_speed = -0.5  # [m/s]  # 最小速度，设置为不可以倒车
            # self.min_speed = -5.0  # [m/s]  # 最小速度，设置为可以倒车
            self.max_yawrate = 10.0  # [rad/s]  # 最大角速度
            self.max_accel = 2  # [m/ss]  # 最大加速度
            self.max_dyawrate = 2.0  # [rad/ss]  # 最大角加速度
            self.v_reso = 0.01  # [m/s]，速度分辨率
            self.yawrate_reso = 0.1  # [rad/s]，角速度分辨率
            self.dt = 0.1  # [s]  # 采样周期
            self.predict_time = 3.0  # [s]  # 向前预估三秒
            self.to_goal_cost_gain = 1.0  # 目标代价增益
            self.speed_cost_gain = 1.0  # 速度代价增益
            self.robot_radius = 0.09  # [m]  # 机器人半径

    def set_env(self, goal, ob):
        """设置环境信息：目标点和障碍物"""
        self.goal = np.array(goal)
        self.ob = np.array(ob)

    def _motion(self, x, u):
        """机器人运动学模型"""
        x[0] += u[0] * math.cos(x[2]) * self.config.dt
        # 适配y轴向下为正方向的坐标系
        x[1] += u[0] * (-math.sin(x[2])) * self.config.dt
        x[2] += u[1] * self.config.dt
        x[3] = u[0]
        x[4] = u[1]
        return x

    def _calc_dynamic_window(self, x):
        """计算动态窗口"""
        vs = [
            self.config.min_speed,
            self.config.max_speed,
            -self.config.max_yawrate,
            self.config.max_yawrate,
        ]

        vd = [
            x[3] - self.config.max_accel * self.config.dt,
            x[3] + self.config.max_accel * self.config.dt,
            x[4] - self.config.max_dyawrate * self.config.dt,
            x[4] + self.config.max_dyawrate * self.config.dt,
        ]

        vr = [
            max(vs[0], vd[0]),
            min(vs[1], vd[1]),
            max(vs[2], vd[2]),
            min(vs[3], vd[3]),
        ]
        return vr

    def _calc_trajectory(self, x_init, v, w):
        """预测轨迹"""
        x = np.array(x_init)
        trajectory = np.array(x)
        time = 0
        while time <= self.config.predict_time:
            x = self._motion(x, [v, w])
            trajectory = np.vstack((trajectory, x))
            time += self.config.dt
        return trajectory

    def _calc_to_goal_cost(self, trajectory):
        """目标距离代价"""
        dx = self.goal[0] - trajectory[-1, 0]
        dy = self.goal[1] - trajectory[-1, 1]
        goal_dis = math.sqrt(dx**2 + dy**2)
        return self.config.to_goal_cost_gain * goal_dis

    def _calc_obstacle_cost(self, trajectory):
        """障碍物距离代价"""
        min_r = float("inf")
        for pos in trajectory:
            for obs in self.ob:
                dx = pos[0] - obs[0]
                dy = pos[1] - obs[1]
                r = math.sqrt(dx**2 + dy**2)

                if r <= self.config.robot_radius:
                    return float("Inf")
                if min_r >= r:
                    min_r = r
        return 1.0 / min_r

    def plan(self, x):
        """
        主入口：根据当前状态 x 计算最优控制指令 u
        x: [x(m), y(m), yaw(rad), v(m/s), omega(rad/s)]
        """
        vr = self._calc_dynamic_window(x)
        min_cost = float("inf")
        best_u = [0.0, 0.0]
        best_trajectory = np.array([x])

        # 遍历速度空间
        for v in np.arange(vr[0], vr[1], self.config.v_reso):
            for w in np.arange(vr[2], vr[3], self.config.yawrate_reso):
                trajectory = self._calc_trajectory(x, v, w)

                # 计算总代价
                to_goal_cost = self._calc_to_goal_cost(trajectory)
                speed_cost = self.config.speed_cost_gain * (
                    self.config.max_speed - trajectory[-1, 3]
                )
                ob_cost = self._calc_obstacle_cost(trajectory)

                final_cost = to_goal_cost + speed_cost + ob_cost

                if min_cost >= final_cost:
                    min_cost = final_cost
                    best_u = [v, w]
                    best_trajectory = trajectory

        return best_u, best_trajectory

    def update_state(self, x, u):
        """根据指令更新机器人当前状态（执行动作）"""
        return self._motion(x, u)
