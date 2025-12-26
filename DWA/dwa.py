import math

import matplotlib.pyplot as plt
import numpy as np

show_animation = True  # 动画


class Config(object):
    """
    用来仿真的参数，
    """

    def __init__(self):
        # robot parameter
        self.max_speed = 1.4  # [m/s]  # 最大速度
        # self.min_speed = -0.5  # [m/s]  # 最小速度，设置为可以倒车
        self.min_speed = 0  # [m/s]  # 最小速度，设置为不倒车
        self.max_yawrate = 40.0 * math.pi / 180.0  # [rad/s]  # 最大角速度
        self.max_accel = 0.2  # [m/ss]  # 最大加速度
        self.max_dyawrate = 40.0 * math.pi / 180.0  # [rad/ss]  # 最大角加速度
        self.v_reso = 0.01  # [m/s]，速度分辨率
        self.yawrate_reso = 0.1 * math.pi / 180.0  # [rad/s]，角速度分辨率
        self.dt = 0.1  # [s]  # 采样周期
        self.predict_time = 3.0  # [s]  # 向前预估三秒
        self.to_goal_cost_gain = 1.0  # 目标代价增益
        self.speed_cost_gain = 1.0  # 速度代价增益
        self.robot_radius = 1.0  # [m]  # 机器人半径


# 机器人运动方程
# 在dt的时间间隔内，基于当前状态x、动作指令u=[v,w]，转移到下一个状态x，
# 两个状态更新机器人的位置(x,y)，以及其超向orientation。
def motion(x, u, dt):
    """
    :param x: 位置参数，在此叫做位置空间
    :param u: 速度和加速度，在此叫做速度空间
    :param dt: 采样时间
    :return:
    """
    # 假设近似，机器人在dt的时间间隔内，默认为速度与转向速度（v,w）是常数
    # 速度更新公式比较简单，在极短时间内，车辆位移也变化较大
    # 采用圆弧求解如何？
    x[0] += u[0] * math.cos(x[2]) * dt  # x方向位移
    x[1] += u[0] * math.sin(x[2]) * dt  # y方向位移
    x[2] += u[1] * dt  # 航向角
    x[3] = u[0]  # 速度v
    x[4] = u[1]  # 角速度w
    # print(x)

    return x


def calc_dynamic_window(x, config):
    """
    位置空间集合
    :param x:当前位置空间,符号参考硕士论文
    :param config:
    :return:目前是两个速度的交集，还差一个
    """

    # 车辆能够达到的最大最小速度
    vs = [config.min_speed, config.max_speed, -config.max_yawrate, config.max_yawrate]

    # 一个采样周期能够变化的最大最小速度
    vd = [
        x[3] - config.max_accel * config.dt,
        x[3] + config.max_accel * config.dt,
        x[4] - config.max_dyawrate * config.dt,
        x[4] + config.max_dyawrate * config.dt,
    ]
    #  print(Vs, Vd)

    # 求出两个速度集合的交集
    vr = [max(vs[0], vd[0]), min(vs[1], vd[1]), max(vs[2], vd[2]), min(vs[3], vd[3])]

    return vr


# 产生运动轨迹
def calc_trajectory(x_init, v, w, config):
    """
    预测3秒内的轨迹
    :param x_init:位置空间
    :param v:速度
    :param w:角速度
    :param config:
    :return: 每一次采样更新的轨迹，位置空间垂直堆叠
    """
    x = np.array(x_init)
    trajectory = np.array(x)
    time = 0
    while time <= config.predict_time:
        x = motion(x, [v, w], config.dt)
        trajectory = np.vstack((trajectory, x))  # 垂直堆叠，vertical
        time += config.dt

        # print(trajectory)
    return trajectory


def calc_to_goal_cost(trajectory, goal, config):
    """
    计算轨迹到目标点的代价
    :param trajectory:轨迹搜索空间
    :param goal:
    :param config:
    :return: 轨迹到目标点欧式距离
    """
    # calc to goal cost. It is 2D norm.

    dx = goal[0] - trajectory[-1, 0]
    dy = goal[1] - trajectory[-1, 1]
    goal_dis = math.sqrt(dx**2 + dy**2)
    cost = config.to_goal_cost_gain * goal_dis

    return cost


def calc_obstacle_cost(traj, ob, config):
    """
    计算预测轨迹和障碍物的最小距离，dist(v,w)
    :param traj:
    :param ob:
    :param config:
    :return:
    """
    # calc obstacle cost inf: collision, 0:free

    min_r = float("inf")  # 距离初始化为无穷大

    for ii in range(0, len(traj[:, 1])):
        for i in range(len(ob[:, 0])):
            ox = ob[i, 0]
            oy = ob[i, 1]
            dx = traj[ii, 0] - ox
            dy = traj[ii, 1] - oy

            r = math.sqrt(dx**2 + dy**2)
            if r <= config.robot_radius:
                return float("Inf")  # collision

            if min_r >= r:
                min_r = r

    return 1.0 / min_r  # 越小越好 与障碍物距离越近，该值会越大；


def calc_final_input(x, u, vr, config, goal, ob):
    """
    计算采样空间的评价函数，选择最合适的那一个作为最终输入
    :param x:位置空间
    :param u:速度空间
    :param vr:速度空间交集
    :param config:
    :param goal:目标位置
    :param ob:障碍物
    :return:
    """
    x_init = x[:]
    min_cost = 10000.0
    min_u = u

    best_trajectory = np.array([x])

    # evaluate all trajectory with sampled input in dynamic window
    # v,生成一系列速度，w，生成一系列角速度
    for v in np.arange(vr[0], vr[1], config.v_reso):  # 对搜索空间dw中的v遍历
        for w in np.arange(vr[2], vr[3], config.yawrate_reso):  # 对搜索空间dw中的w遍历
            # 计算出每一个可能的动作在未来一段时间内所产生的轨迹trajectory(v,w)
            trajectory = calc_trajectory(x_init, v, w, config)

            # calc cost
            to_goal_cost = calc_to_goal_cost(
                trajectory, goal, config
            )  # trajectory与目标的欧式距离
            speed_cost = config.speed_cost_gain * (
                config.max_speed - trajectory[-1, 3]
            )  # v越大，该项越小，则效果越好；
            ob_cost = calc_obstacle_cost(
                trajectory, ob, config
            )  # 返回的是距离的倒数，所以该值约小，距离越大，越好
            #  print(ob_cost)

            # 评价函数多种多样，看自己选择
            # 本文构造的是越小越好
            final_cost = to_goal_cost + speed_cost + ob_cost  # 总计代价，越小轨迹约好

            # search minimum trajectory
            # 在所有动态窗口划出的动作空间(v,w)里，找到一个最好的动作，在这个动作下，未来预测的轨迹评价最好
            if min_cost >= final_cost:
                min_cost = final_cost
                min_u = [v, w]  # 代价最小的时候的运动空间
                best_trajectory = trajectory  # 记此时预测轨迹为最优轨迹

    # print(min_u)
    #  input()

    return min_u, best_trajectory


def dwa_control(x, u, config, goal, ob):
    """
    调用前面的几个函数，生成最合适的速度空间和轨迹搜索空间
    :param x: 起始位置
    :param u:初始速度空间（v,w）
    :param config:
    :param goal:目标位置
    :param ob:障碍物
    :return:
    """
    # Dynamic Window control

    vr = calc_dynamic_window(x, config)  # 计算动态窗口，即机器人的电机实际物理参数范围

    u, trajectory = calc_final_input(
        x, u, vr, config, goal, ob
    )  # 选出最优轨迹搜索空间和速度空间

    return u, trajectory


def plot_arrow(x, y, yaw, length=0.5, width=0.1):
    """
    arrow函数绘制箭头
    :param x:
    :param y:
    :param yaw:航向角
    :param length:
    :param width:参数值为浮点数，代表箭头尾部的宽度，默认值为0.001
    :return:
    length_includes_head：代表箭头整体长度是否包含箭头头部的长度，默认值为False
    head_width：代表箭头头部的宽度，默认值为3*width，即尾部宽度的3倍
    head_length：代表箭头头部的长度度，默认值为1.5*head_width，即头部宽度的1.5倍
    shape：参数值为'full'、'left'、'right'，表示箭头的形状，默认值为'full'
    overhang：代表箭头头部三角形底边与箭头尾部直接的夹角关系，通过该参数可改变箭头的形状。
    默认值为0，即头部为三角形，当该值小于0时，头部为菱形，当值大于0时，头部为鱼尾状
    """
    plt.arrow(
        x,
        y,
        length * math.cos(yaw),
        length * math.sin(yaw),
        head_length=1.5 * width,
        head_width=width,
    )
    plt.plot(x, y)


def main():
    """
    主函数
    :return:
    """
    # print(__file__ + " start!!")
    # 初始化位置空间
    x = np.array([0.0, 0.0, math.pi / 2.0, 0.2, 0.0])

    goal = np.array([10, 10])

    # matrix二维矩阵 障碍物
    ob = np.matrix(
        [
            [-1, -1],
            [0, 2],
            [4.0, 2.0],
            [5.0, 4.0],
            [5.0, 5.0],
            [5.0, 6.0],
            [5.0, 9.0],
            [8.0, 9.0],
            [7.0, 9.0],
            [12.0, 12.0],
        ]
    )
    # ob = np.matrix([[0, 2]])
    # ob = np.matrix([[2, 2]])

    u = np.array([0.2, 0.0])  # 初始速度空间(v,w)
    config = Config()  # 初始化物理运动参数
    trajectory = np.array(x)  # 初始轨迹空间

    for i in range(1000):
        u, best_trajectory = dwa_control(
            x, u, config, goal, ob
        )  # 获取最优速度空间和轨迹参数

        x = motion(x, u, config.dt)  # 机器人运动方程 x为机器人状态空间
        print(x)

        trajectory = np.vstack((trajectory, x))  # store state history

        if show_animation:
            draw_dynamic_search(best_trajectory, x, goal, ob)

        # check goal
        if (
            math.sqrt((x[0] - goal[0]) ** 2 + (x[1] - goal[1]) ** 2)
            <= config.robot_radius
        ):
            print("Goal!!")

            break

    print("Done")

    draw_path(trajectory, goal, ob, x)


def draw_dynamic_search(best_trajectory, x, goal, ob):
    """
    画出动态搜索过程图
    :return:
    """
    plt.cla()  # 清除上次绘制图像
    plt.plot(best_trajectory[:, 0], best_trajectory[:, 1], "-g")
    plt.plot(x[0], x[1], "xr")
    plt.plot(0, 0, "og")
    plt.plot(goal[0], goal[1], "ro")
    plt.plot(ob[:, 0], ob[:, 1], "bs")
    plot_arrow(x[0], x[1], x[2])
    plt.axis("equal")
    plt.grid(True)
    plt.pause(0.0001)


def draw_path(trajectory, goal, ob, x):
    """
    画图函数
    :return:
    """
    plt.cla()  # 清除上次绘制图像

    plt.plot(x[0], x[1], "xr")
    plt.plot(0, 0, "og")
    plt.plot(goal[0], goal[1], "ro")
    plt.plot(ob[:, 0], ob[:, 1], "bs")
    plot_arrow(x[0], x[1], x[2])
    plt.axis("equal")
    plt.grid(True)
    plt.plot(trajectory[:, 0], trajectory[:, 1], "r")
    plt.show()


if __name__ == "__main__":
    main()
