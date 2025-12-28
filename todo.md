[DWA]
1. 静态只有奖励的DWA (done)
    一个崭新的物体能够碰到障碍物——将动作嵌入sparrow(done)
2. 带刷新的DWA(done)
3. 加静态障碍物(done)
4. 加动态障碍物（done）
5. current_a 与real_a之间的延时（unnecessary）
6. 快碰到障碍物时惩罚无限大保证一定不碰到（double edged）
7. 增加朝向奖励（useless）

[DQN]
1. 跑的比较慢
2. 障碍物密集时停滞不前
3. 还是在打圈
-> 增大训练轮数后消失

[其他]
1. TD3？
2. evaluate

# def evaluate(envs, agent, deterministic, turns):
#     step_collector, total_steps = torch.zeros(opt.N, device=opt.dvc), 0
#     r_collector, total_r = torch.zeros(opt.N, device=opt.dvc), 0
#     arrived, finished = 0, 0

#     agent.queue.clear()
#     s, info = envs.reset()
#     ct = torch.ones(opt.N, device=opt.dvc, dtype=torch.bool)
#     while finished < turns:
#         """单步state -> 时序窗口state:"""
#         agent.queue.append(s)  # 将s加入时序窗口队列
#         TW_s = agent.queue.get()  # 取出队列所有数据及
#         a = agent.select_action(TW_s, deterministic)
#         s, r, dw, tr, info = envs.step(a)

#         """解析dones, wins, deads, truncateds, consistents信号："""
#         agent.queue.padding_with_done(~ct)  # 根据上一时刻的ct去padding
#         dones = dw + tr
#         wins = r == envs.AWARD
#         dead_and_tr = dones ^ wins  # dones-wins = deads and truncateds
#         ct = ~dones

#         """统计回合步数："""
#         step_collector += 1
#         total_steps += step_collector[wins].sum()  # 到达,总步数加上真实步数
#         total_steps += (
#             envs.max_ep_steps * dead_and_tr
#         ).sum()  # 未到达,总步数加上回合最大步数
#         step_collector[dones] = 0

#         """统计总奖励："""
#         r_collector += r
#         total_r += r_collector[dones].sum()
#         r_collector[dones] = 0

#         """统计到达率："""
#         finished += dones.sum()
#         arrived += wins.sum()

#     return (
#         int(total_steps.item() / finished.item()),
#         round(total_r.item() / finished.item(), 2),
#         round(arrived.item() / finished.item(), 2),
#     )