'''
This file contains the eMPC class, which is used to control the ev2gym environment using the eMPC algorithm.

Authors: Cesar Diaz-Londono, Stavros Orfanoudakis
'''

import gurobipy as gp
from gurobipy import GRB
from gurobipy import *
import numpy as np

from ev2gym.baselines.mpc.mpc import MPC


class eMPC_V2G(MPC):
    """
    经济型 MPC（含 V2G）的一个实现版本（与 OCMF_* 思路相似，但更简洁）：
    - 决策变量 u（连续）：每端口在每步的 充/放电功率（偶数位=充电、奇数位=放电）
    - 决策变量 Zbin（二进制）：每端口每步充/放电互斥（1=充电，0=放电）
    - 目标函数：最小化 电费（充电付费，放电收益用负价进入）
    - 约束：站内不等式约束 AU u <= bU、端口功率与互斥关系、配变功率上下限等
    - 额外机制：若不可行，会“放松”终点 SoC 约束（降低离站 SoC 目标），直到可行
    """

    def __init__(self, env, control_horizon=10, verbose=False, **kwargs):
        """
        初始化 MPC 控制器

        Args:
            env: 环境对象（包含充电站/车辆/电价/配变等信息与接口）
            control_horizon: 控制时域长度（每次优化考虑未来多少步）
            verbose: 是否打印调试信息
        """
        super().__init__(env, control_horizon, verbose)

        self.na = self.n_ports           # 端口数
        self.nb = 2 * self.na            # 决策维度（每端口有充/放两个功率位）

    def get_action(self, env):
        """
        计算含 V2G 的动作（返回归一化功率：正=充电、负=放电）
        """

        # 当问题不可行时，逐步放松 SoC 终点要求，直到求得可行解
        while True:
            t = env.current_step

            # 更新配变功率上限（随时间可能变化）
            self.update_tr_power(t)

            # 重构下一步状态（SoC 等）并組装状态方程（含 V2G）
            self.reconstruct_state(t)
            self.calculate_XF_V2G(t)

            # 站级模型（单调模型），组装单站约束
            self.v2g_station_models(t)

            # 拼装完整不等式约束 AU, bU（能量演化、端口限值等）
            self.calculate_InequalityConstraints(t)

            # 设置功率上界 UB（由设备/车辆/安全线决定）
            self.set_power_limits_V2G(t)

            if self.verbose:
                self.print_info(t)

            # === 目标函数系数（充电成本、放电收益）===
            f = []
            for i in range(self.control_horizon):
                for j in range(self.n_ports):
                    # 偶数位（充电）：成本 +ch_price
                    f.append(self.T * self.ch_prices[t + i])
                    # 奇数位（放电）：收益用负价进入（-disch_price）
                    f.append(-self.T * self.disch_prices[t + i])
            f = np.array(f).reshape(-1)

            nb = self.nb
            n  = self.n_ports
            h  = self.control_horizon

            # === 建立 Gurobi 模型 ===
            model = gp.Model("optimization_model")

            # 连续变量：功率 u（长度 nb*h）
            u = model.addMVar(nb * h, vtype=GRB.CONTINUOUS, name="u")

            # 二进制变量：充/放互斥，长度 n*h，对应每端口每步
            Zbin = model.addMVar(n * h, vtype=GRB.BINARY, name="Zbin")

            # 额外变量（此版本里未进入目标）：SoC 轨迹、平均 SoC、循环/日历老化度
            # 预留出来以便拓展“健康成本”等
            SoC   = model.addMVar(n * h,              vtype=GRB.CONTINUOUS, name="SoC")
            SOCav = model.addMVar(self.EV_number,     vtype=GRB.CONTINUOUS, name="SOCav")
            d_cyc = model.addMVar(self.EV_number,     vtype=GRB.CONTINUOUS, name="d_cyc")
            d_cal = model.addMVar(self.EV_number,     vtype=GRB.CONTINUOUS, name="d_cal")

            # === 约束 ===

            # 1) 通用不等式约束（能量演化 / 站内限制等）
            model.addConstr((self.AU @ u) <= self.bU, name="constr1")

            # 2) 充电位（偶数索引）非负且受 Zbin 控制：
            #    若 Zbin=1 => 可充电：u[j] <= UB[j]
            #    若 Zbin=0 => 不充电：u[j] <= 0
            model.addConstrs((0 <= u[j] for j in range(0, nb*h, 2)), name="constr3a")
            model.addConstrs((u[j] <= self.UB[j] * Zbin[j//2]
                              for j in range(0, nb*h, 2)), name="constr3b")

            # 3) 放电位（奇数索引）非负且与 (1 - Zbin) 绑定：
            #    若 Zbin=0 => 可放电：u[j] <= UB[j]
            #    若 Zbin=1 => 不放电：u[j] <= 0
            model.addConstrs((0 <= u[j] for j in range(1, nb*h, 2)), name="constr4a")
            model.addConstrs((u[j] <= self.UB[j] * (1 - Zbin[j//2])
                              for j in range(1, nb*h, 2)), name="constr4b")

            # 4) 配变功率约束（上下限）：∑(充-放) + 其他负荷/光伏 ≤ 上限；≥ 下限
            for tr_index in range(self.number_of_transformers):
                for i in range(self.control_horizon):
                    model.addConstr((gp.quicksum((u[j] - u[j+1])
                                    for index, j in enumerate(range(i*self.nb, (i+1)*self.nb, 2))
                                    if self.cs_transformers[index] == tr_index)
                                    + self.tr_loads[tr_index, i] + self.tr_pv[tr_index, i]
                                    <= self.tr_power_limit[tr_index, i]),
                                    name=f'constr5_{tr_index}_t{i}')
            for tr_index in range(self.number_of_transformers):
                for i in range(self.control_horizon):
                    model.addConstr((gp.quicksum((u[j] - u[j+1])
                                    for index, j in enumerate(range(i*self.nb, (i+1)*self.nb, 2))
                                    if self.cs_transformers[index] == tr_index)
                                    + self.tr_loads[tr_index, i] + self.tr_pv[tr_index, i]
                                    >= -self.tr_power_limit[tr_index, :].max()),
                                    name=f'constr5_{tr_index}_t{i}')

            # === 目标：最小化电费（充电成本 - 放电收益）===
            model.setObjective(f @ u, GRB.MINIMIZE)
            model.setParam('OutputFlag', self.output_flag)
            # 此处未设置 NonConvex=2，因为模型线性 + MIP（Zbin），默认即可

            if self.MIPGap is not None:
                model.params.MIPGap = self.MIPGap
            model.params.TimeLimit = self.time_limit

            model.optimize()
            self.total_exec_time += model.Runtime

            # 若不可行/无界：放松离站 SoC 约束并重试
            if model.status == GRB.Status.INF_OR_UNBD or model.status == GRB.Status.INFEASIBLE:
                print(f"INFEASIBLE or Unbounded - step{t} -- Relaxing SoC constraints - try {self.varch2}")
                flagOut = False
                varch = 0

                # 遍历每辆 EV，在 t~t+h 内寻找“离站时刻”，下调其离站 SoC 目标
                for i in range(n):
                    for j in range(t, t + h + 1):
                        # 检测 j 为离站边界：x_final 在 j 为 0，j-1 为正
                        if self.x_final[i, j] == 0 and self.x_final[i, j - 1] > 0:
                            if self.verbose:
                                print(f"EV {i} is departing at {j} with {self.x_final[i, j - 1]}")
                                print(f'XFinal: {self.x_final[i, j]} ')
                                print(f'XNext: {self.x_next[i]}')
                                print(f'Diff: {self.x_final[i, j - 1] - self.p_max_MT[i, j - 1] * self.T}')

                        # 若确有离站且当前 SoC 需求较紧，尝试按最大功率可供给量下调一档
                        if self.x_final[i, j] == 0 and self.x_final[i, j - 1] > 0 and self.x_next[i] > 0:
                            varch += 1
                            if varch > self.varch2:
                                # 将离站前一刻的 SoC 目标降低一格（p_max_MT * T）
                                self.x_final[i, j - 1] = self.x_final[i, j - 1] - self.p_max_MT[i, j - 1] * self.T
                                self.varch2 += 1
                                flagOut = True
                                break
                    if flagOut:
                        break
                    if i == n - 1:
                        # 一轮未触发放松，重置计数
                        self.varch2 = 0
                # 回到 while True 继续求解
                continue

            # === 读取解并构建动作 ===
            a = np.zeros((nb * h, 1))
            for i in range(2 * self.n_ports):
                a[i] = u[i].x

            # 归一化动作（每端口：正=充、负=放；对最大充/放功率归一）
            actions = np.zeros(self.n_ports)
            if self.verbose:
                print(f'Actions:\n {a.reshape(-1, self.n_ports, 2)}')

            e = 1e-3
            for i in range(0, 2*self.n_ports, 2):
                if a[i] > e and a[i + 1] > e:
                    # 理论上被互斥约束禁止，防御性检查
                    raise ValueError(f'Charging and discharging at the same time {i} {a[i]} {a[i+1]}')
                elif a[i] > e:
                    actions[i//2] = a[i] / self.max_ch_power[i//2]
                elif a[i + 1] > e:
                    actions[i//2] = -a[i+1] / abs(self.max_disch_power[i//2])
                else:
                    actions[i//2] = 0.0

            if self.verbose:
                print(f'actions: {actions.shape} \n {actions}')

            return actions


class eMPC_G2V(MPC):
    '''
    仅 G2V 的经济 MPC 实现：
    - 决策变量 u（连续）：每端口在每步的充电功率
    - 目标函数：最小化购电成本
    - 约束：AU u <= bU、0 <= u <= UB、配变功率上下限
    '''

    def __init__(self, env, control_horizon=10, verbose=False, **kwargs):
        """
        初始化 MPC 控制器

        Args:
            env: 环境对象
            control_horizon: 控制时域长度
            verbose: 是否打印调试信息
        """
        super().__init__(env, control_horizon, verbose)

        self.na = self.n_ports  # 端口数
        self.nb = self.na       # 决策维度（仅充电）

    def get_action(self, env):
        """
        计算仅 G2V 的动作（返回 [0,1] 归一化充电功率）
        """
        t = env.current_step

        # 更新配变上限、重构状态、组装单站模型与不等式约束
        self.update_tr_power(t)
        self.reconstruct_state(t)
        self.calculate_XF_G2V(t)
        self.g2v_station_models(t)
        self.calculate_InequalityConstraints(t)
        self.set_power_limits_G2V(t)

        if self.verbose:
            self.print_info(t)

        # === 目标函数系数（购电成本）===
        f = []
        for i in range(self.control_horizon):
            for j in range(self.n_ports):
                f.append(self.T * self.ch_prices[t + i])
        f = np.array(f).reshape(-1)

        nb = self.nb
        n  = self.n_ports
        h  = self.control_horizon

        # === 建立模型 ===
        model = gp.Model("optimization_model")

        # 连续变量：充电功率 u
        u = model.addMVar(nb * h, vtype=GRB.CONTINUOUS, name="u")

        # 约束：AU u <= bU
        model.addConstr((self.AU @ u) <= self.bU, name="constr1")

        # 约束：0 <= u <= UB
        model.addConstr((0 <= u),   name="constr2a")
        model.addConstr((u <= self.UB), name="constr2b")

        # 配变功率上/下限：∑u + 其他负荷/光伏 ≤ 上限；≥ 下限
        for tr_index in range(self.number_of_transformers):
            for i in range(self.control_horizon):
                model.addConstr((gp.quicksum(u[j]
                                 for index, j in enumerate(range(i*self.nb, (i+1)*self.nb))
                                 if self.cs_transformers[index] == tr_index)
                                 + self.tr_loads[tr_index, i] + self.tr_pv[tr_index, i]
                                 <= self.tr_power_limit[tr_index, i]),
                                 name=f'constr5_{tr_index}_t{i}')
        for tr_index in range(self.number_of_transformers):
            for i in range(self.control_horizon):
                model.addConstr((gp.quicksum(u[j]
                                 for index, j in enumerate(range(i*self.nb, (i+1)*self.nb))
                                 if self.cs_transformers[index] == tr_index)
                                 + self.tr_loads[tr_index, i] + self.tr_pv[tr_index, i]
                                 >= -self.tr_power_limit[tr_index, :].max()),
                                 name=f'constr5_{tr_index}_t{i}')

        # 目标：最小化购电成本
        model.setObjective(f @ u, GRB.MINIMIZE)
        model.setParam('OutputFlag', self.output_flag)
        model.params.NonConvex = 2  # 这里保留原实现（虽为线性模型，加了也不影响）

        if self.MIPGap is not None:
            model.params.MIPGap = self.MIPGap
        model.params.TimeLimit = self.time_limit

        model.optimize()
        self.total_exec_time += model.Runtime

        # 不可行则返回零动作（也可在此加入类似 V2G 的 SoC 放松策略）
        if model.status == GRB.Status.INF_OR_UNBD or model.status == GRB.Status.INFEASIBLE:
            print(f"INFEASIBLE (applying default actions) - step{t} !!!")
            actions = np.ones(self.n_ports) * 0  # 0.25 可做保底
            return actions

        # 读取当前步动作（常见做法是取第一个时段的决策）
        a = np.zeros((nb * h, 1))
        for i in range(self.n_ports):
            a[i] = u[i].x

        if self.verbose:
            print(f'Actions:\n {a.reshape(-1, self.n_ports)}')

        # 归一化输出（[0,1]）
        actions = np.zeros(self.n_ports)
        for i in range(self.n_ports):
            # 注：若 max_ch_power 是“逐端口”的数组，理论上应使用 self.max_ch_power[i]
            # 这里保持原实现 self.max_ch_power[i//2] 不改动，只在注释中提示
            actions[i] = a[i] / self.max_ch_power[i//2]

        if self.verbose:
            print(f'actions: {actions.shape} \n {actions}')

        return actions
