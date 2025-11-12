'''
This file contains the implementation of the OCMF_V2G and OCMF_G2V MPC

Authors: Cesar Diaz-Londono, Stavros Orfanoudakis
'''

import gurobipy as gp
from gurobipy import GRB
from gurobipy import *
import numpy as np
import time

from mpc import MPC


class OCMF_V2G(MPC):
    """
    经济型 MPC（含 V2G）：
    - 决策变量 u：对每个充电口在控制时域内的 充/放电功率（偶数位充电、奇数位放电）
    - 通过二进制变量 Zbin 保证同一端口同一时刻“充电/放电”二选一
    - 目标函数：电价 * 功率 * Δt，充电付费、放电收入，并配合 CapF1（容量分流变量）构造线性化的“互斥”结构
    - 含变压器（配变）功率约束和站内不等式约束
    """

    def __init__(self, env, control_horizon=10, verbose=False, **kwargs):
        """
        初始化 MPC 基线控制器

        Args:
            env: 环境（应提供站点/车/配变等信息）
            control_horizon: 控制时域长度（滚动优化一次考虑多少步）
            verbose: 是否打印调试信息
        """
        super().__init__(env, control_horizon, verbose)

        # na: 充电端口数量；nb: 决策维度 = 2*na（每个端口有充/放两个功率位）
        self.na = self.n_ports
        self.nb = 2 * self.na

    def get_action(self, env):
        """
        计算含 V2G 的经济 MPC 动作（归一化到 [-1,1]，正=充电，负=放电）
        """
        t = env.current_step  # 当前时刻索引

        # 根据当前时刻更新配变功率上限（可能随时间/温度/运维策略变化）
        self.update_tr_power(t)

        # 从环境重构下一步状态向量 self.x_next（SOC 等）
        self.reconstruct_state(t)
        # 计算状态转移矩阵/向量（含 V2G 情况），用于后续约束拼装
        self.calculate_XF_V2G(t)

        # 站级模型（单调模型），构建 Amono/Bmono 等
        self.v2g_station_models(t)

        # 计算完整的不等式约束 Au, bu（含能量演化、端口限值等）
        self.calculate_InequalityConstraints(t)

        # 根据设备能力/在桩车辆/安全线等设置功率上界 UB
        self.set_power_limits_V2G(t)

        if self.verbose:
            self.print_info(t)

        # === 目标函数系数 ===
        # f 作用于 u（功率）；f2 作用于 CapF1（容量分流变量，用于互斥线性化）
        f = []
        f2 = []
        for i in range(self.control_horizon):
            for j in range(self.n_ports):
                # 偶数位：充电功率项（成本：买电）
                f.append(self.T * self.ch_prices[t + i])
                # 奇数位：放电功率项（负号表示卖电收益会降低目标值）
                f.append(-self.T * self.disch_prices[t + i])

                # CapF1 的配重（常见做法是配合大 M 技巧控制）
                f2.append(self.T * self.ch_prices[t + i])
                # 对放电的 CapF1 加大权重（此处 *2 是原实现的经验参数）
                f2.append(self.T * self.disch_prices[t + i]*2)

        f = np.array(f).reshape(-1)
        f2 = np.array(f2).reshape(-1)

        nb = self.nb  # 2*n_ports
        n = self.n_ports
        h = self.control_horizon

        # === 建模 ===
        model = gp.Model("optimization_model")

        # 连续变量：功率向量 u（维度 nb*h）
        u = model.addMVar(nb*h, vtype=GRB.CONTINUOUS, name="u")

        # 连续变量：CapF1（容量分流辅助变量，用于构造互斥）
        CapF1 = model.addMVar(nb*h, vtype=GRB.CONTINUOUS, name="CapF1")

        # 二进制变量：Zbin（每个端口每个时刻 1 表示充电，0 表示放电）
        Zbin = model.addMVar(n*h, vtype=GRB.BINARY, name="Zbin")

        # === 约束 ===

        # 1) 通用不等式约束（能量演化、端口/电池容量、站内线路等），AU u <= bU
        model.addConstr((self.AU @ u)  <= self.bU, name="constr1")

        # 2) CapF1 下/上界（0 <= CapF1 <= UB）
        model.addConstr((0 <= CapF1), name="constr2a")
        model.addConstr((CapF1 <= self.UB), name="constr2b")

        # 3) 充电功率位（偶数位 j）与 Zbin 的关系
        # 3a) CapF1[j] <= u[j]（保证 CapF1 不超过对应功率位）
        model.addConstrs((CapF1[j] <= u[j]
                          for j in range(0, nb*h, 2)), name="constr3a")

        # 3b) u[j] <= (UB[j]-CapF1[j]) * Zbin[j//2]
        #     若 Zbin=0，则右侧为0，迫使充电位 u[j] <= 0（通常结合其他约束使其为0）
        model.addConstrs((u[j] <= (self.UB[j]-CapF1[j]) * Zbin[j//2]
                          for j in range(0, nb*h, 2)), name="constr3b")

        # 4) 放电功率位（奇数位 j）与 (1-Zbin) 的关系
        # 4a) CapF1[j] <= u[j]
        model.addConstrs((CapF1[j] <= u[j]
                          for j in range(1, nb*h, 2)),
                         name="constr4a")

        # 4b) u[j] <= (UB[j]-CapF1[j]) * (1 - Zbin[j//2])
        #     若 Zbin=1（在充电），则放电位被压到 0
        model.addConstrs((u[j] <= (self.UB[j]-CapF1[j])*(1-Zbin[j//2])
                          for j in range(1, nb*h, 2)),
                         name="constr4b")

        # 5) 配变功率上/下限：∑(端口功率) + 站内其他负荷/光伏 ≤ 上限
        #    注意：此处端口功率用 (u[充]-u[放]) 聚合得到“净负荷”
        for tr_index in range(self.number_of_transformers):
            for i in range(self.control_horizon):
                model.addConstr((gp.quicksum((u[j] - u[j+1])
                                             for index, j in enumerate(
                                                 range(i*self.nb, (i+1)*self.nb, 2))
                                             if self.cs_transformers[index] == tr_index) +
                                 self.tr_loads[tr_index, i] +
                                 self.tr_pv[tr_index, i] <=
                                 self.tr_power_limit[tr_index, i]),
                                name=f'constr5_{tr_index}_t{i}')
        # 下限（允许双向功率流），用 -max 上限近似
        for tr_index in range(self.number_of_transformers):
            for i in range(self.control_horizon):
                model.addConstr((gp.quicksum((u[j] - u[j+1])
                                             for index, j in enumerate(
                                                 range(i*self.nb, (i+1)*self.nb, 2))
                                             if self.cs_transformers[index] == tr_index) +
                                 self.tr_loads[tr_index, i] +
                                 self.tr_pv[tr_index, i] >=
                                 -self.tr_power_limit[tr_index, :].max()),
                                name=f'constr5_{tr_index}_t{i}')

        # === 目标函数 ===
        # 最小化： fᵀu - f2ᵀCapF1
        # 直觉：u（功率）乘电价，充电产生成本，放电产生负成本（收益）
        # CapF1 作为互斥线性化的“惩罚/奖励”项，配合约束限制同一时刻的充放电互斥
        model.setObjective(f @ u - f2 @ CapF1, GRB.MINIMIZE)

        # Gurobi 参数
        model.setParam('OutputFlag', self.output_flag)
        model.params.NonConvex = 2  # 含二进制/互斥结构

        if self.MIPGap is not None:
            model.params.MIPGap = self.MIPGap
        model.params.TimeLimit = self.time_limit

        model.optimize()
        self.total_exec_time += model.Runtime

        # （注意：这里原代码又设置了一次并再次 optimize，可能是无意重复）
        if self.MIPGap is not None:
            model.params.MIPGap = self.MIPGap
        model.params.TimeLimit = self.time_limit
        model.optimize()

        # 不可行时，返回默认 0 动作（全 0 功率）
        if model.status == GRB.Status.INF_OR_UNBD or \
                model.status == GRB.Status.INFEASIBLE:
            print(f"INFEASIBLE (applying default actions) - step{t} !!!")
            actions = np.ones(self.n_ports) * 0  # 也可设一个安全保底功率
            return actions

        # 读取优化结果
        a = np.zeros((nb*h, 1))
        cap = np.zeros((nb*h, 1))
        z_bin = np.zeros((n*h, 1))

        for i in range(nb*h):
            a[i] = u[i].x
            # cap[i] = CapF1[i].x
            # z_bin[i//2] 可按需读取

        # 构造环境需要的归一化动作（逐端口，正=充、负=放）
        actions = np.zeros(self.n_ports)
        e = 0.001  # 数值阈值，避免浮点误判
        for i in range(0, 2*self.n_ports, 2):
            if a[i] > e and a[i + 1] > e:
                # 理论上被互斥约束禁止，此处做防御性检查
                raise ValueError(f'Charging and discharging at the same time\
                                    {i} {a[i]} {a[i+1]}')
            elif a[i] > e:
                # 充电：按端口最大充电功率归一
                actions[i//2] = a[i]/self.max_ch_power[i//2]
            elif a[i] > -e and a[i + 1] > e:
                # 放电：按端口最大放电功率归一（注意取绝对值）
                actions[i//2] = -a[i+1]/abs(self.max_disch_power[i//2])
            else:
                # 近似 0
                actions[i//2] = 0.0

        if self.verbose:
            print(f'actions: {actions.shape} \n {actions}')

        return actions


class OCMF_G2V(MPC):
    '''
    仅 G2V 的经济 MPC（无放电）：
    - 决策变量 u：每端口在控制时域内的充电功率
    - 无二进制互斥（只有充电），约束更简单
    '''

    def __init__(self, env, control_horizon=10, verbose=False, **kwargs):
        """
        初始化 MPC 基线控制器

        Args:
            env: 环境
            control_horizon: 控制时域长度
            verbose: 是否打印调试信息
        """
        super().__init__(env, control_horizon, verbose)

        self.na = self.n_ports
        self.nb = self.na  # 仅充电功率维度

    def get_action(self, env):
        """
        计算仅 G2V 的经济 MPC 动作（归一化为 [0,1]）
        """
        t = env.current_step

        # 更新配变功率上限
        self.update_tr_power(t)

        # 重构状态，计算状态转移
        self.reconstruct_state(t)
        self.calculate_XF_G2V(t)

        # 站级模型（仅充电）
        self.g2v_station_models(t)

        # 计算不等式约束 AU u <= bU
        self.calculate_InequalityConstraints(t)

        # 设置功率上界 UB（仅充电）
        self.set_power_limits_G2V(t)

        if self.verbose:
            self.print_info(t)

        # === 目标函数系数（买电成本） ===
        f = []
        for i in range(self.control_horizon):
            for j in range(self.n_ports):
                f.append(self.T * self.ch_prices[t + i])
        f = np.array(f).reshape(-1)
        if self.verbose:
            print(f'f: {f.shape}')

        nb = self.nb
        h = self.control_horizon

        # === 建模 ===
        model = gp.Model("optimization_model")

        # 连续变量：充电功率 u
        u = model.addMVar(nb*h, vtype=GRB.CONTINUOUS, name="u")

        # 连续变量：CapF1（辅助，构造线性界限）
        CapF1 = model.addMVar(nb*h, vtype=GRB.CONTINUOUS, name="CapF1")

        # === 约束 ===

        # 1) 通用不等式约束
        model.addConstr((self.AU @ u)  <= self.bU, name="constr1")

        # 2) 0 <= CapF1 <= UB
        model.addConstr((0 <= CapF1), name="constr2a")
        model.addConstr((CapF1 <= self.UB), name="constr2b")

        # 3) 充电位互斥辅助（这里无放电，仅用来线性刻画余量）
        model.addConstr((CapF1 <= u), name="constr3a")
        model.addConstr((u <= (self.UB-CapF1)), name="constr3b")

        # 4) 配变功率约束（仅正向功率：∑u + 其他负荷/光伏 ≤ 上限）
        for tr_index in range(self.number_of_transformers):
            for i in range(self.control_horizon):
                model.addConstr((gp.quicksum(u[j]
                                             for index, j in enumerate(
                                                 range(i*self.nb, (i+1)*self.nb))
                                             if self.cs_transformers[index] == tr_index) +
                                 self.tr_loads[tr_index, i] +
                                 self.tr_pv[tr_index, i] <=
                                 self.tr_power_limit[tr_index, i]),
                                name=f'constr5_{tr_index}_t{i}')
        # 下限（允许一定回馈为负），用 -max 上限近似
        for tr_index in range(self.number_of_transformers):
            for i in range(self.control_horizon):
                model.addConstr((gp.quicksum(u[j]
                                             for index, j in enumerate(
                                                 range(i*self.nb, (i+1)*self.nb))
                                             if self.cs_transformers[index] == tr_index) +
                                 self.tr_loads[tr_index, i] +
                                 self.tr_pv[tr_index, i] >=
                                 -self.tr_power_limit[tr_index, :].max()),
                                name=f'constr5_{tr_index}_t{i}')

        # === 目标函数 ===
        # 最小化： fᵀu - fᵀCapF1
        # 注：CapF1 的系数与 u 相同，等价于“尽量减少有效可用容量被占用”，与上界线性化配合
        model.setObjective(f @ u - f @ CapF1, GRB.MINIMIZE)

        # Gurobi 参数
        model.setParam('OutputFlag', self.output_flag)
        model.params.NonConvex = 2

        if self.MIPGap is not None:
            model.params.MIPGap = self.MIPGap
        model.params.TimeLimit = self.time_limit
        model.optimize()
        self.total_exec_time += model.Runtime

        # 不可行 fallback
        if model.status == GRB.Status.INF_OR_UNBD or \
                model.status == GRB.Status.INFEASIBLE:
            print(f"INFEASIBLE (applying default actions) - step{t} !!!")
            actions = np.ones(self.n_ports) * 0  # 可设置一个小功率保底
            return actions

        # 读取结果（这里只取第一个时段的动作）
        a = np.zeros((nb*h, 1))
        # cap = np.zeros((nb*h, 1))

        for i in range(self.n_ports):
            a[i] = u[i].x
            # cap[i] = CapF1[i].x

        if self.verbose:
            print(f'Actions:\n {a.reshape(-1,self.n_ports)}')
            # print(f'CapF1:\n {cap.reshape(-1,self.n_ports)}')

        # 构造归一化动作（[0,1]）
        actions = np.zeros(self.n_ports)
        for i in range(self.n_ports):
            # 注意：这里用 max_ch_power[i//2] 可能不严谨（若每端口一对一，应为 max_ch_power[i]）
            # 保持原实现不改动，仅加提示
            actions[i] = a[i] / self.max_ch_power[i//2]

        if self.verbose:
            print(f'actions: {actions.shape} \n {actions}')

        return actions
