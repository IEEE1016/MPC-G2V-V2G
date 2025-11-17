import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import matplotlib.patches as mpatches
import yaml
import gurobipy as gp
from gurobipy import GRB
from ev2gym.models.ev2gym_env import EV2Gym
import os


class SOCCurvePlotter:
    def __init__(self, battery_capacity=50, config_file='V2G_MPC.yaml'):
        """
        初始化SOC曲线绘制器
        :param battery_capacity: 电池容量(kWh)
        :param config_file: 配置文件路径
        """
        self.battery_capacity = battery_capacity
        self.time_hours = np.linspace(0, 24, 97)  # 0-24小时，15分钟一个采样点
        self.time_step = 0.25  # 小时
        
        # 从配置文件读取参数
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 充放电参数
        self.max_charge_power = config['ev']['max_ac_charge_power']  # 22 kW
        self.max_discharge_power = abs(config['ev']['max_discharge_power'])  # 22 kW
        self.charge_efficiency = config['ev']['charge_efficiency']  # 1.0
        self.discharge_efficiency = config['ev']['discharge_efficiency']  # 1.0
        
        # 场景参数
        self.arrival_time = 8  # 8:00 到达
        self.departure_time = 16  # 16:00 离开
        self.initial_soc = 0.5  # 初始SOC 50%
        self.target_soc = config['ev']['desired_capacity']  # 0.8
        self.min_soc = config['ev']['min_emergency_battery_capacity'] / battery_capacity  # 最低SOC
        
        # 从EV2Gym环境读取真实电价数据
        print("正在从环境加载电价数据...")
        self.charge_prices, self.discharge_prices = self._load_electricity_prices(config_file)
        self.discharge_price_factor = config['discharge_price_factor']  # 1.2
        
        # 控制参数
        self.control_horizon = 25  # 预测时域
    
    def _load_electricity_prices(self, config_file):
        """从EV2Gym环境加载真实电价数据"""
        env = EV2Gym(
            config_file=config_file,
            verbose=False,
            save_replay=False,
            save_plots=False,
        )
        env.reset()
        
        # 提取价格数据(96个时间步)
        ch_prices = np.abs(env.charge_prices[0, :96])
        disch_prices = np.abs(env.discharge_prices[0, :96])
        
        # 扩展到97个点(包含24:00)
        ch_prices = np.append(ch_prices, ch_prices[-1])
        disch_prices = np.append(disch_prices, disch_prices[-1])
        
        print(f"电价加载成功: 充电 [{ch_prices.min():.4f}, {ch_prices.max():.4f}] €/kWh")
        print(f"          放电 [{disch_prices.min():.4f}, {disch_prices.max():.4f}] €/kWh")
        
        return ch_prices, disch_prices
    
    def _get_time_window_indices(self):
        """获取充电时间窗口的索引(从22:00到次日7:00)"""
        arrival_idx = int(self.arrival_time / self.time_step)  # 22:00 -> 88
        departure_idx = int(self.departure_time / self.time_step)  # 7:00 -> 28
        
        return arrival_idx, departure_idx
    
    def power_to_soc(self, power_curve, initial_soc=None):
        """
        将功率曲线转换为SOC曲线
        :param power_curve: 功率数组(kW)，正为充电，负为放电
        :param initial_soc: 初始SOC(0-1)
        :return: SOC数组
        """
        if initial_soc is None:
            initial_soc = self.initial_soc
            
        soc = np.zeros_like(power_curve)
        soc[0] = initial_soc
        
        for i in range(1, len(power_curve)):
            if power_curve[i] > 0:  # 充电
                efficiency = self.charge_efficiency
            elif power_curve[i] < 0:  # 放电
                efficiency = self.discharge_efficiency
            else:
                efficiency = 1.0
            
            energy_change = power_curve[i] * self.time_step * efficiency / self.battery_capacity
            soc[i] = np.clip(soc[i-1] + energy_change, 0, 1)
        
        return soc
    
    def simulate_empc_g2v(self):
        """使用eMPC算法模拟G2V场景(仅充电)"""
        power_trajectory = np.zeros(97)
        soc_trajectory = np.full(97, np.nan)  # 用NaN表示未接入
        
        arrival_idx, departure_idx = self._get_time_window_indices()  # 88, 28
        
        # 接入瞬间(22:00)SOC变为初始值
        soc_trajectory[arrival_idx] = self.initial_soc
        
        # 从arrival到departure充电(同一天内,包含departure时刻)
        print(f"  充电阶段: t={arrival_idx} ({self.time_hours[arrival_idx]:.1f}h) -> t={departure_idx} ({self.time_hours[departure_idx]:.1f}h)")
        for t in range(arrival_idx, departure_idx + 1):
            current_soc = soc_trajectory[t]
            remaining_steps = departure_idx + 1 - t
            horizon = min(self.control_horizon, remaining_steps)
            
            if horizon <= 1:
                break
            
            power = self._solve_empc_g2v_step(current_soc, t, horizon, remaining_steps)
            power_trajectory[t] = power
            
            # 更新下一时刻SOC
            energy_change = power * self.time_step * self.charge_efficiency / self.battery_capacity
            new_soc = np.clip(current_soc + energy_change, 0, 1)
            if t + 1 < 97:
                soc_trajectory[t + 1] = new_soc
        
        # departure时刻确保达到目标SOC
        final_soc = soc_trajectory[departure_idx]
        if final_soc < self.target_soc - 0.01:
            print(f"  警告: 未达到目标SOC! 实际={final_soc:.3f}, 目标={self.target_soc:.3f}")
        
        # departure后(7:00-22:00)EV离开,充电桩无SOC显示
        # 这些位置保持NaN
        
        print(f"  最终SOC: {final_soc:.3f} (目标: {self.target_soc:.3f})")
        return power_trajectory, soc_trajectory
    
    def _solve_empc_g2v_step(self, current_soc, t, horizon, remaining_steps):
        """求解单步eMPC-G2V优化问题"""
        try:
            model = gp.Model("eMPC_G2V")
            model.setParam('OutputFlag', 0)
            model.setParam('TimeLimit', 10)
            
            # 决策变量: 充电功率序列
            u = model.addVars(horizon, lb=0, ub=self.max_charge_power, name="u")
            soc = model.addVars(horizon + 1, lb=0, ub=1, name="soc")
            
            # 初始SOC约束
            model.addConstr(soc[0] == current_soc)
            
            # SOC演化约束
            for k in range(horizon):
                energy_change = u[k] * self.time_step * self.charge_efficiency / self.battery_capacity
                model.addConstr(soc[k + 1] == soc[k] + energy_change)
            
            # 关键修改:如果是最后几步,必须达到目标SOC
            if remaining_steps <= horizon:
                # 必须在离开前达到目标SOC
                model.addConstr(soc[remaining_steps] >= self.target_soc)
                model.addConstr(soc[remaining_steps] <= self.target_soc)
            else:
                # 中间步骤,逐步接近目标
                model.addConstr(soc[horizon] >= current_soc)  # 至少不降低
            
            # 目标函数: 最小化充电成本
            obj = gp.quicksum(self.charge_prices[(t + k) % 96] * u[k] * self.time_step 
                            for k in range(horizon))
            model.setObjective(obj, GRB.MINIMIZE)
            
            model.optimize()
            
            if model.status == GRB.OPTIMAL:
                return u[0].X
            else:
                # 不可行时使用最大功率充电以确保达标
                return self.max_charge_power
                
        except Exception as e:
            print(f"eMPC-G2V优化失败 t={t}: {e}")
            return self.max_charge_power
    
    def simulate_empc_v2g(self):
        """使用eMPC算法模拟V2G场景(允许充放电)"""
        power_trajectory = np.zeros(97)
        soc_trajectory = np.full(97, np.nan)
        
        arrival_idx, departure_idx = self._get_time_window_indices()
        
        # 接入瞬间SOC变为初始值
        soc_trajectory[arrival_idx] = self.initial_soc
        
        # 从arrival到departure(包含departure时刻)
        print(f"  V2G充放电阶段: t={arrival_idx} -> t={departure_idx}")
        for t in range(arrival_idx, departure_idx + 1):
            current_soc = soc_trajectory[t]
            remaining_steps = departure_idx + 1 - t
            horizon = min(self.control_horizon, remaining_steps)
            
            if horizon <= 1:
                break
            
            power = self._solve_empc_v2g_step(current_soc, t, horizon, remaining_steps)
            power_trajectory[t] = power
            
            if power > 0:
                efficiency = self.charge_efficiency
            else:
                efficiency = self.discharge_efficiency
            energy_change = power * self.time_step * efficiency / self.battery_capacity
            new_soc = np.clip(current_soc + energy_change, self.min_soc, 1)
            if t + 1 < 97:
                soc_trajectory[t + 1] = new_soc
        
        final_soc = soc_trajectory[departure_idx]
        if final_soc < self.target_soc - 0.01:
            print(f"  警告: 未达到目标SOC! 实际={final_soc:.3f}, 目标={self.target_soc:.3f}")
        
        # departure后保持NaN
        
        print(f"  最终SOC: {final_soc:.3f}")
        return power_trajectory, soc_trajectory
    
    def _solve_empc_v2g_step(self, current_soc, t, horizon, remaining_steps):
        """求解单步eMPC-V2G优化问题(可充可放)"""
        try:
            model = gp.Model("eMPC_V2G")
            model.setParam('OutputFlag', 0)
            model.setParam('TimeLimit', 10)
            
            # 决策变量
            u_charge = model.addVars(horizon, lb=0, ub=self.max_charge_power, name="u_ch")
            u_discharge = model.addVars(horizon, lb=0, ub=self.max_discharge_power, name="u_dis")
            z = model.addVars(horizon, vtype=GRB.BINARY, name="z")  # 1=充电, 0=放电
            soc = model.addVars(horizon + 1, lb=self.min_soc, ub=1, name="soc")
            
            # 初始SOC
            model.addConstr(soc[0] == current_soc)
            
            # 互斥约束
            for k in range(horizon):
                model.addConstr(u_charge[k] <= self.max_charge_power * z[k])
                model.addConstr(u_discharge[k] <= self.max_discharge_power * (1 - z[k]))
            
            # SOC演化
            for k in range(horizon):
                charge_energy = u_charge[k] * self.time_step * self.charge_efficiency / self.battery_capacity
                discharge_energy = u_discharge[k] * self.time_step * self.discharge_efficiency / self.battery_capacity
                model.addConstr(soc[k + 1] == soc[k] + charge_energy - discharge_energy)
            
            # 终端约束:必须达到目标SOC
            if remaining_steps <= horizon:
                model.addConstr(soc[remaining_steps] >= self.target_soc)
                model.addConstr(soc[remaining_steps] <= self.target_soc)
            else:
                model.addConstr(soc[horizon] >= self.min_soc + 0.1)  # 保持安全余量
            
            # 目标函数:最小化总成本(充电成本-放电收益)
            obj = gp.quicksum(
                self.charge_prices[(t + k) % 96] * u_charge[k] * self.time_step -
                self.discharge_prices[(t + k) % 96] * u_discharge[k] * self.time_step
                for k in range(horizon)
            )
            model.setObjective(obj, GRB.MINIMIZE)
            
            model.optimize()
            
            if model.status == GRB.OPTIMAL:
                return u_charge[0].X - u_discharge[0].X
            else:
                return self.max_charge_power * 0.8
                
        except Exception as e:
            print(f"eMPC-V2G优化失败 t={t}: {e}")
            return self.max_charge_power * 0.5
    
    def simulate_ocmf_g2v(self):
        """使用OCMF算法模拟G2V场景(更保守的充电策略)"""
        power_trajectory = np.zeros(97)
        soc_trajectory = np.full(97, np.nan)
        
        arrival_idx, departure_idx = self._get_time_window_indices()
        
        # 接入瞬间SOC变为初始值
        soc_trajectory[arrival_idx] = self.initial_soc
        
        print(f"  OCMF-G2V充电阶段: t={arrival_idx} -> t={departure_idx}")
        for t in range(arrival_idx, departure_idx + 1):
            current_soc = soc_trajectory[t]
            remaining_steps = departure_idx + 1 - t
            horizon = min(self.control_horizon, remaining_steps)
            
            if horizon <= 1:
                break
            
            power = self._solve_ocmf_g2v_step(current_soc, t, horizon, remaining_steps)
            power_trajectory[t] = power
            
            energy_change = power * self.time_step * self.charge_efficiency / self.battery_capacity
            new_soc = np.clip(current_soc + energy_change, 0, 1)
            if t + 1 < 97:
                soc_trajectory[t + 1] = new_soc
        
        final_soc = soc_trajectory[departure_idx]
        if final_soc < self.target_soc - 0.01:
            print(f"  警告: 未达到目标SOC! 实际={final_soc:.3f}, 目标={self.target_soc:.3f}")
        
        print(f"  最终SOC: {final_soc:.3f}")
        return power_trajectory, soc_trajectory
    
    def _solve_ocmf_g2v_step(self, current_soc, t, horizon, remaining_steps):
        """求解单步OCMF-G2V优化问题(更平滑的功率分配)"""
        try:
            model = gp.Model("OCMF_G2V")
            model.setParam('OutputFlag', 0)
            model.setParam('TimeLimit', 10)
            
            # 决策变量
            u = model.addVars(horizon, lb=0, ub=self.max_charge_power, name="u")
            soc = model.addVars(horizon + 1, lb=0, ub=1, name="soc")
            
            model.addConstr(soc[0] == current_soc)
            
            # SOC演化
            for k in range(horizon):
                energy_change = u[k] * self.time_step * self.charge_efficiency / self.battery_capacity
                model.addConstr(soc[k + 1] == soc[k] + energy_change)
            
            # 终端约束
            if remaining_steps <= horizon:
                model.addConstr(soc[remaining_steps] >= self.target_soc)
                model.addConstr(soc[remaining_steps] <= self.target_soc)
            else:
                model.addConstr(soc[horizon] >= current_soc)
            
            # OCMF目标:电价成本 + 功率平滑惩罚
            cost_obj = gp.quicksum(
                self.charge_prices[(t + k) % 96] * u[k] * self.time_step
                for k in range(horizon)
            )
            
            # 功率变化平滑项(OCMF特色)
            smooth_obj = gp.quicksum(
                (u[k] - u[k-1]) * (u[k] - u[k-1])
                for k in range(1, horizon)
            ) * 0.01  # 小权重
            
            model.setObjective(cost_obj + smooth_obj, GRB.MINIMIZE)
            
            model.optimize()
            
            if model.status == GRB.OPTIMAL:
                return u[0].X
            else:
                return self.max_charge_power * 0.9
                
        except Exception as e:
            print(f"OCMF-G2V优化失败 t={t}: {e}")
            return self.max_charge_power * 0.7
    
    def simulate_ocmf_v2g(self):
        """使用OCMF算法模拟V2G场景(更激进的充放电策略)"""
        power_trajectory = np.zeros(97)
        soc_trajectory = np.full(97, np.nan)
        
        arrival_idx, departure_idx = self._get_time_window_indices()
        
        # 接入瞬间SOC变为初始值
        soc_trajectory[arrival_idx] = self.initial_soc
        
        print(f"  OCMF-V2G充放电阶段: t={arrival_idx} -> t={departure_idx}")
        for t in range(arrival_idx, departure_idx + 1):
            current_soc = soc_trajectory[t]
            remaining_steps = departure_idx + 1 - t
            horizon = min(self.control_horizon, remaining_steps)
            
            if horizon <= 1:
                break
            
            power = self._solve_ocmf_v2g_step(current_soc, t, horizon, remaining_steps)
            power_trajectory[t] = power
            
            if power > 0:
                efficiency = self.charge_efficiency
            else:
                efficiency = self.discharge_efficiency
            energy_change = power * self.time_step * efficiency / self.battery_capacity
            new_soc = np.clip(current_soc + energy_change, self.min_soc, 1)
            if t + 1 < 97:
                soc_trajectory[t + 1] = new_soc
        
        final_soc = soc_trajectory[departure_idx]
        if final_soc < self.target_soc - 0.01:
            print(f"  警告: 未达到目标SOC! 实际={final_soc:.3f}, 目标={self.target_soc:.3f}")
        
        print(f"  最终SOC: {final_soc:.3f}")
        return power_trajectory, soc_trajectory
    
    def _solve_ocmf_v2g_step(self, current_soc, t, horizon, remaining_steps):
        """求解单步OCMF-V2G优化问题(更激进地利用价差套利)"""
        try:
            model = gp.Model("OCMF_V2G")
            model.setParam('OutputFlag', 0)
            model.setParam('TimeLimit', 10)
            
            # 决策变量
            u_charge = model.addVars(horizon, lb=0, ub=self.max_charge_power, name="u_ch")
            u_discharge = model.addVars(horizon, lb=0, ub=self.max_discharge_power, name="u_dis")
            z = model.addVars(horizon, vtype=GRB.BINARY, name="z")
            soc = model.addVars(horizon + 1, lb=self.min_soc, ub=1, name="soc")
            
            model.addConstr(soc[0] == current_soc)
            
            # 互斥约束
            for k in range(horizon):
                model.addConstr(u_charge[k] <= self.max_charge_power * z[k])
                model.addConstr(u_discharge[k] <= self.max_discharge_power * (1 - z[k]))
            
            # SOC演化
            for k in range(horizon):
                charge_energy = u_charge[k] * self.time_step * self.charge_efficiency / self.battery_capacity
                discharge_energy = u_discharge[k] * self.time_step * self.discharge_efficiency / self.battery_capacity
                model.addConstr(soc[k + 1] == soc[k] + charge_energy - discharge_energy)
            
            # 终端约束
            if remaining_steps <= horizon:
                model.addConstr(soc[remaining_steps] >= self.target_soc)
                model.addConstr(soc[remaining_steps] <= self.target_soc)
            else:
                model.addConstr(soc[horizon] >= self.min_soc + 0.1)
            
            # OCMF-V2G目标:更激进地套利(放电收益权重更大)
            obj = gp.quicksum(
                self.charge_prices[(t + k) % 96] * u_charge[k] * self.time_step -
                self.discharge_prices[(t + k) % 96] * u_discharge[k] * self.time_step * 1.1  # 放电收益加权
                for k in range(horizon)
            )
            model.setObjective(obj, GRB.MINIMIZE)
            
            model.optimize()
            
            if model.status == GRB.OPTIMAL:
                return u_charge[0].X - u_discharge[0].X
            else:
                return self.max_charge_power * 0.7
                
        except Exception as e:
            print(f"OCMF-V2G优化失败 t={t}: {e}")
            return self.max_charge_power * 0.5
    
    def plot_all_curves(self, save_path=None, discharge_price_factor=None):
        """
        绘制所有4条SOC曲线
        :param save_path: 保存路径
        :param discharge_price_factor: 放电价格系数(0.8-1.2)
        """
        if discharge_price_factor is not None:
            self.discharge_price_factor = discharge_price_factor
            self.discharge_prices = self.charge_prices * self.discharge_price_factor
        
        print("正在模拟eMPC-G2V充电策略...")
        _, soc_empc_g2v = self.simulate_empc_g2v()
        
        print("正在模拟eMPC-V2G充放电策略...")
        _, soc_empc_v2g = self.simulate_empc_v2g()
        
        print("正在模拟OCMF-G2V充电策略...")
        _, soc_ocmf_g2v = self.simulate_ocmf_g2v()
        
        print("正在模拟OCMF-V2G充放电策略...")
        _, soc_ocmf_v2g = self.simulate_ocmf_v2g()
        
        # 绘图
        fig, ax = plt.subplots(figsize=(14, 8))
        
        curves = {
            'eMPC G2V': soc_empc_g2v,
            'eMPC V2G': soc_empc_v2g,
            'OCMF G2V': soc_ocmf_g2v,
            'OCMF V2G': soc_ocmf_v2g
        }
        
        colors = ['#FF6B6B', '#4ECDC4', '#9B59B6', '#FFA07A']
        
        for (label, soc), color in zip(curves.items(), colors):
            ax.plot(self.time_hours, soc, linewidth=2.5, label=label, color=color, 
                   marker='o', markersize=3, markevery=8)
        
        # 标注充电时间窗口
        arrival_idx, departure_idx = self._get_time_window_indices()
        arrival_hour = self.time_hours[arrival_idx]
        departure_hour = self.time_hours[min(departure_idx, 96)]
        
        # 添加充电窗口背景
        ax.axvspan(arrival_hour, 24, alpha=0.1, color='green', label='Charging Window')
        if departure_hour < arrival_hour:
            ax.axvspan(0, departure_hour, alpha=0.1, color='green')
        
        # 标注关键时间点
        ax.axvline(x=arrival_hour, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Arrival')
        ax.axvline(x=departure_hour, color='blue', linestyle='--', linewidth=1.5, alpha=0.7, label='Departure')
        
        # 标注电价峰谷时段(背景色)
        # 谷时: 0-7h
        ax.axvspan(0, 7, alpha=0.05, color='blue')
        # 峰时: 18-22h
        ax.axvspan(18, 22, alpha=0.05, color='red')
        
        # 设置X轴网格和刻度(每4小时一格)
        ax.xaxis.set_major_locator(MultipleLocator(4))
        ax.xaxis.set_minor_locator(MultipleLocator(1))
        
        # 设置Y轴网格和刻度(每0.2一格)
        ax.yaxis.set_major_locator(MultipleLocator(0.2))
        ax.yaxis.set_minor_locator(MultipleLocator(0.05))
        
        # 网格设置
        ax.grid(True, which='major', linestyle='-', linewidth=0.7, alpha=0.6)
        ax.grid(True, which='minor', linestyle=':', linewidth=0.3, alpha=0.3)
        
        # 标签和标题
        ax.set_xlabel('Time (h)', fontsize=12, fontweight='bold')
        ax.set_ylabel('SOC', fontsize=12, fontweight='bold')
        title = f'EV SOC Curves: MPC-based Charging Strategies\n'
        title += f'(Arrival: {self.arrival_time}:00, Departure: {self.departure_time}:00, '
        title += f'Initial SOC: {self.initial_soc*100:.0f}%, Target SOC: {self.target_soc*100:.0f}%, '
        title += f'Discharge Factor: {self.discharge_price_factor})'
        ax.set_title(title, fontsize=13, fontweight='bold', pad=20)
        
        # 设置轴范围
        ax.set_xlim(0, 24)
        ax.set_ylim(0, 1)
        
        # 格式化刻度标签
        ax.set_xticks(np.arange(0, 25, 4))
        ax.set_yticks(np.arange(0, 1.1, 0.2))
        
        # 图例
        ax.legend(loc='best', fontsize=10, framealpha=0.95, ncol=2)
        
        # 调整布局
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图表已保存到: {save_path}")
        
        plt.show()
        return fig, ax
    
    def plot_price_and_soc(self, save_path=None, discharge_price_factor=None):
        """
        绘制电价和SOC对比图(双Y轴)
        上方显示电价曲线,下方显示SOC曲线,时间对齐
        """
        if discharge_price_factor is not None:
            self.discharge_price_factor = discharge_price_factor
            self.discharge_prices = self.charge_prices * self.discharge_price_factor
        
        print("\n正在模拟4种策略...")
        _, soc_empc_g2v = self.simulate_empc_g2v()
        _, soc_empc_v2g = self.simulate_empc_v2g()
        _, soc_ocmf_g2v = self.simulate_ocmf_g2v()
        _, soc_ocmf_v2g = self.simulate_ocmf_v2g()
        
        # 创建双Y轴图
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
        
        # ========== 上图: 电价曲线 ==========
        ax1.plot(self.time_hours, self.charge_prices, linewidth=2.5, label='Charge Price', 
                color='#2E86AB', marker='s', markersize=4, markevery=8)
        ax1.plot(self.time_hours, self.discharge_prices, linewidth=2.5, label='Discharge Price', 
                color='#A23B72', marker='^', markersize=4, markevery=8)
        
        ax1.set_ylabel('Electricity Price (€/kWh)', fontsize=12, fontweight='bold')
        ax1.set_title('Real-time Pricing from EV2Gym Environment', 
                     fontsize=13, fontweight='bold', pad=10)
        ax1.legend(loc='upper right', fontsize=11, framealpha=0.95)
        ax1.grid(True, which='major', linestyle='-', linewidth=0.7, alpha=0.6)
        ax1.grid(True, which='minor', linestyle=':', linewidth=0.3, alpha=0.3)
        
        # 标注充电窗口
        arrival_idx, departure_idx = self._get_time_window_indices()
        arrival_hour = self.time_hours[arrival_idx]
        departure_hour = self.time_hours[departure_idx]
        ax1.axvspan(arrival_hour, departure_hour, alpha=0.15, color='green', label='Charging Window')
        ax1.axvline(x=arrival_hour, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
        ax1.axvline(x=departure_hour, color='blue', linestyle='--', linewidth=1.5, alpha=0.7)
        
        # ========== 下图: SOC曲线 ==========
        curves = {
            'eMPC G2V': soc_empc_g2v,
            'eMPC V2G': soc_empc_v2g,
            'OCMF G2V': soc_ocmf_g2v,
            'OCMF V2G': soc_ocmf_v2g
        }
        colors = ['#FF6B6B', '#4ECDC4', '#9B59B6', '#FFA07A']
        
        for (label, soc), color in zip(curves.items(), colors):
            ax2.plot(self.time_hours, soc, linewidth=2.5, label=label, color=color, 
                   marker='o', markersize=3, markevery=8)
        
        ax2.set_xlabel('Time (h)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('SOC', fontsize=12, fontweight='bold')
        ax2.set_title(f'SOC Comparison (Arrival: {self.arrival_time}, Departure: {self.departure_time}, Target: {self.target_soc*100:.0f}%)', 
                     fontsize=13, fontweight='bold', pad=10)
        
        # 标注充电窗口
        ax2.axvspan(arrival_hour, departure_hour, alpha=0.15, color='green', label='Charging Window')
        ax2.axvline(x=arrival_hour, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Arrival')
        ax2.axvline(x=departure_hour, color='blue', linestyle='--', linewidth=1.5, alpha=0.7, label='Departure')
        
        # 设置网格和刻度
        ax2.xaxis.set_major_locator(MultipleLocator(2))
        ax2.xaxis.set_minor_locator(MultipleLocator(0.5))
        ax2.yaxis.set_major_locator(MultipleLocator(0.2))
        ax2.yaxis.set_minor_locator(MultipleLocator(0.05))
        ax2.grid(True, which='major', linestyle='-', linewidth=0.7, alpha=0.6)
        ax2.grid(True, which='minor', linestyle=':', linewidth=0.3, alpha=0.3)
        ax2.set_xlim(0, 24)
        ax2.set_ylim(0, 1)
        ax2.legend(loc='best', fontsize=10, framealpha=0.95, ncol=2)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\n电价-SOC对比图已保存到: {save_path}")
        
        plt.show()
        return fig, (ax1, ax2)


def main():
    # 创建results目录
    os.makedirs('results', exist_ok=True)
    
    # 创建绘制器并生成图表
    print("初始化SOC曲线绘制器...")
    plotter = SOCCurvePlotter(battery_capacity=50, config_file='V2G_MPC.yaml')
    
    # 生成基于真实MPC优化的SOC对比曲线
    print("\n开始生成基于MPC优化的SOC对比曲线...")
    plotter.plot_all_curves(save_path='results/soc_curves_mpc_comparison.png', discharge_price_factor=1.2)
    
    # 生成电价-SOC对比图
    print("\n开始生成电价-SOC对比图...")
    plotter.plot_price_and_soc(save_path='results/price_soc_comparison.png', discharge_price_factor=1.2)
    
    print("\n完成! 可以尝试不同的放电价格系数(0.8-1.2)查看策略变化")
    print("\n电价数据来源说明:")
    print("- 电价数据从EV2Gym环境实时获取")
    print("- EV2Gym基于真实电网数据模拟的动态电价")
    print("- 充电电价: charge_prices (96个15分钟时间步)")
    print("- 放电电价: discharge_prices = charge_prices × discharge_price_factor (默认1.2)")
    print("- 数据范围: 0-24小时,每15分钟一个采样点")


if __name__ == '__main__':
    main()


