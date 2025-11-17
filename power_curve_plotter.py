import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import matplotlib.patches as mpatches
import yaml
import gurobipy as gp
from gurobipy import GRB
from ev2gym.models.ev2gym_env import EV2Gym
import os


class PowerCurvePlotter:
    def __init__(self, battery_capacity=50, config_file='V2G_MPC.yaml'):
        """
        初始化充电桩功率曲线绘制器
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
        """获取充电时间窗口的索引"""
        arrival_idx = int(self.arrival_time / self.time_step)
        departure_idx = int(self.departure_time / self.time_step)
        
        return arrival_idx, departure_idx
    
    def simulate_empc_g2v(self):
        """使用eMPC算法模拟G2V场景(仅充电)"""
        power_trajectory = np.full(97, np.nan)  # 用NaN表示未接入
        soc_trajectory = np.full(97, np.nan)
        
        arrival_idx, departure_idx = self._get_time_window_indices()
        soc_trajectory[arrival_idx] = self.initial_soc
        
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
            if t + 1 <= departure_idx:
                soc_trajectory[t + 1] = new_soc
        
        final_soc = soc_trajectory[departure_idx]
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
            
            # 终端约束
            if remaining_steps <= horizon:
                model.addConstr(soc[remaining_steps] >= self.target_soc)
                model.addConstr(soc[remaining_steps] <= self.target_soc)
            else:
                model.addConstr(soc[horizon] >= current_soc)
            
            # 目标函数: 最小化充电成本
            obj = gp.quicksum(self.charge_prices[(t + k) % 96] * u[k] * self.time_step 
                            for k in range(horizon))
            model.setObjective(obj, GRB.MINIMIZE)
            
            model.optimize()
            
            if model.status == GRB.OPTIMAL:
                return u[0].X
            else:
                return self.max_charge_power
                
        except Exception as e:
            print(f"eMPC-G2V优化失败 t={t}: {e}")
            return self.max_charge_power
    
    def simulate_empc_v2g(self):
        """使用eMPC算法模拟V2G场景(允许充放电)"""
        power_trajectory = np.full(97, np.nan)  # 用NaN表示未接入
        soc_trajectory = np.full(97, np.nan)
        
        arrival_idx, departure_idx = self._get_time_window_indices()
        soc_trajectory[arrival_idx] = self.initial_soc
        
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
            if t + 1 <= departure_idx:
                soc_trajectory[t + 1] = new_soc
        
        final_soc = soc_trajectory[departure_idx]
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
                model.addConstr(soc[horizon] >= self.min_soc + 0.1)
            
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
        power_trajectory = np.full(97, np.nan)  # 用NaN表示未接入
        soc_trajectory = np.full(97, np.nan)
        
        arrival_idx, departure_idx = self._get_time_window_indices()
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
            if t + 1 <= departure_idx:
                soc_trajectory[t + 1] = new_soc
        
        final_soc = soc_trajectory[departure_idx]
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
            
            # 功率变化平滑项
            smooth_obj = gp.quicksum(
                (u[k] - u[k-1]) * (u[k] - u[k-1])
                for k in range(1, horizon)
            ) * 0.01
            
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
        power_trajectory = np.full(97, np.nan)  # 用NaN表示未接入
        soc_trajectory = np.full(97, np.nan)
        
        arrival_idx, departure_idx = self._get_time_window_indices()
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
            if t + 1 <= departure_idx:
                soc_trajectory[t + 1] = new_soc
        
        final_soc = soc_trajectory[departure_idx]
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
            
            # OCMF-V2G目标:更激进地套利
            obj = gp.quicksum(
                self.charge_prices[(t + k) % 96] * u_charge[k] * self.time_step -
                self.discharge_prices[(t + k) % 96] * u_discharge[k] * self.time_step * 1.1
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
    
    def plot_power_curves(self, save_path=None, discharge_price_factor=None):
        """
        绘制所有4条充电桩功率曲线
        :param save_path: 保存路径
        :param discharge_price_factor: 放电价格系数(0.8-1.2)
        """
        if discharge_price_factor is not None:
            self.discharge_price_factor = discharge_price_factor
            self.discharge_prices = self.charge_prices * self.discharge_price_factor
        
        print("正在模拟eMPC-G2V充电策略...")
        power_empc_g2v, _ = self.simulate_empc_g2v()
        
        print("正在模拟eMPC-V2G充放电策略...")
        power_empc_v2g, _ = self.simulate_empc_v2g()
        
        print("正在模拟OCMF-G2V充电策略...")
        power_ocmf_g2v, _ = self.simulate_ocmf_g2v()
        
        print("正在模拟OCMF-V2G充放电策略...")
        power_ocmf_v2g, _ = self.simulate_ocmf_v2g()
        
        # 绘图
        fig, ax = plt.subplots(figsize=(14, 8))
        
        curves = {
            'eMPC G2V': power_empc_g2v,
            'eMPC V2G': power_empc_v2g,
            'OCMF G2V': power_ocmf_g2v,
            'OCMF V2G': power_ocmf_v2g
        }
        
        colors = ['#FF6B6B', '#4ECDC4', '#9B59B6', '#FFA07A']
        
        for (label, power), color in zip(curves.items(), colors):
            ax.plot(self.time_hours, power, linewidth=2.5, label=label, color=color, 
                   drawstyle='steps-post')
        
        # 标注充电时间窗口
        arrival_idx, departure_idx = self._get_time_window_indices()
        arrival_hour = self.time_hours[arrival_idx]
        departure_hour = self.time_hours[min(departure_idx, 96)]
        
        # 添加充电窗口背景
        ax.axvspan(arrival_hour, departure_hour, alpha=0.1, color='green', label='Charging Window')
        
        # 标注关键时间点
        ax.axvline(x=arrival_hour, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Arrival')
        ax.axvline(x=departure_hour, color='blue', linestyle='--', linewidth=1.5, alpha=0.7, label='Departure')
        
        # 添加零功率参考线
        ax.axhline(y=0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
        
        # 标注充放电区域
        ax.text(12, self.max_charge_power * 0.9, 'Charging (+)', 
               fontsize=11, ha='center', va='center', 
               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
        ax.text(12, -self.max_discharge_power * 0.9, 'Discharging (-)', 
               fontsize=11, ha='center', va='center',
               bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))
        
        # 设置X轴网格和刻度
        ax.xaxis.set_major_locator(MultipleLocator(4))
        ax.xaxis.set_minor_locator(MultipleLocator(1))
        
        # 设置Y轴网格和刻度
        y_max = max(self.max_charge_power, self.max_discharge_power)
        ax.yaxis.set_major_locator(MultipleLocator(5))
        ax.yaxis.set_minor_locator(MultipleLocator(1))
        
        # 网格设置
        ax.grid(True, which='major', linestyle='-', linewidth=0.7, alpha=0.6)
        ax.grid(True, which='minor', linestyle=':', linewidth=0.3, alpha=0.3)
        
        # 标签和标题
        ax.set_xlabel('Time (h)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Charging Power (kW)', fontsize=12, fontweight='bold')
        title = f'EV Charging Power Curves: MPC-based Strategies\n'
        title += f'(Arrival: {self.arrival_time}:00, Departure: {self.departure_time}:00, '
        title += f'Initial SOC: {self.initial_soc*100:.0f}%, Target SOC: {self.target_soc*100:.0f}%, '
        title += f'Max Power: {self.max_charge_power:.0f}kW, Discharge Factor: {self.discharge_price_factor})'
        ax.set_title(title, fontsize=13, fontweight='bold', pad=20)
        
        # 设置轴范围
        ax.set_xlim(0, 24)
        ax.set_ylim(-y_max * 1.1, y_max * 1.1)
        
        # 图例
        ax.legend(loc='best', fontsize=10, framealpha=0.95, ncol=2)
        
        # 调整布局
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图表已保存到: {save_path}")
        
        plt.show()
        return fig, ax
    
    def plot_soc_and_power(self, save_path=None, discharge_price_factor=None):
        """
        绘制SOC和功率对比图(双Y轴)
        上方显示SOC曲线,下方显示功率曲线,时间对齐
        """
        if discharge_price_factor is not None:
            self.discharge_price_factor = discharge_price_factor
            self.discharge_prices = self.charge_prices * self.discharge_price_factor
        
        print("\n正在模拟4种策略...")
        power_empc_g2v, soc_empc_g2v = self.simulate_empc_g2v()
        power_empc_v2g, soc_empc_v2g = self.simulate_empc_v2g()
        power_ocmf_g2v, soc_ocmf_g2v = self.simulate_ocmf_g2v()
        power_ocmf_v2g, soc_ocmf_v2g = self.simulate_ocmf_v2g()
        
        # 创建双Y轴图
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
        
        # ========== 上图: SOC曲线 ==========
        soc_curves = {
            'eMPC G2V': soc_empc_g2v,
            'eMPC V2G': soc_empc_v2g,
            'OCMF G2V': soc_ocmf_g2v,
            'OCMF V2G': soc_ocmf_v2g
        }
        colors = ['#FF6B6B', '#4ECDC4', '#9B59B6', '#FFA07A']
        
        for (label, soc), color in zip(soc_curves.items(), colors):
            ax1.plot(self.time_hours, soc, linewidth=2.5, label=label, color=color, 
                   marker='o', markersize=3, markevery=8)
        
        ax1.set_ylabel('SOC', fontsize=12, fontweight='bold')
        ax1.set_title('SOC and Charging Power Curves', 
                     fontsize=14, fontweight='bold', pad=10)
        ax1.legend(loc='best', fontsize=11, framealpha=0.95, ncol=2)
        ax1.grid(True, which='major', linestyle='-', linewidth=0.7, alpha=0.6)
        ax1.grid(True, which='minor', linestyle=':', linewidth=0.3, alpha=0.3)
        ax1.yaxis.set_major_locator(MultipleLocator(0.2))
        ax1.yaxis.set_minor_locator(MultipleLocator(0.05))
        ax1.set_ylim(0, 1)
        
        # 标注充电窗口
        arrival_idx, departure_idx = self._get_time_window_indices()
        arrival_hour = self.time_hours[arrival_idx]
        departure_hour = self.time_hours[departure_idx]
        ax1.axvspan(arrival_hour, departure_hour, alpha=0.15, color='green')
        ax1.axvline(x=arrival_hour, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
        ax1.axvline(x=departure_hour, color='blue', linestyle='--', linewidth=1.5, alpha=0.7)
        
        # ========== 下图: 功率曲线 ==========
        power_curves = {
            'eMPC G2V': power_empc_g2v,
            'eMPC V2G': power_empc_v2g,
            'OCMF G2V': power_ocmf_g2v,
            'OCMF V2G': power_ocmf_v2g
        }
        
        for (label, power), color in zip(power_curves.items(), colors):
            ax2.plot(self.time_hours, power, linewidth=2.5, label=label, color=color, 
                   drawstyle='steps-post')
        
        ax2.set_xlabel('Time (h)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Charging Power (kW)', fontsize=12, fontweight='bold')
        ax2.set_title(f'Power Comparison (+ Charging, - Discharging)', 
                     fontsize=13, fontweight='bold', pad=10)
        
        # 标注充电窗口
        ax2.axvspan(arrival_hour, departure_hour, alpha=0.15, color='green', label='Charging Window')
        ax2.axvline(x=arrival_hour, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Arrival')
        ax2.axvline(x=departure_hour, color='blue', linestyle='--', linewidth=1.5, alpha=0.7, label='Departure')
        ax2.axhline(y=0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
        
        # 设置网格和刻度
        ax2.xaxis.set_major_locator(MultipleLocator(2))
        ax2.xaxis.set_minor_locator(MultipleLocator(0.5))
        ax2.yaxis.set_major_locator(MultipleLocator(5))
        ax2.yaxis.set_minor_locator(MultipleLocator(1))
        ax2.grid(True, which='major', linestyle='-', linewidth=0.7, alpha=0.6)
        ax2.grid(True, which='minor', linestyle=':', linewidth=0.3, alpha=0.3)
        ax2.set_xlim(0, 24)
        y_max = max(self.max_charge_power, self.max_discharge_power)
        ax2.set_ylim(-y_max * 1.1, y_max * 1.1)
        ax2.legend(loc='best', fontsize=10, framealpha=0.95, ncol=2)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\nSOC-功率对比图已保存到: {save_path}")
        
        plt.show()
        return fig, (ax1, ax2)


def main():
    # 创建results目录
    os.makedirs('results', exist_ok=True)
    
    # 创建绘制器并生成图表
    print("初始化充电桩功率曲线绘制器...")
    plotter = PowerCurvePlotter(battery_capacity=50, config_file='V2G_MPC.yaml')
    
    # 生成功率对比曲线
    print("\n开始生成充电桩功率对比曲线...")
    plotter.plot_power_curves(save_path='results/power_curves_mpc_comparison.png', discharge_price_factor=1.2)
    
    # 生成SOC-功率对比图
    print("\n开始生成SOC-功率对比图...")
    plotter.plot_soc_and_power(save_path='results/soc_power_comparison.png', discharge_price_factor=1.2)
    
    print("\n完成! 功率曲线说明:")
    print("- 正值: 充电功率 (kW)")
    print("- 负值: 放电功率 (kW)")
    print("- G2V策略: 仅充电,功率始终≥0")
    print("- V2G策略: 可充可放,根据电价套利")
    print("- eMPC: 基于成本优化的模型预测控制")
    print("- OCMF: 考虑功率平滑的优化控制")


if __name__ == '__main__':
    main()
