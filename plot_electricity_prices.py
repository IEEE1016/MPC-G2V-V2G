"""
绘制24小时电价曲线（充电和放电）。
从EV2Gym环境中提取价格数据并用matplotlib可视化。
"""
import numpy as np
import matplotlib.pyplot as plt
from ev2gym.models.ev2gym_env import EV2Gym

def plot_electricity_prices(config_file=r"E:\code\MPC-G2V-V2G\V2G_MPC.yaml", 
                             save_path=r"E:\code\MPC-G2V-V2G\results\electricity_prices.png"):
    """
    绘制24小时的充电和放电电价曲线。
    
    Args:
        config_file: EV2Gym配置文件的路径
        save_path: 生成图表的保存路径
    """
    # 初始化EV2Gym环境（random_day: true会随机选择一天）
    env = EV2Gym(
        config_file=config_file,
        verbose=False,
        save_replay=False,
        save_plots=False,
    )
    
    # 重置环境以获取价格数据
    state, _ = env.reset()
    
    # 提取价格数据（96个时间步，对应24小时的15分钟间隔）
    ch_prices = np.abs(env.charge_prices[0, :])  # 充电价格
    disch_prices = np.abs(env.discharge_prices[0, :])  # 放电价格
    
    # 创建时间轴（小时：0, 1, 2, ..., 23）
    # 96步 = 24小时 * 4（每小时4个15分钟间隔）
    hours = np.arange(0, 24, 1)  # 仅显示整点标签
    time_steps = np.arange(96)  # 所有96个时间步用于绘图
    time_hours = time_steps / 4  # 转换为小时（0, 0.25, 0.5, ..., 23.75）
    
    # 创建图表和坐标轴
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # 绘制充电价格（蓝色）和放电价格（红色），不显示点标记
    ax.plot(time_hours, ch_prices, color='blue', linewidth=2.5, label='Charging Price', linestyle='-')
    ax.plot(time_hours, disch_prices, color='red', linewidth=2.5, label='Discharging Price', linestyle='-')
    
    # 设置x轴仅显示整点标签
    ax.set_xticks(hours)
    ax.set_xticklabels([f'{int(h):02d}:00' for h in hours])
    
    # 设置y轴为0.15-0.35，每0.05变化一格
    y_ticks = np.arange(0.15, 0.36, 0.05)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f'{y:.2f}' for y in y_ticks])
    
    # 添加网格
    ax.grid(True, linestyle='--', alpha=0.6, which='both')
    
    # 设置标签和标题（全为英文）
    ax.set_xlabel('Time (Hour)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Price (€/kWh)', fontsize=12, fontweight='bold')
    ax.set_title('24-Hour Electricity Price Curve', fontsize=14, fontweight='bold')
    
    # 设置坐标轴范围
    ax.set_xlim(-0.5, 23.5)
    ax.set_ylim(0.15, 0.35)
    
    # 添加图例
    ax.legend(loc='upper left', fontsize=11, framealpha=0.95)
    
    # 紧凑布局
    plt.tight_layout()
    
    # 保存图表
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Price plot saved to: {save_path}")
    
    # 打印价格统计信息
    print("\n=== Price Statistics ===")
    print(f"Charging Price:")
    print(f"  Min: {ch_prices.min():.4f} €/kWh")
    print(f"  Max: {ch_prices.max():.4f} €/kWh")
    print(f"  Mean: {ch_prices.mean():.4f} €/kWh")
    print(f"\nDischarging Price:")
    print(f"  Min: {disch_prices.min():.4f} €/kWh")
    print(f"  Max: {disch_prices.max():.4f} €/kWh")
    print(f"  Mean: {disch_prices.mean():.4f} €/kWh")
    print(f"\nDischarge Price Factor: {disch_prices[0] / ch_prices[0]:.2f}")
    
    plt.show()


if __name__ == "__main__":
    plot_electricity_prices()