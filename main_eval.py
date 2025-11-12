"""
This script is used to evaluate the performance of the EVsSimulator environment.
"""
from ev2gym.models.ev2gym_env import EV2Gym
from ocmf_mpc import OCMF_V2G, OCMF_G2V
from eMPC import eMPC_V2G, eMPC_G2V

import numpy as np
import matplotlib.pyplot as plt

def eval():
    """
    运行 EV2Gym 环境的评估流程：
    - 构建环境（可从回放加载，或新开一轮仿真）
    - 选择一个 MPC 控制器（G2V 或 V2G）
    - 在整个仿真时域内循环：根据当前环境求动作 -> 环境前进一步 -> 可视化/统计
    """
    save_plots = True          # 是否由环境在结束时保存内置图表（取决于 env 的实现）
    
    replay_path = None         # 若给出回放文件路径，则会按该回放重放同一场景；None 表示不从回放加载

    # 配置文件（站点/配变/电价/到达分布等参数）
    config_file = r"E:\code\MPC-G2V-V2G\V2G_MPC.yaml"

    # 构建环境
    env = EV2Gym(
        config_file=config_file,
        load_from_replay_path=replay_path,
        verbose=True,          # 打印环境内部的关键信息，便于调试
        save_replay=False,     # 是否保存这次仿真的回放数据（这里设为 False）
        save_plots=save_plots, # 结束时是否由环境生成并保存图表
    )

    # 若保存回放，通常会约定一个回放文件路径（这里演示生成路径，不实际使用）
    new_replay_path = f"replay/replay_{env.sim_name}.pkl"

    # 重置环境，获取初始状态
    state, _ = env.reset()

    # 一些基础统计：EV 数量、停留时长范围等（用于 sanity check）
    ev_profiles = env.EVs_profiles    
    print(f'Number of EVs: {len(ev_profiles)}')
    max_time_of_stay = max([ev.time_of_departure - ev.time_of_arrival 
                            for ev in ev_profiles])
    min_time_of_stay = min([ev.time_of_departure - ev.time_of_arrival
                            for ev in ev_profiles])
    
    print(f'Max time of stay: {max_time_of_stay}')
    print(f'Min time of stay: {min_time_of_stay}')
    
    # 选择控制器（四选一）：
    # - OCMF_V2G / OCMF_G2V：带 CapF1 互斥线性化的 MPC 版本（你前面贴的那个）
    # - eMPC_V2G / eMPC_G2V：另一套实现（例如更简化或不同目标）
    # control_horizon=25 表示滚动优化每次看未来 25 个时段
    # agent = OCMF_V2G(env, control_horizon=25, verbose=False)
    # agent = OCMF_G2V(env, control_horizon=25, verbose=False)
    # agent = eMPC_V2G(env, control_horizon=25, verbose=False)
    agent = eMPC_G2V(env, control_horizon=25, verbose=False)

    # 主循环：遍历整个仿真时域
    for t in range(env.simulation_length):
        # 由控制器基于当前环境求解当下动作（通常是归一化功率分配）
        actions = agent.get_action(env)

        # 将动作作用于环境，环境前进一步
        # visualize=True：很多环境实现里会实时画图/更新曲线（若频繁绘图可能会稍慢）
        new_state, reward, done, _, stats = env.step(actions, visualize=True)

        # 仿真结束（例如所有 EV 已离场，或到达设定步数）
        if done:
            print(stats)  # 打印环境统计信息（总成本、违约次数、SOC 统计等，依实现而定）
            print(f'End of simulation at step {env.current_step}')
            break


if __name__ == "__main__":
    eval()
