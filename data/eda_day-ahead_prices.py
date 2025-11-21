import pandas as pd
import matplotlib.pyplot as plt

# =============================
# 读取并预处理荷兰光伏数据（pv_netherlands.csv）
# =============================

# 读取 CSV 文件（当前目录下）
data = pd.read_csv(r'pv_netherlands.csv', sep=',', header=0)
print(data)

# 删除原始 time 列（注意：文件里有 'time' 和 'local_time' 两列）
data.drop('time', inplace=True, axis=1)

# 将 'local_time' 列重命名为统一的 'time'
data.rename(columns={'local_time': 'time'}, inplace=True)

# =============================
# 仿真参数（来自环境变量）
# =============================
desired_timescale = 10      # 期望的时间尺度（分钟），例如 10 分钟一条
simulation_length = 96      # 仿真步数（例如 96 表示 96 个时间步）
simulation_date = '2023-2-1 08:00:00'  # 仿真起始日期
number_of_transformers = 3  # 变压器数量

# =============================
# 数据集原始时间尺度与起始日期
# =============================
dataset_timescale = 60                   # 数据集原始粒度（分钟）——示例：60 分钟一条
dataset_starting_date = '2022-01-01 00:00:00'

# =============================
# 重采样到期望时间尺度
# 若 desired_timescale > dataset_timescale：下采样（取更粗的粒度）
# 若 desired_timescale < dataset_timescale：上采样（重复行扩展）
# =============================

if desired_timescale > dataset_timescale:
    # 下采样：把多行聚合成一行（这里用 max，你也可以改成 mean/median/sum）
    factor = int(desired_timescale / dataset_timescale)  # 聚合倍数（必须为整数）
    # 使用整数分组键（index // factor）来分组聚合
    data = data.groupby(data.index // factor, as_index=False).max()
elif desired_timescale < dataset_timescale:
    # 上采样：行重复扩展，倍数=原始粒度/目标粒度
    factor = int(dataset_timescale / desired_timescale)  # 重复倍数（必须为整数）
    # 通过重复索引的方式扩展数据长度
    data = data.loc[data.index.repeat(factor)].reset_index(drop=True)
    # 如果想要把功率/能量按时间尺度均摊，可在此处进行处理（本例保留原值）
    # data = data / factor

# =============================
# 平滑处理：滚动均值 + 指数加权平均（EWM）
# 窗口大小：每小时内的样本数 = 60 / desired_timescale
# =============================
window = max(1, 60 // desired_timescale)  # 避免窗口为 0
# 先做滚动均值（min_periods=1 保证开头不丢数据）
data['electricity'] = data['electricity'].rolling(window=window, min_periods=1).mean()
# 再做 EWM 平滑（平滑强度由 span 决定）
data['electricity'] = data['electricity'].ewm(span=window, adjust=True).mean()

# 将数值放大到百分比/更直观量级（按需调整）
data['electricity'] = data['electricity'] * 100

# =============================
# 绘图：绘制一周数据（24*7 小时），对应的样本数= (60/desired_timescale) * 24 * 7
# =============================
points_one_week = (60 // desired_timescale) * 24 * 7
data['electricity'][:points_one_week].plot()
plt.xlabel('样本点（前一周）')
plt.ylabel('electricity（缩放后）')
plt.title('荷兰光伏电力：一周片段（平滑后）')
plt.grid(True)
plt.show()

# =============================
# 数据扩展：拼接自身，形成两年的数据量（简单倍增）
# =============================
data = pd.concat([data, data], ignore_index=True)

# =============================
# 下面是你曾经的“年”维度可视化草稿（当前数据未提取年）
# 如果需要按年绘图，可先将 time 转为 datetime 并提取 year
# =============================

# # 将 'time' 转为 datetime，并提取年、月、日、小时（可选）
# data['time'] = pd.to_datetime(data['time'])
# data['year'] = data['time'].dt.year
# data['month'] = data['time'].dt.month
# data['day'] = data['time'].dt.day
# data['hour'] = data['time'].dt.hour

# # 多子图按年画（示例）
# plt.figure(figsize=(10, 7))
# years = sorted(data['year'].dropna().unique())
# for i, year in enumerate(years[:9]):  # 只展示前 9 个年，避免子图太多
#     data_temp = data[data['year'] == year]
#     plt.subplot(3, 3, i + 1)
#     plt.title(f'year {year}')
#     plt.xlabel('time')
#     plt.ylabel('electricity')
#     plt.plot(data_temp['time'], data_temp['electricity'])
# plt.tight_layout()
# plt.savefig('data/eda_pv_netherlands_by_year.png')

# =============================
# 说明：
# 1) 若要与日前电价数据（Netherlands_day-ahead-2015-2023.csv）联动，可用 time 进行合并。
# 2) 若需要严格的能量守恒，放大/平滑前后请根据时间尺度进行归一或积分。
# 3) 上/下采样策略可根据业务需要更换（mean/median/sum/max 等）。
# =============================
