import matplotlib.pyplot as plt
import numpy as np
import re
plt.rcParams['font.family'] = 'Times New Roman ,simsun'  # 设置字体族，中文为SimSun，英文为Times New Roman
plt.rcParams['mathtext.fontset'] = 'stix'  # 设置数学公式字体为stix
plt.switch_backend('TkAgg')
# 设置支持中文的字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

file_path_D = r"D:\mamba_GAN\code\experiment\ST_Mamba_GAN_4\error\errlog.txt"
file_path_G = r"D:\mamba_GAN\code\experiment\ST_Mamba_GAN_4\error\errlog.txt"

# 读取第一个txt文件
with open(file_path_D, 'r') as file:
    data_DD = file.readlines()

data_D = []
for line in data_DD:
    match = re.search(r'Loss_D: ([\d\.]+)', line)
    if match:
        data_D.append(float(match.group(1)))

# 获取第一个数据列
y_values_D = data_D

# 生成第一个横坐标
x_values_D = np.arange(0, len(y_values_D)) * (246300 / len(y_values_D))

# 计算第一个数据集的均值（仅最后60个数，若不足60则使用全部）
y_mean_D = np.mean(y_values_D[-60:]) if len(y_values_D) > 0 else float('nan')

plt.figure(figsize=(8, 5))
# 绘制第一个折线图
plt.plot(x_values_D, y_values_D, label='Loss_D', linewidth=0.3)

# 在图上绘制第一个数据集的均值
# plt.axhline(y=y_mean_D, color='r', linestyle='--', label='mean_D')
# 读取第二个txt文件

with open(file_path_G, 'r') as file:
    data_GG = file.readlines()

data_G = []
for line in data_GG:
    match = re.search(r'Loss_G: ([\d\.]+)', line)
    if match:
        data_G.append(float(match.group(1)))

# 数据列
y_values_G = data_G

# 生成第二个横坐标
x_values_G = np.arange(0, len(y_values_G)) * (246300 / len(y_values_G))

# 计算第二个数据集的均值（仅最后60个数，若不足60则使用全部）
y_mean_G = np.mean(y_values_G[-60:]) if len(y_values_G) > 0 else float('nan')

# 绘制第二个折线图
plt.plot(x_values_G, y_values_G, label='Loss_G', linewidth=0.2)
# plt.axhline(y=y_mean_G, color='y', linestyle='--', label='mean_G')

# 使用七次多项式拟合并绘制拟合曲线
# degree = 7
# coeff_D = np.polyfit(x_values_D, y_values_D, degree)
# coeff_G = np.polyfit(x_values_G, y_values_G, degree)
#
# x_fit_D = np.linspace(min(x_values_D), max(x_values_D), 100)
# y_fit_D = np.polyval(coeff_D, x_fit_D)
#
# x_fit_G = np.linspace(min(x_values_G), max(x_values_G), 100)
# y_fit_G = np.polyval(coeff_G, x_fit_G)
# 绘制拟合曲线
# plt.plot(x_fit_D, y_fit_D, label='The trend of D', linestyle。='--')
# plt.plot(x_fit_G, y_fit_G, label='The trend of G', linestyle='--')

# 设置纵坐标的标签为10的幂次方
plt.yscale('log', base=10)
# 设置纵坐标范围和刻度
plt.ylim(1e-1, 1e1)
plt.yticks([ 1e-1, 1e0, 1e1])
# plt.ylim(1e-2, 1e1)
# plt.yticks([ 1e-2,1e-1, 1e0, 1e1])
# 设置图表标题和坐标轴标签
# plt.title('LOSS G & D', fontsize=16)
plt.ylabel('Loss', fontsize=14)
plt.xlabel('Training times (batch)', fontsize=14)
# 设置 x 轴刻度标签
plt.xticks(np.arange(0, 246301, 20000))

# 添加图例，并设置图例字体
legend = plt.legend(fontsize=12)
# for text in legend.texts:
#     text.set_fontproperties('Times New Roman ,simsun')

plt.savefig('loss_G&D.png', dpi=500, bbox_inches='tight', pad_inches=0.05)

print('Last 3 epochs Mean D = ', y_mean_D)
print('Last 3 epochs Mean G = ', y_mean_G)

# 显示图表
plt.show()