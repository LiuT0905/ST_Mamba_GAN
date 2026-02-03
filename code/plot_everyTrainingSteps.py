import numpy as np
import matplotlib.pyplot as plt
import re

plt.rcParams['font.family'] = 'Times New Roman ,simsun'  # 设置字体族，中文为SimSun，英文为Times New Roman
plt.rcParams['mathtext.fontset'] = 'stix'  # 设置数学公式字体为stix

def parse_log_file(file_path):
    """解析日志文件，提取所有MSE值"""
    mse_values = []
    with open(file_path, 'r') as f:
        for line in f:
            # 使用正则表达式匹配MSE值
            match = re.search(r'MSE:\s*([\d.]+)', line)
            if match:
                mse_values.append(float(match.group(1)))
    return mse_values


def plot_rmse_trend(file_path, save_path=None):
    """绘制RMSE趋势图（基于MSE计算）"""
    # 1. 提取数据
    mse_values = parse_log_file(file_path)

    # 检查是否有足够的数据点
    if len(mse_values) == 0:
        print("未找到MSE数据，请检查日志文件格式")
        return

    # 将MSE转换为RMSE
    rmse_values = np.sqrt(mse_values)

    # 创建横坐标，将原始步数乘以40
    steps = np.arange(len(rmse_values)) * 40

    # 2. 创建图表
    plt.figure(figsize=(20, 12))

    # 绘制所有RMSE点（用细线连接）
    plt.plot(steps,rmse_values, color='blue', alpha=0.5, linewidth=0.8,
             label='Per-step RMSE')

    # 计算滑动平均（窗口大小为100步）使趋势更明显
    # window_size = 150
    # if len(rmse_values) > window_size:
    #     rmse_smooth = np.convolve(rmse_values, np.ones(window_size) / window_size, mode='valid')
    #     plt.plot(range(window_size - 1, len(rmse_values)), rmse_smooth,
    #              color='red', linewidth=2, label=f'{window_size}-step Moving Avg')
    # else:
    #     print(f"警告：数据点不足({len(rmse_values)})，无法计算{window_size}步滑动平均")

    # 图表装饰
    # plt.title('RMSE Trend Across All Training Steps', fontsize=25)
    plt.xlabel('Training times (batch)', fontsize=25)
    plt.ylabel('RMSE', fontsize=25)
    plt.grid(True, linestyle='--', alpha=0.3)

    # plt.legend(
    #     loc='lower left',  # 位置：'upper right', 'lower left', 'center left', 'best' 等
    #     fontsize=20,  # 图例字体大小
    #     frameon=True,  # 是否显示边框
    #     fancybox=True,  # 圆角边框
    #     shadow=False,  # 阴影效果
    #     framealpha=0.9,  # 边框透明度
    #     edgecolor='black',  # 边框颜色
    # )

    plt.tick_params(axis='both', which='major', labelsize=16)  # 主刻度
    plt.tick_params(axis='both', which='minor', labelsize=16)  # 次刻度
    # 自动调整y轴范围，排除极端值
    if len(rmse_values) > 0:
        lower = np.percentile(rmse_values, 5)
        upper = np.percentile(rmse_values, 95)
        plt.ylim(max(0, lower * 0.9), min(upper * 1.1, np.max(rmse_values)))

    # 添加对数刻度选项（如果数据范围很大）
    if max(rmse_values) > min(rmse_values) * 100:  # 如果数据跨越多个数量级
        plt.yscale('log')
        plt.ylabel('RMSE')

    # 保存或显示
    if save_path:
        plt.savefig(save_path, dpi=500, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

# 使用示例 - 替换为你的实际文件路径
log_file = r"D:\mamba_GAN\code\experiment\ST_Mamba_GAN_4\error\errlog.txt"  # 请修改为你的日志文件路径
output_image = "training_rmse_trend.png"  # 可选：保存图片的路径

plot_rmse_trend(log_file, output_image)
