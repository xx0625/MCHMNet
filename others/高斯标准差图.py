import matplotlib.pyplot as plt
import numpy as np

# 数据整理
stds = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]  # 高斯噪声标准差

# 平均准确率数据（均值和标准差）
acc_means = [0.3384, 0.3344, 0.3508, 0.3385, 0.3290, 0.3344, 0.3314, 0.3292, 0.3289, 0.3279]
acc_stds = [0.0148, 0.0153, 0.0078, 0.0163, 0.0124, 0.0154, 0.0176, 0.0099, 0.0080, 0.0207]

# 平均F1分数数据（均值和标准差）
f1_means = [0.3236, 0.3143, 0.3432, 0.3275, 0.3100, 0.3212, 0.3199, 0.3204, 0.3218, 0.3216]
f1_stds = [0.0174, 0.0111, 0.0102, 0.0116, 0.0034, 0.0075, 0.0106, 0.0078, 0.0025, 0.0173]

# 设置中文字体显示
plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

# 创建画布
plt.figure(figsize=(10, 6))

# 绘制平均准确率曲线（带误差线）
plt.errorbar(
    stds, acc_means, yerr=acc_stds,
    fmt='-o', color='dodgerblue', ecolor='lightblue',
    elinewidth=2, capsize=4, label='平均准确率'
)

# 绘制平均F1分数曲线（带误差线）
plt.errorbar(
    stds, f1_means, yerr=f1_stds,
    fmt='-s', color='crimson', ecolor='lightcoral',
    elinewidth=2, capsize=4, label='平均F1分数'
)

# 设置坐标轴标签和标题
plt.xlabel('高斯噪声标准差', fontsize=12)
plt.ylabel('分数', fontsize=12)
plt.title('不同高斯噪声标准差下的模型性能指标', fontsize=14)

# 设置坐标轴范围，让数据更清晰
plt.xlim(0.005, 0.105)
plt.ylim(0.30, 0.36)

# 添加网格线
plt.grid(linestyle='--', alpha=0.7)

# 添加图例
plt.legend(fontsize=10)

# 调整布局
plt.tight_layout()

# 显示图像
plt.show()

# 如需保存图像，可取消下面一行的注释
# plt.savefig('noise_performance.png', dpi=300, bbox_inches='tight')