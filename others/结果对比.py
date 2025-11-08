import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator

# 设置中文字体
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 数据准备
# data = {
#     '模型': ['EEGConformer', 'EEGDeformer', 'EEGNet', 'EEG-TCNet',
#             'LGGNet t', 'MLP', 'MCTMNet'],
#     'Acc均值': [0.3476, 0.3141, 0.3479, 0.3368, 0.3395, 0.2992, 0.3508],
#     'Acc标准差': [0.0051, 0.0093, 0.0035, 0.0059, 0.0215, 0.0068, 0.0078],
#     'F1均值': [0.3374, 0.3029, 0.3367, 0.3258, 0.3060, 0.2680, 0.3432],
#     'F1标准差': [0.0081, 0.0146, 0.0061, 0.0119, 0.0291, 0.0167, 0.0102]
# }
data = {
    '模型': ['EEGConformer', 'EEGDeformer', 'EEGNet', 'EEG-TCNet', 'LGGNet', 'MLP', 'MCHMNet'],
    'Acc均值': [0.3131, 0.3062, 0.3217, 0.3368, 0.3353, 0.2850, 0.3508],
    'Acc标准差': [0.0092, 0.0172, 0.0082, 0.0059, 0.0111, 0.0063, 0.0078],
    'F1均值': [0.3088, 0.2966, 0.3139, 0.3258, 0.3267, 0.2593, 0.3432],
    'F1标准差': [0.0148, 0.0091, 0.0099, 0.0119, 0.0099, 0.0063, 0.0102]
}
df = pd.DataFrame(data)

# 创建画布
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
x = np.arange(len(df['模型']))  # 标签位置
width = 0.8  # 柱形宽度

# 绘制Accuracy对比图
bars1 = ax1.bar(x, df['Acc均值'], width, yerr=df['Acc标准差'],
                capsize=5, color='skyblue', edgecolor='black')
ax1.set_xlabel('模型', fontsize=12)
ax1.set_ylabel('Accuracy', fontsize=12)
ax1.set_title('不同模型的Accuracy对比', fontsize=14)
ax1.set_xticks(x)
ax1.set_xticklabels(df['模型'], rotation=45, ha='right')
ax1.yaxis.set_major_locator(MaxNLocator(nbins=8))
ax1.grid(axis='y', linestyle='--', alpha=0.7)

# 在柱状图上添加数值
for bar in bars1:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.005,
             f'{height:.4f}', ha='center', va='bottom', rotation=0)

# 绘制F1分数对比图
bars2 = ax2.bar(x, df['F1均值'], width, yerr=df['F1标准差'],
                capsize=5, color='lightgreen', edgecolor='black')
ax2.set_xlabel('模型', fontsize=12)
ax2.set_ylabel('F1 Score', fontsize=12)
ax2.set_title('不同模型的F1分数对比', fontsize=14)
ax2.set_xticks(x)
ax2.set_xticklabels(df['模型'], rotation=45, ha='right')
ax2.yaxis.set_major_locator(MaxNLocator(nbins=8))
ax2.grid(axis='y', linestyle='--', alpha=0.7)

# 在柱状图上添加数值
for bar in bars2:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.005,
             f'{height:.4f}', ha='center', va='bottom', rotation=0)

# 调整布局
plt.tight_layout()

# 显示图形
plt.show()
