import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 设置中文字体
plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 数据准备
data = [
    ["EEGConformer", "会话1", 0.3247, 0.3246],
    ["EEGConformer", "会话2", 0.3021, 0.2891],
    ["EEGConformer", "会话3", 0.3126, 0.3126],
    ["EEGDeformer", "会话1", 0.3084, 0.2982],
    ["EEGDeformer", "会话2", 0.3260, 0.3069],
    ["EEGDeformer", "会话3", 0.2841, 0.2848],
    ["EEGNet", "会话1", 0.3310, 0.3275],
    ["EEGNet", "会话2", 0.3231, 0.3103],
    ["EEGNet", "会话3", 0.3110, 0.3040],
    ["EEG-TCNet", "会话1", 0.3434, 0.3359],
    ["EEG-TCNet", "会话2", 0.3378, 0.3091],
    ["EEG-TCNet", "会话3", 0.3291, 0.3325],
    ["LGGNet", "会话1", 0.3355, 0.3351],
    ["LGGNet", "会话2", 0.3488, 0.3322],
    ["LGGNet", "会话3", 0.3217, 0.3128],
    ["MLP", "会话1", 0.2778, 0.2683],
    ["MLP", "会话2", 0.2931, 0.2548],
    ["MLP", "会话3", 0.2841, 0.2549],
    ["MCHMNet", "会话1", 0.3405, 0.3335],
    ["MCHMNet", "会话2", 0.3524, 0.3388],
    ["MCHMNet", "会话3", 0.3595, 0.3573],
]

df = pd.DataFrame(data, columns=["模型", "测试集", "准确率", "F1分数"])

# 将准确率和F1分数转换为浮点数用于绘图
df["准确率"] = df["准确率"].astype(float)
df["F1分数"] = df["F1分数"].astype(float)

# 获取唯一的模型和测试集
models = df["模型"].unique()
sessions = df["测试集"].unique()

# 设置绘图风格
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
bar_width = 0.25
x = np.arange(len(models))

# 为每个测试集设置不同的偏移量
offsets = [-bar_width, 0, bar_width]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # 三种颜色区分三个session

# 绘制准确率对比图
for i, session in enumerate(sessions):
    session_data = [df[(df["模型"] == model) & (df["测试集"] == session)]["准确率"].values[0]
                   for model in models]
    ax1.bar(x + offsets[i], session_data, width=bar_width, label=session,
            color=colors[i], edgecolor='black')

# 设置准确率图的属性
ax1.set_xlabel('模型', fontsize=12)
ax1.set_ylabel('准确率', fontsize=12)
ax1.set_title('不同模型在各测试集上的准确率对比', fontsize=14)
ax1.set_xticks(x)
ax1.set_xticklabels(models, rotation=45, ha='right')
ax1.legend(title='测试集')
ax1.set_ylim(0.25, 0.38)  # 调整y轴范围突出差异
ax1.grid(axis='y', linestyle='--', alpha=0.7)

# 突出显示MCTMNet
# mctm_idx = np.where(models == "MCTMNet")[0][0]
# ax1.annotate('MCTMNet', xy=(mctm_idx, df[df["模型"] == "MCTMNet"]["准确率"].max()),
#              xytext=(mctm_idx, df[df["模型"] == "MCTMNet"]["准确率"].max() + 0.01),
#              arrowprops=dict(facecolor='black', shrink=0.05),
#              ha='center', fontweight='bold', color='red')

# 绘制F1分数对比图
for i, session in enumerate(sessions):
    session_data = [df[(df["模型"] == model) & (df["测试集"] == session)]["F1分数"].values[0]
                   for model in models]
    ax2.bar(x + offsets[i], session_data, width=bar_width, label=session,
            color=colors[i], edgecolor='black')

# 设置F1图的属性
ax2.set_xlabel('模型', fontsize=12)
ax2.set_ylabel('F1分数', fontsize=12)
ax2.set_title('不同模型在各测试集上的F1分数对比', fontsize=14)
ax2.set_xticks(x)
ax2.set_xticklabels(models, rotation=45, ha='right')
ax2.legend(title='测试集')
ax2.set_ylim(0.24, 0.37)  # 调整y轴范围突出差异
ax2.grid(axis='y', linestyle='--', alpha=0.7)

# # 突出显示MCTMNet
# ax2.annotate('MCTMNet', xy=(mctm_idx, df[df["模型"] == "MCTMNet"]["F1分数"].max()),
#              xytext=(mctm_idx, df[df["模型"] == "MCTMNet"]["F1分数"].max() + 0.01),
#              arrowprops=dict(facecolor='black', shrink=0.05),
#              ha='center', fontweight='bold', color='red')

# 调整布局
plt.tight_layout()

# 显示图形
plt.show()