import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
# 混淆矩阵数据
session1_cm = np.array([[762, 162, 247, 279],
                        [345, 249, 372, 484],
                        [163, 241, 432, 614],
                        [307, 251, 360, 532]])

session2_cm = np.array([[857, 206, 101, 286],
                        [388, 414, 269, 379],
                        [441, 364, 277, 368],
                        [422, 322, 210, 496]])

session3_cm = np.array([[762, 248, 203, 237],
                        [319, 417, 381, 333],
                        [323, 342, 447, 338],
                        [181, 390, 420, 459]])

# 标题与数据对应关系
session_data = [
    (session1_cm, "会话1\n准确率: 0.3405, F1分数: 0.3335"),
    (session2_cm, "会话2\n准确率: 0.3524, F1分数: 0.3388"),
    (session3_cm, "会话3\n准确率: 0.3595, F1分数: 0.3573")
]

classes = [0, 1, 2, 3]  # 类别标签

# 逐个绘制混淆矩阵
for cm, title in session_data:
    plt.figure(figsize=(8, 6))  # 每张图独立画布
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes,
                cbar=True)  # 显示颜色条辅助判断数值大小
    plt.title(title, fontsize=14)
    plt.xlabel('预测值', fontsize=12)
    plt.ylabel('真实值', fontsize=12)
    plt.tight_layout()  # 自动调整布局
    plt.show()  # 显示当前图像