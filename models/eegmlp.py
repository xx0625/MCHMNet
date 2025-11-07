import torch
import torch.nn as nn


# 修正后的MLP模型定义（保持MLP本质，移除卷积相关概念）
class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, nb_classes):
        super(SimpleMLP, self).__init__()
        self.flatten = nn.Flatten()  # 将输入展平为一维向量
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.bn1 = nn.BatchNorm1d(hidden_dim1)  # 批归一化
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)  # dropout防止过拟合

        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.bn2 = nn.BatchNorm1d(hidden_dim2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)

        self.fc3 = nn.Linear(hidden_dim2, nb_classes)  # 输出层

    def forward(self, x):
        x = self.flatten(x)  # 展平为(batch_size, input_dim)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.dropout2(x)

        x = self.fc3(x)
        return x


if __name__ == "__main__":
    num_channels = 61  # 通道数
    hidden_dim1 = 512  # 第一个隐藏层维度（可根据需要调整）
    hidden_dim2 = 128  # 第二个隐藏层维度（可根据需要调整）
    num_classes = 4  # 分类数

    for num_timesteps in [250, 512, 1024]:
        # 计算MLP的输入维度：通道数 × 时间步（展平后的数据）
        input_dim = num_channels * num_timesteps
        # 实例化模型（参数与构造函数匹配）
        model = SimpleMLP(
            input_dim=input_dim,
            hidden_dim1=hidden_dim1,
            hidden_dim2=hidden_dim2,
            nb_classes=num_classes
        )

        print(f"\n时间步长: {num_timesteps}")
        print(f"MLP输入维度: {input_dim}（{num_channels}通道 × {num_timesteps}时间步）")
        # 测试输入输出形状
        x = torch.randn(8, num_channels, num_timesteps)  # 模拟输入：(batch_size=8, 通道数, 时间步)
        y = model(x)
        print(f"输出形状: {y.shape}（应为 (batch_size, 分类数) = (8, {num_classes})）")