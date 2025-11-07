import torch
import torch.nn as nn
import torch.nn.functional as F


# ======================= 基础组件 =======================
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x if self.chomp_size == 0 else x[:, :, :-self.chomp_size].contiguous()


class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, dilation=dilation)
        self.chomp = Chomp1d(padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.chomp(x)
        x = self.bn(x)
        return self.relu(x)


class BTCConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super().__init__()
        padding = ((kernel_size - 1) // 2) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, dilation=dilation)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)


# ======================= HTFA 模块 =======================
class HTFA(nn.Module):
    def __init__(self, channels, reduction=8, kernel_size=7, use_fft=True):
        super().__init__()
        self.use_fft = use_fft

        # 通道注意（时域）
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(
            nn.Conv1d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv1d(channels // reduction, channels, 1, bias=False)
        )

        # FFT通道注意（频域）
        if use_fft:
            self.fft_fc = nn.Sequential(
                nn.Linear(channels, channels // reduction, bias=False),
                nn.ReLU(),
                nn.Linear(channels // reduction, channels, bias=False),
                nn.Sigmoid()
            )

        # 时间注意
        self.conv_time = nn.Conv1d(2, 1, kernel_size=kernel_size,
                                   padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid_time = nn.Sigmoid()

    def forward(self, x):  # x: (B, C, T)
        # 通道注意
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        ca = torch.sigmoid(avg_out + max_out)  # 时域通道注意

        if self.use_fft:
            # 频域注意
            xf = torch.fft.rfft(x, dim=-1).abs()  # 取频谱幅值
            xf = xf.mean(dim=-1)  # (B, C)
            fa = self.fft_fc(xf)  # (B, C)
            fa = fa.unsqueeze(-1)
            ca = ca * fa  # 融合时频注意

        x = x * ca

        # 时间注意
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        time_att = torch.cat([avg_out, max_out], dim=1)
        ta = self.sigmoid_time(self.conv_time(time_att))
        x = x * ta
        return x


# ======================= 简化的时序块（替换MS_CBTCBlock） =======================
class SimpleConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, reduction=8, use_htfa=True):
        super().__init__()
        # 使用最简单的卷积替换多分支结构
        self.conv = BTCConv1d(in_channels, out_channels, kernel_size)
        self.htfa = HTFA(out_channels, reduction=reduction) if use_htfa else None
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        # 残差连接
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        conv_out = self.conv(x)
        residual_out = self.residual(x)

        # 确保长度一致
        min_len = min(conv_out.shape[2], residual_out.shape[2])
        conv_out = conv_out[:, :, :min_len]
        residual_out = residual_out[:, :, :min_len]

        out = conv_out + residual_out
        if self.htfa:
            out = self.htfa(out)
        return self.relu(self.bn(out))


# ======================= 轻量化 Mamba 模块 =======================
class LightMambaBlock(nn.Module):
    def __init__(self, d_model, kernel_size=3):
        super().__init__()
        self.proj = nn.Linear(d_model, 3 * d_model)
        self.conv = nn.Conv1d(d_model, d_model, kernel_size, padding=kernel_size // 2, groups=d_model)
        self.glu = nn.GLU(dim=1)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):  # x: (B, T, C)
        res = x
        x = self.norm(x)
        params = self.proj(x)
        A, B, C = params.chunk(3, dim=-1)
        x_t = x.transpose(1, 2)
        conv_out = self.conv(x_t * A.sigmoid().transpose(1, 2))
        gate = (conv_out * B.transpose(1, 2)).sigmoid()
        out = gate * C.transpose(1, 2)
        return out.transpose(1, 2) + res


# ======================= MCTMNet 主模型（消融版本） =======================
class MCTMNet(nn.Module):
    def __init__(self, num_channels, num_timesteps, num_classes,
                 noise_std=0.05, htfa_reduction=8, use_htfa=True,
                 mamba_kernel_size=3):
        super().__init__()
        self.noise_std = noise_std
        self.htfa_reduction = htfa_reduction
        self.use_htfa = use_htfa
        self.mamba_kernel_size = mamba_kernel_size

        # 确定卷积核尺寸（简化版）
        t1 = num_timesteps
        t2 = num_timesteps // 2
        t3 = num_timesteps // 4
        k1 = min(max(3, num_timesteps // 8), t1)
        k2 = min(max(3, num_timesteps // 16), t2)
        k3 = min(max(3, num_timesteps // 32), t3)
        # 确保卷积核为奇数
        self.kernel_size1 = k1 if k1 % 2 == 1 else k1 - 1
        self.kernel_size2 = k2 if k2 % 2 == 1 else k2 - 1
        self.kernel_size3 = k3 if k3 % 2 == 1 else k3 - 1

        # 三层特征提取（使用简化的卷积块）
        self.block1 = SimpleConvBlock(num_channels, 32, kernel_size=self.kernel_size1,
                                      reduction=htfa_reduction, use_htfa=use_htfa)
        self.block2 = SimpleConvBlock(32, 64, kernel_size=self.kernel_size2,
                                      reduction=htfa_reduction, use_htfa=use_htfa)
        self.block3 = SimpleConvBlock(64, 128, kernel_size=self.kernel_size3,
                                      reduction=htfa_reduction, use_htfa=use_htfa)

        # 轻量化 Mamba 模块 + 分类头
        self.mamba = LightMambaBlock(128, kernel_size=mamba_kernel_size)
        self.fc_input_size = 128 * (num_timesteps // 8)
        self.fc = nn.Linear(self.fc_input_size, num_classes)

    def forward(self, x):
        if self.training:
            noise = torch.randn_like(x) * self.noise_std
            x = x + noise

        x = self.block1(x)
        x = F.max_pool1d(x, 2)
        x = self.block2(x)
        x = F.max_pool1d(x, 2)
        x = self.block3(x)
        x = F.max_pool1d(x, 2)

        x = x.transpose(1, 2)
        x = self.mamba(x)
        x = x.transpose(1, 2)
        x = x.view(x.size(0), -1)
        return self.fc(x)


# ======================= 测试 =======================
if __name__ == "__main__":
    for num_timesteps in [250, 512, 1024]:
        model = MCTMNet(num_channels=61, num_timesteps=num_timesteps, num_classes=4)
        print(f"\n时间步长: {num_timesteps}")
        print(f"第一层卷积核: {model.kernel_size1}")
        print(f"第二层卷积核: {model.kernel_size2}")
        print(f"第三层卷积核: {model.kernel_size3}")
        x = torch.randn(8, 61, num_timesteps)
        y = model(x)
        print(f"输出形状: {y.shape}")