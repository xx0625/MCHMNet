import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import mne
import re
import pandas as pd
from mne.viz import plot_topomap
import sys

sys.path.append(r'C:/xx/Chinese/code/models')
from MCTMNet_explain import MCTMNet

# --- 设置matplotlib支持中文 ---
plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# --- 模型和数据参数 (请根据你的设置修改) ---
MODEL_PATH = './best/checkpoint_unknown_model_session1.pt'  # 替换为你最好的模型路径
DATA_PATH = "C:/xx/Chinese/data/preprocess_data"  # 你的数据路径
TEST_SESSION = 1  # 要加载哪个session的模型
N_TIMESTEPS = 250
CHANNEL = 61
N_CLASSES = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHANNEL_NAMES_61 = [
    'Fp1', 'Fpz', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6',
    'T7', 'C3', 'Cz', 'C4', 'T8', 'CP5', 'CP1', 'CP2', 'CP6', 'P7', 'P3', 'Pz',
    'P4', 'P8', 'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'O1', 'Oz', 'O2', 'AF7',
    'AF3', 'AFz', 'AF4', 'AF8', 'F5', 'F1', 'F2', 'F6', 'FC3', 'FCz', 'FC4',
    'C5', 'C1', 'C2', 'C6', 'CP3', 'CPz', 'CP4', 'P5', 'P1', 'P2', 'P6',
    'PO5', 'PO6', 'TP7', 'TP8'
]

# 检查通道数量是否匹配
if len(CHANNEL_NAMES_61) != CHANNEL:
    raise ValueError(f"通道名称列表长度 ({len(CHANNEL_NAMES_61)}) 与模型通道数 ({CHANNEL}) 不匹配")


def load_and_reshape(csv_path, session_id, label, n_timesteps=250, channel=61, window_type='hamming'):
    """加载数据并重塑为模型输入格式"""
    df = pd.read_csv(csv_path)

    # 前61列是信息，之后是EEG数据，倒数第三列是身份
    data = df.iloc[:, :channel].values * 1e6
    identities = df.iloc[:, -3].values  # 身份信息（倒数第三列）

    # 创建带会话和标签信息的身份标识
    full_identities = [f"s{session_id}_id{int(identity)}_label{label}"
                       for identity in identities]

    # 选择窗函数
    if window_type == 'hamming':
        window = np.hamming(n_timesteps)
    elif window_type == 'hanning':
        window = np.hanning(n_timesteps)
    elif window_type == 'blackman':
        window = np.blackman(n_timesteps)
    else:
        window = np.ones(n_timesteps)

    reshaped_data = []
    reshaped_labels = []
    reshaped_identities = []

    # 滑动窗口分割数据
    for i in range(0, len(data) - n_timesteps + 1, n_timesteps):
        windowed_segment = data[i:i + n_timesteps] * window[:, np.newaxis]
        reshaped_data.append(windowed_segment)
        reshaped_labels.append(label)
        reshaped_identities.append(full_identities[i])

    reshaped_data = np.array(reshaped_data)
    return reshaped_data.transpose(0, 2, 1), np.array(reshaped_labels), reshaped_identities


def prepare_data_with_ids(train_sessions, test_session, base_path, n_timesteps=256, channel=14):
    """准备数据并保留身份信息"""
    train_X, train_y, train_identities = [], [], []
    val_X, val_y, val_identities = [], [], []
    test_X, test_y, test_identities = [], [], []

    # 加载训练会话数据
    for session in train_sessions:
        session_files = [f for f in os.listdir(base_path)
                         if f.startswith(f"{session}_") and f.endswith(".csv")]

        for file in session_files:
            try:
                label = int(file.split('_')[1].split('.')[0])
                if label < 0 or label > 3:
                    print(f"警告: 文件 {file} 包含无效标签 {label}，已跳过")
                    continue
            except (IndexError, ValueError):
                print(f"警告: 无法从文件名 {file} 提取标签，已跳过")
                continue

            file_path = os.path.join(base_path, file)
            data, labels, identities = load_and_reshape(
                file_path, session, label, n_timesteps, channel
            )

            indices = np.random.permutation(len(data))
            num_val = int(len(data) * 0.1)

            val_X.append(data[indices[:num_val]])
            val_y.append(labels[indices[:num_val]])
            val_identities.extend([identities[i] for i in indices[:num_val]])

            train_X.append(data[indices[num_val:]])
            train_y.append(labels[indices[num_val:]])
            train_identities.extend([identities[i] for i in indices[num_val:]])

    # 合并训练和验证数据
    train_X = np.vstack(train_X) if train_X else np.array([])
    train_y = np.hstack(train_y) if train_y else np.array([])
    train_identities = np.array(train_identities)

    val_X = np.vstack(val_X) if val_X else np.array([])
    val_y = np.hstack(val_y) if val_y else np.array([])
    val_identities = np.array(val_identities)

    # 加载测试会话数据
    test_files = [f for f in os.listdir(base_path)
                  if f.startswith(f"{test_session}_") and f.endswith(".csv")]

    for file in test_files:
        try:
            label = int(file.split('_')[1].split('.')[0])
            if label < 0 or label > 3:
                print(f"警告: 测试文件 {file} 包含无效标签 {label}，已跳过")
                continue
        except (IndexError, ValueError):
            print(f"警告: 无法从测试文件名 {file} 提取标签，已跳过")
            continue

        file_path = os.path.join(base_path, file)
        data, labels, identities = load_and_reshape(
            file_path, test_session, label, n_timesteps, channel
        )

        test_X.append(data)
        test_y.append(labels)
        test_identities.extend(identities)

    # 合并测试数据
    test_X = np.vstack(test_X) if test_X else np.array([])
    test_y = np.hstack(test_y) if test_y else np.array([])
    test_identities = np.array(test_identities)

    return train_X, train_y, train_identities, val_X, val_y, val_identities, test_X, test_y, test_identities


def load_model_and_data(model_path, test_session, base_path, n_timesteps, channel):
    """加载模型和测试数据（包含身份信息）"""
    print(f"正在加载模型: {model_path}")

    # 初始化模型
    model = MCTMNet(
        num_channels=channel,
        num_timesteps=n_timesteps,
        num_classes=N_CLASSES,
        noise_std=0.03,
        htfa_reduction=4,
        use_htfa=True,
        mamba_kernel_size=3
    ).to(DEVICE)

    # 加载训练好的权重
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()  # 设置为评估模式

    print("正在加载数据...")
    train_sessions = [s for s in [1, 2, 3] if s != test_session]
    _, _, _, _, _, _, X_test, y_test, test_identities = prepare_data_with_ids(
        train_sessions, test_session, base_path, n_timesteps, channel
    )

    if len(X_test) == 0:
        raise Exception("未加载到测试数据，请检查路径和session ID")

    # 转换为PyTorch张量
    X_test = torch.from_numpy(X_test).float().to(DEVICE)
    y_test = torch.from_numpy(y_test).long().to(DEVICE)

    print(f"数据加载完毕. X_test shape: {X_test.shape}")
    return model, X_test, y_test, test_identities


def get_saliency_map(model, input_tensor, target_class):
    """计算显著图 (输入梯度)"""
    input_tensor.requires_grad = True

    # 前向传播
    output = model(input_tensor)

    # 清零梯度
    model.zero_grad()

    # 获取目标类别的得分
    score = output[0, target_class]

    # 反向传播
    score.backward()

    # 获取梯度并取绝对值
    saliency = input_tensor.grad.data.abs().squeeze().cpu().numpy()

    return saliency


def plot_saliency_heatmap(saliency_map, person_id, label, true_label, pred_label):
    """绘制2D显著图热力图"""
    plt.figure(figsize=(12, 6))
    plt.imshow(saliency_map, aspect='auto', cmap='hot')
    plt.colorbar(label='梯度绝对值 (重要性)')
    plt.xlabel('时间步 (Timesteps)')
    plt.ylabel('EEG 通道 (Channels)')
    plt.title(f'被试 {person_id} 标签 {label} 显著图 (真实: {true_label}, 预测: {pred_label})')
    plt.tight_layout()
    os.makedirs(f'./results/person_{person_id}', exist_ok=True)
    plt.savefig(f'./results/person_{person_id}/saliency_map_label_{label}.png')
    print(f'已保存: ./results/person_{person_id}/saliency_map_label_{label}.png')
    plt.close()


def plot_spatial_importance(saliency_map, channel_names, person_id, label):
    """绘制空间重要性地形图"""
    # 沿时间维度计算平均重要性
    spatial_importance = np.mean(saliency_map, axis=1)

    # 创建 MNE Info 对象
    info = mne.create_info(ch_names=channel_names, sfreq=250, ch_types='eeg')

    # 设置电极位置
    montage = mne.channels.make_standard_montage('standard_1020')
    info.set_montage(montage, on_missing='ignore')

    # 绘制地形图
    fig, ax = plt.subplots(figsize=(6, 5))
    im, _ = plot_topomap(spatial_importance, info, axes=ax, cmap='Reds', show=False, contours=6)

    # 添加颜色条
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('平均重要性')

    ax.set_title(f'被试 {person_id} 标签 {label} 空间重要性')
    plt.tight_layout()
    os.makedirs(f'./results/person_{person_id}', exist_ok=True)
    plt.savefig(f'./results/person_{person_id}/spatial_topo_map_label_{label}.png')
    print(f'已保存: ./results/person_{person_id}/spatial_topo_map_label_{label}.png')
    plt.close()


def plot_temporal_importance(saliency_map, person_id, label):
    """绘制时间重要性折线图"""
    # 沿通道维度计算平均重要性
    temporal_importance = np.mean(saliency_map, axis=0)

    plt.figure(figsize=(10, 4))
    plt.plot(temporal_importance)
    plt.xlabel('时间步 (Timesteps)')
    plt.ylabel('平均重要性')
    plt.title(f'被试 {person_id} 标签 {label} 时间重要性')
    plt.grid(True, linestyle='--')
    plt.tight_layout()
    os.makedirs(f'./results/person_{person_id}', exist_ok=True)
    plt.savefig(f'./results/person_{person_id}/temporal_importance_label_{label}.png')
    print(f'已保存: ./results/person_{person_id}/temporal_importance_label_{label}.png')
    plt.close()


def plot_internal_time_attention(model, person_id, label):
    """绘制内部HTFA时间注意力"""
    try:
        ta_b1 = model.block1.htfa.last_ta.squeeze().cpu().detach().numpy()
        ta_b2 = model.block2.htfa.last_ta.squeeze().cpu().detach().numpy()
        ta_b3 = model.block3.htfa.last_ta.squeeze().cpu().detach().numpy()
    except AttributeError:
        print("错误：无法从模型中提取 'last_ta'。")
        print("请确保你已经修改了 MCTMNet.py 中的 HTFA 类。")
        return

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 9), sharex=False)

    ax1.plot(ta_b1, color='blue')
    ax1.set_title(f'Block 1 内部时间注意力 (长度: {len(ta_b1)})')
    ax1.set_ylabel('注意力权重')
    ax1.grid(True, linestyle='--')

    ax2.plot(ta_b2, color='green')
    ax2.set_title(f'Block 2 内部时间注意力 (长度: {len(ta_b2)})')
    ax2.set_ylabel('注意力权重')
    ax2.grid(True, linestyle='--')

    ax3.plot(ta_b3, color='red')
    ax3.set_title(f'Block 3 内部时间注意力 (长度: {len(ta_b3)})')
    ax3.set_ylabel('注意力权重')
    ax3.set_xlabel('时间步 (下采样后)')
    ax3.grid(True, linestyle='--')

    fig.suptitle(f'被试 {person_id} 标签 {label} 的内部HTFA时间注意力', fontsize=16, y=1.02)
    plt.tight_layout()
    os.makedirs(f'./results/person_{person_id}', exist_ok=True)
    plt.savefig(f'./results/person_{person_id}/internal_time_attention_label_{label}.png')
    print(f'已保存: ./results/person_{person_id}/internal_time_attention_label_{label}.png')
    plt.close()


def get_person_id(identity_str):
    """从身份标识字符串中提取被试ID"""
    # 身份标识格式: s{session}_id{identity}_label{label}
    match = re.search(r'id(\d+)', identity_str)
    if match:
        return match.group(1)
    return None


def analyze_person(model, X_test, y_test, test_identities, person_id):
    """分析指定被试的4种标签数据"""
    print(f"\n--- 开始分析被试 {person_id} ---")

    # 获取该被试的所有样本索引
    person_indices = [i for i, ident in enumerate(test_identities)
                      if get_person_id(ident) == person_id]

    if not person_indices:
        print(f"未找到被试 {person_id} 的数据")
        return

    # 获取该被试的所有标签
    person_labels = set(y_test[i].item() for i in person_indices)
    print(f"被试 {person_id} 包含的标签: {person_labels}")

    # 检查是否有完整的4种标签
    if len(person_labels) < 4:
        print(f"被试 {person_id} 缺少部分标签，仅包含 {len(person_labels)} 种标签，跳过分析")
        return

    # 为每种标签找到一个分类正确的样本
    with torch.no_grad():
        outputs = model(X_test[person_indices])
        _, predicted = torch.max(outputs.data, 1)

    # 存储每种标签对应的样本索引
    label_samples = {}

    for i, idx in enumerate(person_indices):
        true_label = y_test[idx].item()
        pred_label = predicted[i].item()

        # 只保存分类正确的样本
        if true_label not in label_samples and true_label == pred_label:
            label_samples[true_label] = {
                'index': idx,
                'true_label': true_label,
                'pred_label': pred_label
            }

    # 检查是否每种标签都有可用样本
    missing_labels = [label for label in range(4) if label not in label_samples]
    if missing_labels:
        print(f"被试 {person_id} 缺少以下标签的正确分类样本: {missing_labels}，跳过分析")
        return

    # 为每种标签生成可视化结果
    for label in sorted(label_samples.keys()):
        sample_info = label_samples[label]
        print(f"\n--- 分析被试 {person_id} 的标签 {label} ---")
        print(
            f"样本索引: {sample_info['index']}, 真实标签: {sample_info['true_label']}, 预测标签: {sample_info['pred_label']}")

        # 准备单个样本
        sample_x = X_test[sample_info['index']].unsqueeze(0)

        # 生成显著图
        print(f"(1/4) 正在计算标签 {label} 的显著图...")
        saliency_map = get_saliency_map(model, sample_x, sample_info['true_label'])

        # 绘制各类图表
        plot_saliency_heatmap(saliency_map, person_id, label,
                              sample_info['true_label'], sample_info['pred_label'])

        print(f"(2/4) 正在绘制标签 {label} 的空间重要性地形图...")
        plot_spatial_importance(saliency_map, CHANNEL_NAMES_61, person_id, label)

        print(f"(3/4) 正在绘制标签 {label} 的时间重要性图...")
        plot_temporal_importance(saliency_map, person_id, label)

        print(f"(4/4) 正在绘制标签 {label} 的内部时间注意力...")
        # 重新运行前向传播以获取注意力信息
        _ = model(sample_x)
        plot_internal_time_attention(model, person_id, label)


if __name__ == "__main__":
    # 确保保存结果的根目录存在
    os.makedirs('./results', exist_ok=True)

    # 加载模型和数据（包含身份信息）
    model, X_test, y_test, test_identities = load_model_and_data(
        MODEL_PATH, TEST_SESSION, DATA_PATH, N_TIMESTEPS, CHANNEL
    )

    # 获取所有被试的ID
    all_person_ids = list(set(get_person_id(ident) for ident in test_identities
                              if get_person_id(ident) is not None))
    all_person_ids = [pid for pid in all_person_ids if pid is not None]

    print(f"\n测试集中共发现 {len(all_person_ids)} 个被试")

    # 找到第一个拥有4种完整标签的被试
    target_person_id = None
    for pid in all_person_ids:
        person_indices = [i for i, ident in enumerate(test_identities)
                          if get_person_id(ident) == pid]
        person_labels = set(y_test[i].item() for i in person_indices)
        if len(person_labels) == 4:
            target_person_id = pid
            break

    if target_person_id:
        print(f"找到拥有4种完整标签的被试: {target_person_id}")
        analyze_person(model, X_test, y_test, test_identities, target_person_id)
    else:
        print("未找到拥有4种完整标签的被试")

    print("\n--- 可解释性分析完成！所有图像已保存到 ./results/ 目录下 ---")