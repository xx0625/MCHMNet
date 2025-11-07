import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from tqdm import tqdm
import sys
from sklearn.preprocessing import StandardScaler
# 导入thop库用于计算参数量和FLOPs



# 简单的MLP模型定义
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


def setup_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


setup_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def load_and_reshape(csv_path, session_id, label, n_timesteps=500, channel=61, window_type='hamming'):
    """加载数据并重塑为模型输入格式"""
    df = pd.read_csv(csv_path)

    # 前61列是EEG数据，倒数第三列是身份
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
    # MLP不需要通道在前，保持(样本数, 时间步, 通道数)格式
    return reshaped_data, np.array(reshaped_labels), reshaped_identities


def prepare_data(train_sessions, test_session, base_path, n_timesteps=256, channel=14):
    """准备训练、验证和测试数据"""
    train_X, train_y, train_identities = [], [], []
    val_X, val_y, val_identities = [], [], []

    # 加载训练会话数据
    for session in train_sessions:
        session_files = [f for f in os.listdir(base_path)
                         if f.startswith(f"{session}_") and f.endswith(".csv")]

        for file in session_files:
            try:
                # 分割文件名获取标签（格式如 "session_label.csv"）
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

            # 划分训练集和验证集（9:1）
            indices = np.random.permutation(len(data))
            num_val = int(len(data) * 0.1)

            val_X.append(data[indices[:num_val]])
            val_y.append(labels[indices[:num_val]])
            val_identities.extend([identities[i] for i in indices[:num_val]])

            train_X.append(data[indices[num_val:]])
            train_y.append(labels[indices[num_val:]])
            train_identities.extend([identities[i] for i in indices[num_val:]])

    # 合并训练数据
    train_X = np.vstack(train_X) if train_X else np.array([])
    train_y = np.hstack(train_y) if train_y else np.array([])
    train_identities = np.array(train_identities)

    # 合并验证数据
    val_X = np.vstack(val_X) if val_X else np.array([])
    val_y = np.hstack(val_y) if val_y else np.array([])
    val_identities = np.array(val_identities)

    # 加载测试会话数据
    test_X, test_y, test_identities = [], [], []
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


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, path='./best/checkpoint.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


if __name__ == "__main__":
    base_path = "C:/xx/Chinese/data/preprocess_data"  # 修改为你的数据目录
    sessions = [1, 2, 3]  # 三个session
    n_timesteps = 250  # 时间步长
    batch_size = 128
    max_epochs = 200
    patience = 20
    channel = 61  # 通道数
    all_acc, all_f1 = [], []
    flops_calculated = False  # 标记是否已计算过参数量和FLOPs

    # 创建保存最佳模型的目录
    os.makedirs('./best', exist_ok=True)

    # MLP参数设置
    input_dim = channel * n_timesteps  # 输入维度 = 通道数 × 时间步长
    hidden_dim1 = 512  # 第一个隐藏层维度
    hidden_dim2 = 128  # 第二个隐藏层维度

    for test_session in sessions:
        setup_seed(42)
        print(f"\n=== 留一session out - 测试session: {test_session} (Device: {device}) ===")
        train_sessions = [s for s in sessions if s != test_session]

        # 准备数据
        X_train, y_train, train_identities, X_val, y_val, val_identities, X_test, y_test, test_identities = prepare_data(
            train_sessions, test_session, base_path, n_timesteps, channel
        )
        # 检查数据是否加载成功
        if len(X_train) == 0 or len(X_test) == 0:
            print(f"警告: 未能加载到session {test_session} 的有效数据，已跳过")
            continue

        # 转换为PyTorch张量 (MLP输入格式为 [样本数, 时间步, 通道数])
        X_train = torch.from_numpy(X_train).float().to(device)
        y_train = torch.from_numpy(y_train).long().to(device)
        X_val = torch.from_numpy(X_val).float().to(device)
        y_val = torch.from_numpy(y_val).long().to(device)
        X_test = torch.from_numpy(X_test).float().to(device)
        y_test = torch.from_numpy(y_test).long().to(device)

        # 初始化MLP模型（四分类）
        model = SimpleMLP(
            input_dim=input_dim,
            hidden_dim1=hidden_dim1,
            hidden_dim2=hidden_dim2,
            nb_classes=4
        ).to(device)



        # 定义优化器和损失函数
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()  # 多分类交叉熵损失

        # 早停机制
        early_stopping = EarlyStopping(
            patience=patience,
            verbose=True,
            path=f'./best/checkpoint_session{test_session}.pt'
        )

        # 训练模型
        for epoch in tqdm(range(max_epochs), desc=f"训练进度 (测试session: {test_session})"):
            model.train()
            running_loss = 0.0
            total = 0
            correct = 0
            permutation = torch.randperm(X_train.size(0))

            for i in range(0, X_train.size(0), batch_size):
                indices = permutation[i:i + batch_size]
                batch_X, batch_y = X_train[indices], y_train[indices]
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()

            epoch_loss = running_loss / (max(1, X_train.size(0) // batch_size))
            epoch_acc = correct / total if total > 0 else 0
            print(f'Epoch {epoch + 1}, 训练准确率: {epoch_acc:.4f}, 训练损失: {epoch_loss:.4f}')

            # 验证
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val)
                val_loss = criterion(val_outputs, y_val)
                _, val_predicted = torch.max(val_outputs, 1)
                val_acc = accuracy_score(y_val.cpu().numpy(), val_predicted.cpu().numpy())
                print(f'验证准确率: {val_acc:.4f}, 验证损失: {val_loss.item():.4f}')

            # 早停检查
            early_stopping(val_loss.item(), model)
            if early_stopping.early_stop:
                print(f"在第 {epoch + 1} 轮提前停止训练")
                break

        # 加载最佳模型并测试
        model.load_state_dict(torch.load(f'./best/checkpoint_session{test_session}.pt'))
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test)
            _, predicted = torch.max(test_outputs, 1)
            acc = accuracy_score(y_test.cpu().numpy(), predicted.cpu().numpy())
            f1 = f1_score(y_test.cpu().numpy(), predicted.cpu().numpy(), average='macro')  # 多分类使用macro F1
            cm = confusion_matrix(y_test.cpu().numpy(), predicted.cpu().numpy())

            all_acc.append(acc)
            all_f1.append(f1)
            print(f"\n测试session {test_session} 结果 - 准确率: {acc:.4f}, F1分数: {f1:.4f}")
            print(f"混淆矩阵:\n{cm}")

    # 输出最终结果
    print("\n=== 留一session out 交叉验证最终结果 ===")
    for session, acc, f1 in zip(sessions, all_acc, all_f1):
        print(f"Session {session} 作为测试集 - 准确率: {acc:.4f}, F1分数: {f1:.4f}")
    if all_acc:  # 确保有有效数据
        print(f"平均准确率: {np.mean(all_acc):.4f} ± {np.std(all_acc):.4f}")
        print(f"平均F1分数: {np.mean(all_f1):.4f} ± {np.std(all_f1):.4f}")