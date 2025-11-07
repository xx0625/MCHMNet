import mne
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from mne.preprocessing import ICA, create_eog_epochs, create_ecg_epochs
from mne_icalabel import label_components  # 导入 label_components
import matplotlib.pyplot as plt
import os  # 用于处理文件路径
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)
mne.set_log_level('WARNING')

# 修改 random_state
random_state = 42


def extract_psd_features(raw, sfreq, freq_bands, n_fft, channel_names):
    """提取PSD特征并生成对应的列名"""
    all_band_features = []
    feature_names = []

    # 为每个频段生成特征名称
    for (fmin, fmax) in freq_bands:
        band_name = f"{fmin}-{fmax}hz"
        # 为该频段的每个通道添加名称
        for ch in channel_names:
            feature_names.append(f"{ch}_{band_name}")

        band_psd_means = []
        for start_idx in range(0, len(raw.times), n_fft):
            end_idx = start_idx + n_fft
            if end_idx > len(raw.times):
                break
            epoch_data = raw.get_data(start=start_idx, stop=end_idx)
            psd, _ = mne.time_frequency.psd_array_welch(
                epoch_data, sfreq=sfreq, fmin=fmin, fmax=fmax, n_fft=n_fft)
            psd_mean = psd.mean(axis=1)
            band_psd_means.append(psd_mean)
        all_band_features.append(np.array(band_psd_means))

    combined_features = np.hstack(all_band_features)
    combined_features *= 10 ** 12

    return combined_features, feature_names


def process_subject(subject_id, session, task_type, file_path, channel_names_of_interest, freq_bands, tmin, tmax):
    """处理单个被试的单个任务数据"""
    print(f"处理被试 {subject_id}，会话 {session}，任务 {task_type}: {file_path}")

    try:
        raw = mne.io.read_raw_eeglab(file_path, preload=True)
    except FileNotFoundError:
        print(f"文件 {file_path} 未找到，跳过。")
        return None, None
    except Exception as e:
        print(f"读取文件 {file_path} 时出错: {e}，跳过。")
        return None, None

    # 检查通道是否存在
    available_channels = raw.ch_names
    missing_channels = [ch for ch in channel_names_of_interest if ch not in available_channels]
    if missing_channels:
        print(f"缺失通道: {missing_channels}，将从分析中排除。")
        channel_names = [ch for ch in channel_names_of_interest if ch in available_channels]
    else:
        channel_names = channel_names_of_interest.copy()

    if not channel_names:
        print("没有可用的感兴趣通道，跳过此数据。")
        return None, None

    # 选择感兴趣的 EEG 通道
    raw.pick(channel_names)

    # 设置电极布局为 10-20 系统标准
    raw.set_montage('standard_1020')

    # 对原始数据进行带通滤波和陷波滤波
    raw.notch_filter(freqs=[48, 52])
    raw.notch_filter(freqs=[58, 62])
    raw.filter(1, 100, fir_design='firwin')  # 调整为1-100Hz以符合ICLabel要求

    # 降采样
    raw.resample(sfreq=250, npad="auto")

    # 裁剪原始数据到指定的时间段并加载到内存
    raw.crop(tmin, tmax, include_tmax=False).load_data()

    # 设置 EEG 数据的参考电极为平均参考
    raw.set_eeg_reference(ref_channels='average', projection=False)

    # 创建并拟合 ICA 对象
    ica = ICA(n_components=len(channel_names) - 1,
              random_state=random_state,
              max_iter='auto',
              method="infomax",
              fit_params=dict(extended=True))

    # 拟合 ICA
    print("拟合 ICA 中...")
    try:
        ica.fit(raw)
    except Exception as e:
        print(f"ICA 拟合时出错: {e}，跳过。")
        return None, None
    print("ICA 拟合完成。")

    # 使用 mne_icalabel 对 ICA 组件进行分类
    print("使用 mne_icalabel 进行 ICA 组件分类...")
    try:
        labels = label_components(raw, ica, method='iclabel')
    except Exception as e:
        print(f"组件分类时出错: {e}，跳过。")
        return None, None
    print("组件分类完成。")

    # 定义需要排除的类别及其概率阈值
    artifact_categories = ['eye blink', 'muscle artifact', "heart beat"]
    probability_threshold = 0.9

    # 根据分类结果排除高概率的伪影组件
    exclude_inds = []
    for idx, label in enumerate(labels['labels']):
        # 获取每个组件的预测概率
        prob = labels['y_pred_proba'][idx]
        # 检查当前组件是否属于需要排除的类别且概率超过阈值
        if label in artifact_categories and prob > probability_threshold:
            exclude_inds.append(idx)

    # 设置并应用排除的组件
    if exclude_inds:
        ica.exclude = exclude_inds
        print(f"排除的组件索引: {ica.exclude}")
        try:
            ica.apply(raw)
            print("应用 ICA 排除伪影组件完成。")
        except Exception as e:
            print(f"应用 ICA 时出错: {e}")
    else:
        print("未检测到需要排除的伪影组件。")

    # 提取 PSD 特征（修改部分：同时获取特征名称）
    n_fft = int(1.0 * raw.info['sfreq'])
    psd_features, feature_names = extract_psd_features(raw, raw.info['sfreq'], freq_bands, n_fft, channel_names)

    # 提取时域数据（已使用通道名作为列名）
    try:
        # 获取处理后的数据（channels x time）并转置为 (time x channels)
        time_data = raw.get_data().T
        time_df = pd.DataFrame(time_data, columns=channel_names)
        time_df['subject_id'] = subject_id
        time_df['session'] = session
        time_df['task_type'] = task_type
    except Exception as e:
        print(f"处理时域数据时出错: {e}")
        time_df = None

    # 处理PSD特征数据（修改部分：使用生成的特征名称）
    try:
        psd_df = pd.DataFrame(psd_features, columns=feature_names)
        psd_df['subject_id'] = subject_id
        psd_df['session'] = session
        psd_df['task_type'] = task_type
    except Exception as e:
        print(f"处理PSD特征时出错: {e}")
        psd_df = None

    return psd_df, time_df


# 配置参数
subject_ids = range(1, 30)  # sub-01 到 sub-29
sessions = ['S1', 'S2', 'S3']  # 三个会话
task_types = ['RS_Beg_EO', 'zeroBACK', 'oneBACK', 'twoBACK']  # 任务类型

# 基础路径
base_path = r"E:/Desktop/研究生/研一/data/COG"

# 按脑区排列的电极列表
channel_names_of_interest = [
    "Fp1", "Fp2",

    # 前额叶过渡区
    "AF7", "AF3", "AFz", "AF4", "AF8",

    # 额叶
    "F7", "F5", "F3", "F1", "Fz", "F2", "F4", "F6", "F8",

    # 额中央过渡区
    "FT9", "FT7", "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6", "FT8", "FT10",

    # 中央区
    "T7", "C5", "C3", "C1", "C2", "C4", "C6", "T8",

    # 中央顶叶过渡区
    "TP7", "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6", "TP8",

    # 顶叶
    "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8",

    # 顶枕叶过渡区
    "PO7", "PO3", "POz", "PO4", "PO8",

    # 枕叶
    "O1", "Oz", "O2"
]

freq_bands = [(4, 7), (8, 12), (13, 30), (30, 45)]  # 定义感兴趣的频段
tmin, tmax = 5, 55  # 时间范围

# 创建保存结果的目录
output_dir = r"E:/Desktop/EEG_results"
os.makedirs(output_dir, exist_ok=True)

# 处理所有组合
for session in sessions:
    for task_type in task_types:
        print(f"\n===== 开始处理 会话 {session}，任务 {task_type} =====")

        all_psd_data = []
        all_time_data = []

        for subject_id in subject_ids:
            # 构建文件路径
            sub_dir = f"sub-{subject_id:02d}"
            file_name = f"{task_type}.set" if task_type != "RS_Beg_EO" else f"{task_type}.set"

            file_path = os.path.join(
                base_path,
                sub_dir,
                sub_dir,
                f"ses-{session}",
                "eeg",
                file_name
            )

            # 处理单个被试数据
            psd_df, time_df = process_subject(
                subject_id, session, task_type, file_path,
                channel_names_of_interest, freq_bands, tmin, tmax
            )

            if psd_df is not None:
                all_psd_data.append(psd_df)
            if time_df is not None:
                all_time_data.append(time_df)

        # 保存当前会话和任务类型的结果
        if all_psd_data:
            combined_psd = pd.concat(all_psd_data, ignore_index=True)
            psd_csv_path = os.path.join(output_dir, f"matb_{session}_{task_type}_psd.csv")
            combined_psd.to_csv(psd_csv_path, index=False)
            print(f"会话 {session}，任务 {task_type} 的PSD特征数据已保存到 {psd_csv_path}")

        if all_time_data:
            combined_time = pd.concat(all_time_data, ignore_index=True)
            time_csv_path = os.path.join(output_dir, f"matb_{session}_{task_type}_time.csv")
            combined_time.to_csv(time_csv_path, index=False)
            print(f"会话 {session}，任务 {task_type} 的时域数据已保存到 {time_csv_path}")

print("\n所有数据处理完成。")
