import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import scipy.io as scio

# 把路径加进去,方便引用 utils
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# ================= 配置区 =================
dataset_name = 'GearBox.BJUT' # 或者 'GearBox.BJUT'
source_domain = '1200'        # 选一个工况
sigma = 0.1                   # 这里填你目前使用的参数
sample_index = 0              # 选择第几个样本进行可视化
# ========================================

def freq_aug_demo(x, sigma=0.1):
    """
    这是你 DWCN.py 里的增强逻辑副本
    """
    # 转为 tensor
    x = torch.tensor(x).float().unsqueeze(1) # [N, 1, L]
    
    # FFT
    fft_x = torch.fft.rfft(x, dim=-1)
    amp = torch.abs(fft_x)
    pha = torch.angle(fft_x)
    
    # 模拟增强：纯缩放
    batch_size = x.size(0)
    scale = (torch.rand(batch_size, 1, 1) - 0.5) * 2 * sigma + 1.0
    amp_aug = amp * scale
    
    # 逆变换
    aug_fft = amp_aug * torch.exp(1j * pha)
    x_aug = torch.fft.irfft(aug_fft, n=x.shape[-1], dim=-1)
    
    return x_aug.squeeze(1).numpy(), amp[0,0,:].numpy(), amp_aug[0,0,:].numpy()

def main():
    # 1. 加载真实数据
    print(f"Loading data from {dataset_name}...")
    
    # 构建数据文件路径
    data_dir = os.path.join(parent_dir, 'data', dataset_name)
    if dataset_name == 'Bearing.BJTU':
        file_name = f'BearingData_{source_domain}_5.mat'
        data_key = f'BearingData_{source_domain}_5'
    else:  # GearBox.BJUT
        file_name = f'GearData_{source_domain}_5.mat'
        data_key = f'GearData_{source_domain}_5'
    
    file_path = os.path.join(data_dir, file_name)
    print(f"Loading file: {file_path}")
    
    # 读取 .mat 文件
    data_mat = scio.loadmat(file_path)
    data = data_mat[data_key]
    
    # 提取一个样本 (取前1024个点，这是你模型使用的长度)
    original_signal = data[sample_index:sample_index+1, :1024]  # [1, 1024]
    print(f"Loaded sample shape: {original_signal.shape}, class label: {data[sample_index, -1]}")
    
    # 2. 运行增强
    aug_signal, orig_spec, aug_spec = freq_aug_demo(original_signal, sigma=sigma)
    
    # 3. 画图
    plt.figure(figsize=(12, 6))
    
    # 子图1：时域波形
    plt.subplot(2, 1, 1)
    plt.plot(original_signal[0], label='Original', alpha=0.8)
    plt.plot(aug_signal[0], label='Augmented (sigma={})'.format(sigma), alpha=0.6, linestyle='--')
    plt.title("Time Domain Waveform")
    plt.legend()
    
    # 子图2：频域频谱 (只画前一半频率)
    plt.subplot(2, 1, 2)
    plt.plot(orig_spec[:200], label='Original Spectrum', color='blue')
    plt.plot(aug_spec[:200], label='Augmented Spectrum', color='orange', linestyle='--')
    plt.title("Frequency Domain Spectrum (Zoomed In)")
    plt.legend()
    plt.tight_layout()
    
    save_path = 'vis_fft_check.png'
    plt.savefig(save_path)
    print(f"图已保存至: {save_path}")
    plt.show()

if __name__ == '__main__':
    main()