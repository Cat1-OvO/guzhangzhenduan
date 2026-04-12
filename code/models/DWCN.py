import pywt
import torch
import torch.nn as nn
from torch import optim
from torch.nn import functional as F
import numpy as np
from utils.custom_loss import instance_contrastive_loss

class SConv_1D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel):
        super(SConv_1D, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel), # 1D Convolution卷积层
            nn.BatchNorm1d(out_ch), # 批归一化层
            nn.ReLU(inplace=True),  # 激活函数ReLU
        )

    def forward(self, x):
        return self.conv(x)
# 扰动器
class LP(nn.Module):       
    def __init__(self, style_dim, num_features):
        super().__init__()

        self.norm = nn.InstanceNorm1d(num_features, affine=False)
        self.fc1 = nn.Linear(style_dim, num_features)
        self.fc2 = nn.Linear(style_dim, num_features)

    def forward(self, x, s1, s2):
        mu = x.mean(dim=2, keepdim=True)
        var = x.var(dim=2, keepdim=True)
        sig = (var + 0.1).sqrt()
        mu, sig = mu.detach(), sig.detach()
        x_normed = (x - mu) / sig

        h1 = (self.fc1(s1))
        h2 = (self.fc2(s2))

        gamma = h1.view(h1.size(0), h1.size(1), 1)
        beta = h2.view(h2.size(0), h2.size(1), 1)

        return x + (gamma*x_normed+ beta) , gamma , beta
# 小波卷积层
class DWConv(nn.Module):
    def __init__(self, wavelet='db4', num_channels=1):
        super(DWConv, self).__init__()
        self.wavelet=wavelet
        self.num_channels = num_channels
        self.l_filter, self.h_filter = self.get_wavelet_filters(self.wavelet)
        self.kernel_size = len(self.l_filter)
        self.mWDN1 = nn.Parameter(
            torch.cat([torch.tensor(self.l_filter).float().unsqueeze(0).repeat(1, 1, 1),
                       torch.tensor(self.h_filter).float().unsqueeze(0).repeat(1, 1, 1)]*self.num_channels, dim=0),
            requires_grad=True
        )
        self.dropout = nn.Dropout(p=0.1)
# 获取小波滤波器
    def get_wavelet_filters(self, wavelet):
        wavelet_obj = pywt.Wavelet(wavelet)
        return wavelet_obj.filter_bank[0], wavelet_obj.filter_bank[1]
# 小波变换前向传播
    def forward(self, input):
        batch_size, num_channels, signal_length = input.shape
        outsize = pywt.dwt_coeff_len(signal_length, self.kernel_size, mode="zero")
        p = 2 * (outsize - 1) - signal_length + self.kernel_size
        pad_left = p // 2
        pad_right = pad_left if p % 2 == 0 else pad_left + 1
        input_padded = F.pad(input, (pad_left, pad_right))
        freq = F.conv1d(input_padded, self.mWDN1, groups=num_channels, padding=0, stride=2)
        lp_out = freq[:, ::2, :].contiguous()
        hp_out = freq[:, 1::2, :].contiguous()
        all_out = torch.cat([lp_out, hp_out], dim=1)
        all_out = self.dropout(all_out)
        return lp_out, hp_out, all_out,self.mWDN1.detach().cpu()
# 分类器
class Classifier(nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.c=nn.Linear(256, num_classes)

    def forward(self, x):
        y=self.c(x)
        return y
# 特征提取器
class Fea_Extraction(nn.Module):
    def __init__(self, in_channel=1,wavelet=""):
        super(Fea_Extraction, self).__init__()
        num_c=16
        self.wavelet=wavelet
        self.wave_func_1 = DWConv(num_channels=in_channel,wavelet=self.wavelet)
        self.wave_func_2 = DWConv(num_channels=num_c,wavelet=self.wavelet)
        self.wave_func_3 = DWConv(num_channels=num_c*2,wavelet=self.wavelet)
        self.wave_func_4 = DWConv(num_channels=num_c*4,wavelet=self.wavelet)
        self.wave_func_5 = DWConv(num_channels=num_c*8,wavelet=self.wavelet)
        self.conv1 = SConv_1D(2, num_c, 3, )
        self.conv2 = SConv_1D(num_c*2,  num_c*2, 3,)
        self.conv3 = SConv_1D(num_c*4, num_c*4, 3, )
        self.conv4 = SConv_1D(num_c*8, num_c*8, 3, )
        self.conv5 = SConv_1D(num_c*16, num_c*16, 3, )
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
    def forward(self, input):
        # Layer1
        x10, x11, x1,_ = self.wave_func_1(input)
        x1 = self.conv1(x1)
        self.l0 = x1
        # Layer2
        x20, x21, x2,_ = self.wave_func_2(x1)
        x2 = self.conv2(x2)
        # Layer3
        x30, x31, x3,_  = self.wave_func_3(x2)
        x3 = self.conv3(x3)
        # Layer4
        x40, x41, x4,_  = self.wave_func_4(x3)
        x4 = self.conv4(x4)
        # Layer5
        x50, x51, x5,_  = self.wave_func_5(x4)
        x5 = self.conv5(x5)

        x = self.avg_pool(x5)
        x = x.view(x.size(0), -1)
        return x

# 增强特征提取器
class Fea_Extraction_te(nn.Module):
    def __init__(self, in_channel=1,wavelet=""):
        super(Fea_Extraction_te, self).__init__()
        num_c = 16
        self.wavelet=wavelet
        self.wave_func_1 = DWConv(num_channels=in_channel,wavelet=self.wavelet)
        self.wave_func_2 = DWConv(num_channels=num_c,wavelet=self.wavelet)
        self.wave_func_3 = DWConv(num_channels=num_c * 2,wavelet=self.wavelet)
        self.wave_func_4 = DWConv(num_channels=num_c * 4,wavelet=self.wavelet)
        self.wave_func_5 = DWConv(num_channels=num_c * 8,wavelet=self.wavelet)

        self.conv1 = SConv_1D(2, num_c, 3, )
        self.conv2 = SConv_1D(num_c * 2, num_c * 2, 3, )
        self.conv3 = SConv_1D(num_c * 4, num_c * 4, 3, )
        self.conv4 = SConv_1D(num_c * 8, num_c * 8, 3, )
        self.conv5 = SConv_1D(num_c * 16, num_c * 16, 3, )

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.zdim1= 16
        self.dp= LP(self.zdim1, 16)

# 前向传播
    def forward(self, input, perturb=False):
        # Layer1
        x10, x11, x1,_ = self.wave_func_1(input)
        x1 = self.conv1(x1)
        if perturb:
            z11 = torch.randn(len(x1), self.zdim1, device=x1.device)
            z22 = torch.randn(len(x1), self.zdim1, device=x1.device)
            x1, game, beat = self.dp(x1, z11, z22)
        self.l0 = x1

        # Layer2
        x20, x21, x2,_ = self.wave_func_2(x1)
        x2 = self.conv2(x2)
        # Layer3
        x30, x31, x3,_ = self.wave_func_3(x2)
        x3 = self.conv3(x3)
        # Layer4
        x40, x41, x4,_ = self.wave_func_4(x3)
        x4 = self.conv4(x4)
        # Layer5
        x50, x51, x5,_ = self.wave_func_5(x4)
        x5 = self.conv5(x5)

        x = self.avg_pool(x5)
        x = x.view(x.size(0), -1)

        return x

class DWCN(nn.Module):
    def __init__(
        self,
        in_channel=1,
        num_classes=8,
        lr=0.01,
        sigma=0.1,
        noise_std=0.05,
        step_size=30,
        use_contrastive_loss=True,
        alpha_contrastive=0.005,
    ):
        super(DWCN, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 初始化参数
        self.lr = lr
        self.num_classes = num_classes
        self.wavelet = "db4"
        
        # 频域增强参数 (新增)
        self.sigma = sigma          # 幅度缩放强度
        self.noise_std = noise_std  # 噪声注入强度 (0 表示不注入噪声)
        self.use_contrastive_loss = use_contrastive_loss  # 是否使用对比损失
        self.alpha_contrastive = alpha_contrastive
        
        # 初始化特征提取器 (G) 和分类器 (C)
        self.G_te = Fea_Extraction_te(in_channel=in_channel, wavelet=self.wavelet).to(self.device)
        self.C_te = Classifier(256, num_classes).to(self.device)
        self.G_st = Fea_Extraction(in_channel=in_channel, wavelet=self.wavelet).to(self.device)
        self.C_st = Classifier(256, num_classes).to(self.device)
        
        # 定义损失函数
        self.criterion = nn.CrossEntropyLoss().to(self.device)  # 监督损失

        # 初始化优化器
        self.optimizer_te = optim.Adam(
            [{'params': self.G_te.parameters(), 'lr': lr},
             {'params': self.C_te.parameters(), 'lr': lr}],
        )
        self.optimizer_st = optim.Adam(
            [{'params': self.G_st.parameters(), 'lr': lr},
             {'params': self.C_st.parameters(), 'lr': lr}],
        )
        self.optimizer_LD = optim.Adam(
            [{'params': self.G_te.dp.parameters(), 'lr': lr}],
        )
        
        # 学习率调度器 (新增)
        self.scheduler_te = optim.lr_scheduler.StepLR(self.optimizer_te, step_size=step_size, gamma=0.1)
        self.scheduler_st = optim.lr_scheduler.StepLR(self.optimizer_st, step_size=step_size, gamma=0.1)
        self.scheduler_LD = optim.lr_scheduler.StepLR(self.optimizer_LD, step_size=step_size, gamma=0.1)

    def freq_aug(self, x):
        """
        Frequency-domain augmentation based on smooth band-wise perturbation.
        It changes the spectral envelope while keeping the phase and overall
        signal energy stable, which is closer to realistic condition shifts.
        """
        fft_x = torch.fft.rfft(x, dim=-1)
        amp = torch.abs(fft_x)
        pha = torch.angle(fft_x)

        batch_size, channels, freq_bins = amp.shape
        knot_count = max(4, min(16, freq_bins // 8))

        smooth_noise = torch.randn(batch_size * channels, 1, knot_count, device=x.device)
        smooth_noise = F.interpolate(smooth_noise, size=freq_bins, mode='linear', align_corners=True)
        smooth_noise = smooth_noise.view(batch_size, channels, freq_bins).tanh()

        # A smooth spectral tilt complements the local band perturbation.
        freq_axis = torch.linspace(-1.0, 1.0, freq_bins, device=x.device).view(1, 1, -1)
        tilt = torch.randn(batch_size, channels, 1, device=x.device) * (0.5 * self.sigma)

        band_mask = 1.0 + self.sigma * smooth_noise + tilt * freq_axis
        band_mask = band_mask.clamp_min(0.2)
        amp = amp * band_mask

        if self.noise_std > 0:
            noise = torch.randn_like(amp) * self.noise_std * amp
            amp = amp + noise

        aug_fft = amp * torch.exp(1j * pha)
        x_aug = torch.fft.irfft(aug_fft, n=x.shape[-1], dim=-1)

        # Keep the sample RMS close to the original so the augmentation focuses
        # on domain-style changes rather than trivial energy scaling.
        rms_x = x.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
        rms_aug = x_aug.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
        x_aug = x_aug * (rms_x / rms_aug)

        return x_aug
    
    def forward(self, SR_dataloader):
        self.G_te.train()
        self.C_te.train()
        self.G_st.train()
        self.C_st.train()
        
        # 初始化损失累加器
        epoch_loss_c = 0.0
        epoch_1 = 0.0
        epoch_2 = 0.0
        
        # 批次训练循环
        for batch_x_0, batch_y_0, batch_domain_0 in SR_dataloader:
            self.optimizer_te.zero_grad()
            self.optimizer_st.zero_grad()
            
            # 准备数据
            input_x = batch_x_0.to(self.device)
            target_y = batch_y_0.to(self.device)

            # -----------------------------------------------------------
            # [Step 1] 对辅助网络 (Teacher/Auxiliary)
            # 保持原论文逻辑：使用带扰动模块(Perturb=True)的原始数据
            # -----------------------------------------------------------
            features_te = self.G_te(input_x, perturb=True)
            scores_te = self.C_te(features_te)
            loss_c_te = self.criterion(scores_te, target_y)
            
            # -----------------------------------------------------------
            # [Step 2] 对主网络 (Student/Main) --- 修改点在这里！
            # 引入频域增强：主网络看到的是"合成的未知工况"数据
            # -----------------------------------------------------------
            # ==========================================
            # [NEW_MODULE_START]: Instance Contrastive Loss Integration
            # ==========================================
            if getattr(self, 'use_contrastive_loss', False):
                features_original = self.G_st(input_x)
                input_aug = self.freq_aug(input_x)
                features_aug = self.G_st(input_aug)
                scores_st = self.C_st(features_aug)
                loss_c_st = self.criterion(scores_st, target_y)

                loss_c = (loss_c_te + loss_c_st) * 0.5
                loss_cons = self.compute_kl_loss(scores_st, scores_te, T=3)

                f_st = F.normalize(features_aug, dim=1)
                logits_sim_st = self._calculate_isd_sim(f_st)

                f_te = F.normalize(features_te, dim=1)
                logits_sim_te = self._calculate_isd_sim(f_te)

                inputs = F.log_softmax(logits_sim_st, dim=1)
                targets = F.softmax(logits_sim_te, dim=1)
                loss_distill_1 = F.kl_div(inputs, targets, reduction='batchmean')

                inputs = F.log_softmax(logits_sim_te, dim=1)
                targets = F.softmax(logits_sim_st, dim=1)
                loss_distill_2 = F.kl_div(inputs, targets, reduction='batchmean')
                loss_isl = (loss_distill_1 + loss_distill_2) * 0.5

                loss_contrastive = instance_contrastive_loss(features_original, features_aug)
                loss_all = loss_c + loss_cons + loss_isl * 0.5 + self.alpha_contrastive * loss_contrastive
            else:
                # ==========================================
                # [ORIGINAL_CODE]: ?????????
                # ==========================================
                # ??????
                input_aug = self.freq_aug(input_x)
                
                # ??????? (??????)
                features_st = self.G_st(input_aug)
                scores_st = self.C_st(features_st)
                
                # ?????? (??????????? y ???????)
                loss_c_st = self.criterion(scores_st, target_y)

                # -----------------------------------------------------------
                # [Step 3] ??????
                # -----------------------------------------------------------
                loss_c = (loss_c_te + loss_c_st) * 0.5
                
                # ????? (Consistency Loss)
                # ???????[????] ? [????????] ????
                loss_cons = self.compute_kl_loss(scores_st, scores_te, T=3)

                # ??????? (ISL)
                f_st = F.normalize(features_st, dim=1)
                logits_sim_st = self._calculate_isd_sim(f_st)
                
                f_te = F.normalize(features_te, dim=1)
                logits_sim_te = self._calculate_isd_sim(f_te)
                
                inputs = F.log_softmax(logits_sim_st, dim=1)
                targets = F.softmax(logits_sim_te, dim=1)
                loss_distill_1 = F.kl_div(inputs, targets, reduction='batchmean')
                
                inputs = F.log_softmax(logits_sim_te, dim=1)
                targets = F.softmax(logits_sim_st, dim=1)
                loss_distill_2 = F.kl_div(inputs, targets, reduction='batchmean')
                loss_isl = (loss_distill_1 + loss_distill_2) * 0.5

                # ???????
                loss_all = loss_c + loss_cons + loss_isl * 0.5
            # ==========================================
            # [NEW_MODULE_END]
            # ==========================================
            loss_all.backward()
            self.optimizer_st.step()
            self.optimizer_te.step()

            # -----------------------------------------------------------
            # [Step 4] 更新 CCP 模块 (最大化差异)
            # -----------------------------------------------------------
            # 注意：为了计算 Gram 矩阵距离，这里最好再次前向传播一次原始数据
            # 这样保证 Gram 矩阵计算的是"原始分布"下的差异，避免增强数据的干扰
            _ = self.G_te(input_x, perturb=True)
            _ = self.G_st(input_x) # 这里用回 input_x
            
            l0_te = self.G_te.l0
            l0_st = self.G_st.l0
            simi_tea0 = -self.F_distance(self.gram(l0_st), self.gram(l0_te))
            loss_ccp = 0.05 * simi_tea0
            
            self.optimizer_LD.zero_grad()
            loss_ccp.backward()
            self.optimizer_LD.step()

            # 累加损失
            epoch_loss_c += (loss_c.item()) / 64
            if getattr(self, 'use_contrastive_loss', False):
                epoch_1 += (loss_c.item() + loss_cons.item() + loss_isl.item() + self.alpha_contrastive * loss_contrastive.item()) / 64
            else:
                epoch_1 += (loss_c.item() + loss_cons.item() + loss_isl.item()) / 64
            epoch_2 += (loss_ccp.item()) / 64

        # 更新学习率调度器 (每个 epoch 结束后调用)
        self.scheduler_te.step()
        self.scheduler_st.step()
        self.scheduler_LD.step()

        # 准备损失摘要
        loss_summary_log = {"loss_c": epoch_loss_c, "loss_1": epoch_1, "loss_2": epoch_2}

        return loss_summary_log

    def _calculate_isd_sim(self, features):
        sim_q = torch.mm(features, features.T)
        logits_mask = torch.scatter(
            torch.ones_like(sim_q),
            1,
            torch.arange(sim_q.size(0)).view(-1, 1).to(self.device),
            0
        )
        row_size = sim_q.size(0)
        sim_q = sim_q[logits_mask.bool()].view(row_size, -1)
        return sim_q / 0.05

    def gram(self, y):
        (b, c, h) = y.size()
        features = y.view(b, c, h)
        features_t = features.transpose(1, 2)
        gram_y = features.bmm(features_t) / (c * h)
        return gram_y

    def F_distance(self, x, y):
        return (torch.norm(x - y)).mean()

    def compute_kl_loss(self, p, q, pad_mask=None, T=3):
        p_T = p / T
        q_T = q / T
        p_loss = F.kl_div(F.log_softmax(p_T, dim=-1), F.softmax(q_T, dim=-1), reduction='none')
        q_loss = F.kl_div(F.log_softmax(q_T, dim=-1), F.softmax(p_T, dim=-1), reduction='none')
        if pad_mask is not None:
            p_loss.masked_fill_(pad_mask, 0.)
            q_loss.masked_fill_(pad_mask, 0.)
        p_loss = p_loss.mean()
        q_loss = q_loss.mean()
        loss = (p_loss + q_loss) / 2
        return loss

    def model_inference(self, input):
        with torch.no_grad():
            features = self.G_st(input)
            prediction = self.C_st(features)
        return prediction
