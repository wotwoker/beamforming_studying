# -*- coding: utf-8 -*-
"""
CBF for CW 与 HFM（宽带 DAS）波束形成：Python 版
依赖：numpy, scipy, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
from scipy.signal.windows import kaiser 
from numpy import pi

plt.rcParams['font.sans-serif'] = ['SimHei']  # or 'Microsoft YaHei'
plt.rcParams['axes.unicode_minus'] = False    # keep minus sign readable

# ============== 一些小工具 ==============
def db_normalize(p):
    p = np.asarray(p, dtype=float)
    p = p / (np.max(p) + np.finfo(float).eps)
    return 10 * np.log10(p + np.finfo(float).eps)

def hpbw_from_db_curve(angles_deg, p_db):
    """根据 dB 曲线求 -3 dB 半功率宽度（简单的左右扫描法）。"""
    peak_idx = int(np.argmax(p_db))
    # 左侧
    left_idx = peak_idx
    while left_idx > 0 and p_db[left_idx] > -3:
        left_idx -= 1
    # 右侧
    right_idx = peak_idx
    while right_idx < len(p_db) - 1 and p_db[right_idx] > -3:
        right_idx += 1
    return (angles_deg[peak_idx], angles_deg[left_idx], angles_deg[right_idx],
            angles_deg[right_idx] - angles_deg[left_idx])

def interp_complex(x, xp, fp):
    """对复信号做一维线性插值：分别插值实部与虚部，越界补零。"""
    real = np.interp(x, xp, fp.real, left=0.0, right=0.0)
    imag = np.interp(x, xp, fp.imag, left=0.0, right=0.0)
    return real + 1j * imag

def set_pretty(ax, title=None, xlabel=None, ylabel=None, legend=True, grid=True):
    ax.set_title(title, fontsize=12, pad=8)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    if grid:
        ax.grid(True, linestyle=':', linewidth=0.8, alpha=0.8)
    if legend:
        ax.legend(frameon=False, fontsize=9, loc='best')
    for spine in ax.spines.values():
        spine.set_alpha(0.4)

# ============== 基本参数 ==============
np.random.seed(42)  # 可复现实验
N = 64                  # 阵元数量
d = 0.5                 # 阵元间距 (m)
c = 1500                # 声速 (m/s)
f = 1500                # CW 频率 (Hz)
fs = 5000               # 采样频率 (Hz)
T = 0.1                 # 信号时长 (s)
t = np.arange(0, T, 1/fs)     # 时间向量
L = t.size                    # 采样点数（快拍数）

signal_doa_deg = -30
signal_doa_rad = np.deg2rad(signal_doa_deg)

# 派生量
_lambda = c / f
k0 = 2 * pi / _lambda
element_pos = np.arange(N)[:, None] * d             # (N,1)
scan_angles_deg = np.arange(-90, 91, 1)
scan_angles_rad = np.deg2rad(scan_angles_deg)
K = scan_angles_deg.size

# ============== 模拟 CW 接收信号 ==============
tau = element_pos * np.sin(signal_doa_rad) / c                # (N,1)
time_matrix = t[None, :] - tau                                # (N,L)
received_signals = np.exp(1j * 2 * pi * f * time_matrix)      # (N,L)
received_signals += 0.1 * (np.random.randn(N, L) + 1j * np.random.randn(N, L))

# ============== CBF（窄带相移波束形成）===============
# 阵列流形 S：N x K
S = np.exp(-1j * k0 * (element_pos @ np.sin(scan_angles_rad)[None, :]))  # (N,K)
Rxx = (received_signals @ received_signals.conj().T) / L                 # (N,N)
# P(theta) = a^H Rxx a
P = np.real(np.sum(np.conj(S) * (Rxx @ S), axis=0))  # (K,)
P_db = db_normalize(P)

# -3 dB 半功率宽度
peak_deg, left_deg, right_deg, hpbw = hpbw_from_db_curve(scan_angles_deg, P_db)
print(f"主瓣峰值位于: {peak_deg:.0f}°")
print(f"-3 dB 点位于: {left_deg:.0f}° 和 {right_deg:.0f}°")
print(f"波束宽度 (HPBW) 约为: {hpbw:.0f}°")

# ============== 绘图：CBF 方向图 ==============
plt.figure(figsize=(8, 4.8))
ax1 = plt.gca()
ax1.plot(scan_angles_deg, P_db, label='CBF')
ax1.plot([left_deg, right_deg], [-3, -3], 'k', linewidth=2,
         label=f'-3 dB 宽度: {hpbw:.0f}°')
ax1.axvline(signal_doa_deg, linestyle='--', linewidth=1.2, color='red',
            label=f'真实 DOA ({signal_doa_deg}°)')
ax1.set_xlim(-90, 90)
ax1.set_ylim(-50, 5)
set_pretty(ax1, '常规波束形成图 (CW)', '角度 (°)', '归一化功率 (dB)')

# ============== 波束域时间序列（相移法）===============
B_phase = (S.conj().T @ received_signals)  # (K,L)
plt.figure(figsize=(8, 4.8))
ax2 = plt.gca()
im = ax2.imshow(np.abs(B_phase), aspect='auto', origin='lower',
                extent=[t[0], t[-1], scan_angles_deg[0], scan_angles_deg[-1]])
plt.colorbar(im, ax=ax2)
set_pretty(ax2, '波束空间的时间序列 (相移法，幅度)', '时间 (s)', '扫描角度 (°)', legend=False)

# ============== 阵元域图像（CW 实部）===============
plt.figure(figsize=(8, 4.8))
ax3 = plt.gca()
im = ax3.imshow(np.real(received_signals), aspect='auto', origin='lower',
                extent=[t[0], t[-1], 1, N])
plt.colorbar(im, ax=ax3)
set_pretty(ax3, 'CW 信号阵元域 (实部)', '时间 (s)', '阵元索引', legend=False)

# ============== HFM：宽带延时求和波束形成 ==============
f_start = 1500
f_end = 500
f_center = 0.5 * (f_start + f_end)  # 仅参考

# HFM 超曲调频（Hyperbolic FM）基带：与 MATLAB 公式一致
k_hfm = (f_start / T) * (f_start / f_end - 1.0)
phi = (2 * pi * f_start**2 / k_hfm) * np.log(1.0 + (k_hfm / f_start) * t)
hfm_signal_base = np.exp(1j * phi)  # 单位幅度

# 阵列接收：按几何延时采样
tau = (element_pos * np.sin(signal_doa_rad) / c).ravel()  # (N,)
received_signals_hfm = np.zeros((N, L), dtype=complex)
for n in range(N):
    received_signals_hfm[n, :] = interp_complex(t - tau[n], t, hfm_signal_base)

received_signals_hfm += 0.1 * (np.random.randn(N, L) + 1j * np.random.randn(N, L))

# 宽带延时求和（DAS）：按角度扫描
B_das = np.zeros((K, L), dtype=complex)
for kk, theta in enumerate(scan_angles_rad):
    tau_scan = (element_pos * np.sin(theta) / c).ravel()  # (N,)
    y_sum = np.zeros(L, dtype=complex)
    for n in range(N):
        y_n = interp_complex(t + tau_scan[n], t, received_signals_hfm[n, :])
        y_sum += y_n
    B_das[kk, :] = y_sum / N

P_hfm = np.mean(np.abs(B_das) ** 2, axis=1)
P_db_hfm = db_normalize(P_hfm)

# HFM -3 dB 宽度
peak_deg_h, left_deg_h, right_deg_h, hpbw_h = hpbw_from_db_curve(scan_angles_deg, P_db_hfm)
print(f"[HFM] 主瓣峰值位于: {peak_deg_h:.0f}°")
print(f"[HFM] -3 dB 点位于: {left_deg_h:.0f}° 和 {right_deg_h:.0f}°")
print(f"[HFM] 波束宽度 (HPBW) 约为: {hpbw_h:.0f}°")

# HFM 方向图
plt.figure(figsize=(8, 4.8))
ax4 = plt.gca()
ax4.plot(scan_angles_deg, P_db_hfm, label='DAS (HFM)')
ax4.plot([left_deg_h, right_deg_h], [-3, -3], 'k', linewidth=2,
         label=f'-3 dB 宽度: {hpbw_h:.0f}°')
ax4.axvline(signal_doa_deg, linestyle='--', linewidth=1.2, color='red',
            label=f'真实 DOA ({signal_doa_deg}°)')
ax4.set_xlim(-90, 90)
ax4.set_ylim(-50, 5)
set_pretty(ax4, 'HFM 宽带延时求和方向图', '角度 (°)', '归一化功率 (dB)')

# HFM 波束域时间序列
plt.figure(figsize=(8, 4.8))
ax5 = plt.gca()
im = ax5.imshow(np.abs(B_das), aspect='auto', origin='lower',
                extent=[t[0], t[-1], scan_angles_deg[0], scan_angles_deg[-1]])
plt.colorbar(im, ax=ax5)
set_pretty(ax5, 'HFM 波束空间的时间序列（延时求和，幅度）', '时间 (s)', '扫描角度 (°)', legend=False)

# HFM 阵元域（实部）
plt.figure(figsize=(8, 4.8))
ax6 = plt.gca()
im = ax6.imshow(np.real(received_signals_hfm), aspect='auto', origin='lower',
                extent=[t[0], t[-1], 1, N])
plt.colorbar(im, ax=ax6)
set_pretty(ax6, 'HFM 信号阵元域 (实部)', '时间 (s)', '阵元索引', legend=False)

# ============== HFM 基带信号可视化 ==============
plt.figure(figsize=(8, 5.2))
ax7a = plt.subplot(2, 1, 1)
ax7a.plot(t, np.real(hfm_signal_base), label='HFM 实部')
set_pretty(ax7a, 'HFM 时域波形（实部）', '时间 (s)', '幅度')

# 瞬时频率：f(t) = (1/2π) dφ/dt
inst_freq = np.diff(np.unwrap(phi)) / (2 * pi * np.diff(t))  # 长度 L-1
ax7b = plt.subplot(2, 1, 2)
ax7b.plot(t[:-1], inst_freq, label='HFM 瞬时频率')
ax7b.set_ylim([f_end - 100, f_start + 100])
set_pretty(ax7b, 'HFM 瞬时频率', '时间 (s)', '频率 (Hz)')

# 谱图（时频分析）
plt.figure(figsize=(8, 4.8))
# 使用与 MATLAB 近似的参数：kaiser(128, 5), noverlap=120, nfft=256
f_spec, t_spec, Sxx = spectrogram(hfm_signal_base,
                                  window=kaiser(128, 5),
                                  noverlap=120,
                                  nperseg=128,
                                  nfft=256,
                                  fs=fs,
                                  mode='magnitude')
Sxx_db = db_normalize(Sxx)  # 归一化到 dB 便于显示
ax8 = plt.gca()
mesh = ax8.pcolormesh(t_spec, f_spec, Sxx_db, shading='gouraud')
plt.colorbar(mesh, ax=ax8, label='归一化幅度 (dB)')
set_pretty(ax8, 'HFM 信号的谱图（时频分析）', '时间 (s)', '频率 (Hz)', legend=False)

plt.tight_layout()
plt.show()
