# -*- coding: utf-8 -*-
"""
MVDR (CW) 与 宽带 MVDR（HFM 频域非相干平均）——Python 版
依赖：numpy, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy import pi

# ============== 小工具 ==============
def db_normalize(p):
    p = np.asarray(p, dtype=float)
    p = p / (np.max(p) + np.finfo(float).eps)
    return 10 * np.log10(p + np.finfo(float).eps)

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

def interp_complex(x, xp, fp):
    """复值一维线性插值（越界补0），等价 MATLAB interp1(...,'linear',0)"""
    real = np.interp(x, xp, fp.real, left=0.0, right=0.0)
    imag = np.interp(x, xp, fp.imag, left=0.0, right=0.0)
    return real + 1j * imag

def steering_matrix_deg(n_elem, d, k0, angles_deg):
    """返回阵列流形 S ∈ C^{N×K}，a(θ)=exp(-j*k0*d*n*sindθ)"""
    n = np.arange(n_elem)[:, None]                # N×1
    s = np.sin(np.deg2rad(angles_deg))[None, :]   # 1×K
    return np.exp(-1j * k0 * d * n * s)           # N×K


# ============== 1. 基本参数 ==============
np.random.seed(42)

N = 64                 # 阵元数
d = 0.5                # 阵元间距 (m)
c = 1500               # 声速 (m/s)
f = 1500               # CW 频率 (Hz)
fs = 5000              # 采样频率 (Hz)
T  = 0.1               # 时长 (s)
t  = np.arange(0, T, 1/fs)      # L
L  = t.size
signal_doa_deg = 35    # 真实 DOA
SNR_dB = 0             # 单阵元 SNR (dB)

lam = c / f
k0  = 2 * pi / lam

# ============== 2. 模拟 CW 接收信号 ==============
# 导向矢量
a_true = np.exp(-1j * k0 * d * np.arange(N) * np.sin(np.deg2rad(signal_doa_deg)))  # N
# 基带 CW
s = np.exp(1j * 2 * pi * f * t)  # L
# 信号幅度（噪声方差=1）
signal_power = 10 ** (SNR_dB / 10.0)
As = np.sqrt(signal_power)

# X_clean = a*s^T
X_clean = As * (a_true[:, None] @ s[None, :])     # N×L
# 复高斯白噪声（功率=1）
noise = (np.random.randn(N, L) + 1j*np.random.randn(N, L))/np.sqrt(2)
received_signals = X_clean + noise                # N×L

# ============== 3. 协方差估计 + 对角加载 ==============
Rxx = (received_signals @ received_signals.conj().T) / L   # N×N
epsilon = 1e-3
Rxx = Rxx + epsilon * (np.trace(Rxx).real / N) * np.eye(N)

# ============== 4. 窄带 MVDR 与 CBF 扫描 ==============
scan_angles_deg = np.arange(-90, 91, 1)  # K=181
K = scan_angles_deg.size
S = steering_matrix_deg(N, d, k0, scan_angles_deg)         # N×K

# MVDR：P(θ)=1/(a^H R^{-1} a)
R_inv_S = np.linalg.solve(Rxx, S)                          # N×K 代替 inv(Rxx)@S
den = np.sum(np.conj(S) * R_inv_S, axis=0)                 # K
P_mvdr = np.real(1.0 / (den + np.finfo(float).eps))
P_mvdr_db = db_normalize(P_mvdr)

# 生成每个角度的最优权并作用到时序：B_mvdr_cw ∈ C^{K×L}
W = R_inv_S / (den[None, :] + np.finfo(float).eps)         # N×K
B_mvdr_cw = (W.conj().T @ received_signals)                # K×L

# CBF：P_bartlett(θ)=a^H R a
P_cbf = np.real(np.sum(np.conj(S) * (Rxx @ S), axis=0))    # K
P_cbf_db = db_normalize(P_cbf)

# 估计 DOA
doa_estimate_mvdr = scan_angles_deg[int(np.argmax(P_mvdr_db))]
print(f"MVDR 估计角度: {doa_estimate_mvdr:.0f}°")

# ============== 5. 可视化（窄带） ==============
plt.figure(figsize=(8, 4.8))
ax1 = plt.gca()
ax1.plot(scan_angles_deg, P_mvdr_db, linewidth=1.4, label='MVDR')
ax1.plot(scan_angles_deg, P_cbf_db, '-.', linewidth=1.2, label='CBF (Bartlett)')
ax1.axvline(signal_doa_deg, color='r', linestyle='--', linewidth=1.2, label='真实 DOA')
ax1.set_xlim(-90, 90)
ax1.set_ylim(-20, 0)
set_pretty(ax1, 'MVDR 与 CBF 空间谱对比', '角度 (°)', '归一化功率 (dB)')

plt.figure(figsize=(8, 4.8))
ax2 = plt.gca()
im = ax2.imshow(np.abs(B_mvdr_cw), aspect='auto', origin='lower',
                extent=[t[0], t[-1], scan_angles_deg[0], scan_angles_deg[-1]])
plt.colorbar(im, ax=ax2)
set_pretty(ax2, '窄带 (CW) MVDR 波束空间时间序列', '时间 (s)', '扫描角度 (°)', legend=False)

plt.figure(figsize=(8, 4.8))
ax3 = plt.gca()
im = ax3.imshow(np.real(received_signals), aspect='auto', origin='lower',
                extent=[t[0], t[-1], 1, N])
plt.colorbar(im, ax=ax3)
set_pretty(ax3, 'CW 信号阵元域 (实部)', '时间 (s)', '阵元索引', legend=False)

print('完成!')

# ============== 6. HFM 信号生成（宽带场景） ==============
f_start = 1500
f_end   = 500

k_hfm = (f_start / T) * (f_start / f_end - 1.0)
phi   = (2 * pi * f_start**2 / k_hfm) * np.log(1.0 + (k_hfm / f_start) * t)
hfm_signal_base = np.exp(1j * phi)  # L

# 几何延时
tau = d * np.arange(N) * np.sin(np.deg2rad(signal_doa_deg)) / c   # N
received_signals_hfm = np.zeros((N, L), dtype=complex)
for n in range(N):
    # 在 t - tau[n] 位置取样
    received_signals_hfm[n, :] = interp_complex(t - tau[n], t, hfm_signal_base)

# 叠加幅度与噪声
X_clean_hfm = As * received_signals_hfm
noise_hfm = (np.random.randn(N, L) + 1j*np.random.randn(N, L)) / np.sqrt(2)
received_signals_hfm = X_clean_hfm + noise_hfm

# ============== 7. 宽带 MVDR（频域非相干平均 + IFFT 合成） ==============
N_fft = L
freq_bins = np.arange(N_fft) * fs / N_fft       # 0 ... fs*(Nfft-1)/Nfft
freq_idx = np.where((freq_bins >= f_end) & (freq_bins <= f_start))[0]
print(f'将处理 {freq_idx.size} 个频箱...')

# 沿时间轴做 FFT -> X_fft ∈ C^{N×N_fft}
X_fft = np.fft.fft(received_signals_hfm, n=N_fft, axis=1)

B_mvdr_hfm_freq = np.zeros((K, N_fft), dtype=complex)

for fi in freq_idx:
    f_fi = freq_bins[fi]
    X_fi = X_fft[:, fi]                      # N

    # 频率 f_fi 的 CSDM（单快照 + 对角加载）
    Rxx_fi = np.outer(X_fi, X_fi.conj())
    Rxx_fi = Rxx_fi + epsilon * (np.trace(Rxx_fi).real / N) * np.eye(N)

    # 对应频率的阵列流形 S_fi
    k0_fi = 2 * pi * f_fi / c
    S_fi = steering_matrix_deg(N, d, k0_fi, scan_angles_deg)  # N×K

    # Y_fi(θ) = (a^H R^{-1} X) / (a^H R^{-1} a)
    Rinv_S = np.linalg.solve(Rxx_fi, S_fi)                    # N×K
    den_fi = np.sum(np.conj(S_fi) * Rinv_S, axis=0)           # K
    num_fi = (np.conj(S_fi).T @ np.linalg.solve(Rxx_fi, X_fi))  # K
    Y_fi   = num_fi / (den_fi + np.finfo(float).eps)          # K

    B_mvdr_hfm_freq[:, fi] = Y_fi

# IFFT 合成角-时域输出：K×N_fft
B_mvdr_hfm_time = np.fft.ifft(B_mvdr_hfm_freq, n=N_fft, axis=1)
P_mvdr_hfm = np.mean(np.abs(B_mvdr_hfm_time) ** 2, axis=1)   # K
P_mvdr_hfm_db = db_normalize(P_mvdr_hfm)

doa_estimate_mvdr_hfm = scan_angles_deg[int(np.argmax(P_mvdr_hfm_db))]
print(f"[HFM] MVDR 估计角度: {doa_estimate_mvdr_hfm:.0f}°")

# ============== 8. 可视化（宽带 HFM） ==============
plt.figure(figsize=(8, 4.8))
ax4 = plt.gca()
ax4.plot(scan_angles_deg, P_mvdr_hfm_db, linewidth=1.5, label='MVDR (HFM)')
ax4.axvline(signal_doa_deg, color='r', linestyle='--', linewidth=1.5, label='真实 DOA')
ax4.set_xlim(-90, 90)
ax4.set_ylim(-42, 0)
set_pretty(ax4, '宽带 (HFM) MVDR 空间谱（频域平均法）', '角度 (°)', '归一化功率 (dB)')

plt.figure(figsize=(8, 4.8))
ax5 = plt.gca()
im = ax5.imshow(np.abs(B_mvdr_hfm_time), aspect='auto', origin='lower',
                extent=[t[0], t[-1], scan_angles_deg[0], scan_angles_deg[-1]])
plt.colorbar(im, ax=ax5)
set_pretty(ax5, '宽带 (HFM) MVDR 波束空间时间序列（频域合成）', '时间 (s)', '扫描角度 (°)', legend=False)

plt.figure(figsize=(8, 4.8))
ax6 = plt.gca()
im = ax6.imshow(np.real(received_signals_hfm), aspect='auto', origin='lower',
                extent=[t[0], t[-1], 1, N])
plt.colorbar(im, ax=ax6)
set_pretty(ax6, 'HFM 信号阵元域 (实部)', '时间 (s)', '阵元索引', legend=False)

plt.tight_layout()
plt.show()
