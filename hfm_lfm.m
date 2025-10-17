clear; clc; close all;
% 参数设置
fs = 1000e3;        % 采样率 1 MHz
T = 0.001;          % 脉冲宽度 1 ms
fc = 100e3;         % 中心频率 100 kHz
B = 50e3;           % 带宽 50 kHz
t = -T/2 : 1/fs : T/2 - 1/fs; % 时间轴 (-T/2 到 T/2)

% 计算调频率 k
% 根据瞬时频率公式：f_start = fc / (1 - k'*(-T/2))， f_end = fc / (1 - k'*(T/2))
% 可以推导出 k'。这里用一种近似方法：k = B/T
k = B / T; % 这是一个近似值，用于定义频率变化率

% 生成HFM信号（使用积分关系，瞬时相位是瞬时频率的积分）
% 瞬时频率 fi = fc + k * t (这是一种简化，更精确的需用双曲公式)
% 对于严格的HFM，相位应为对数形式。这里用LFM近似，如果要精确生成HFM，相位需为 log(1 - k*t) 形式。

% 方法1：使用LFM（作为对比）
s_lfm = cos(2*pi*fc*t + pi*k*t.^2);

% 方法2：生成HFM信号（精确）
% 定义调频率参数 k_prime
k_prime = (B / fc) / T; % 一个简化的定义方式
% 瞬时相位积分：∫ (fc / (1 - k_prime * τ)) dτ = -(fc/k_prime) * ln|1 - k_prime * τ|
phi_hfm = -(2*pi*fc / k_prime) * log(abs(1 - k_prime * t));
s_hfm = cos(phi_hfm);

% 绘制时域波形
figure;
subplot(2,1,1);
plot(t*1000, real(s_lfm));
title('线性调频信号时域波形');
xlabel('时间 (ms)');
ylabel('幅度');
grid on;
xlim([-0.5, 0.5]);

subplot(2,1,2);
plot(t*1000, real(s_hfm));
title('双曲调频信号时域波形');
xlabel('时间 (ms)');
ylabel('幅度');
grid on;
xlim([-0.5, 0.5]);

% 绘制频谱
NFFT = 2^nextpow2(length(s_hfm));
f = fs*(-NFFT/2:NFFT/2-1)/NFFT;

S_lfm = fftshift(fft(s_lfm, NFFT));
S_hfm = fftshift(fft(s_hfm, NFFT));

figure;
subplot(2,1,1);
plot(f/1000, 20*log10(abs(S_lfm/max(abs(S_lfm)))));
title('线性调频信号频谱');
xlabel('频率 (kHz)');
ylabel('幅度 (dB)');
grid on;
xlim([fc/1000-B/2000, fc/1000+B/2000]*2);

subplot(2,1,2);
plot(f/1000, 20*log10(abs(S_hfm/max(abs(S_hfm)))));
title('双曲调频信号频谱');
xlabel('频率 (kHz)');
ylabel('幅度 (dB)');
grid on;
xlim([fc/1000-B/2000, fc/1000+B/2000]*2);