%% 低频 CW 信号微观仿真：2000Hz, 50周期脉宽 (参数适配版)
% Figure 1: 左侧放波形(2x2)，右侧放频谱(2x2)
% Figure 2: 放时频图(2x2)
clear; clc; close all;

%% 1. 全局参数设置 (适配低频)
fs = 50e3;              % 采样率：50kHz (对于2kHz信号足够)
f0 = 2000;              % 信号频率：2000Hz (低频)
T_total = 0.2;          % 总时长：200ms
t = 0:1/fs:T_total-1/fs;% 时间轴

% 脉冲定义：50个周期 
period = 1/f0;          % 周期 = 0.5ms
pulse_duration = 50 * period; % 脉宽 = 25ms
t_start1 = 0.02;        % 第一个脉冲在 20ms 处开始

% 定义 4 种时延间隔 (Delta t)
% 由于脉宽变大(25ms)，时延也要相应变大才能看出区别
% 10ms(重叠), 20ms(严重重叠), 35ms(分离), 60ms(远距离)
gaps = [0.010, 0.020, 0.035, 0.060]; 
N_scenarios = length(gaps);

%% 2. 信号生成
% 定义单脉冲的长度点数
N_pulse = round(pulse_duration * fs);
% 生成窗函数 (Tukey窗)
soft_envelope = tukeywin(N_pulse, 0.2)'; 

signals = zeros(N_scenarios, length(t));

for i = 1:N_scenarios
    gap = gaps(i);
    % 生成全零背景
    sig_temp = zeros(size(t));
    
    % --- 填入第一个脉冲 ---
    idx1_start = round(t_start1 * fs) + 1;
    idx1_end = idx1_start + N_pulse - 1;
    sig_temp(idx1_start:idx1_end) = soft_envelope .* cos(2*pi*f0*t(idx1_start:idx1_end));
    
    % --- 填入第二个脉冲 (t_start2 = t_start1 + gap) ---
    t_start2 = t_start1 + gap;
    idx2_start = round(t_start2 * fs) + 1;
    idx2_end = idx2_start + N_pulse - 1;
    
    % 边界检查，防止数组越界
    if idx2_end <= length(t)
        sig_temp(idx2_start:idx2_end) = sig_temp(idx2_start:idx2_end) + ...
            soft_envelope .* cos(2*pi*f0*t(idx2_start:idx2_end));
    end
    
    % 添加微量噪声
    signals(i, :) = sig_temp + 0.01 * randn(size(t));
end

%% 3. STFT 参数设计 (适配低频)
% 关键修改：增加窗长以获得足够的频率分辨率
win_duration = 0.005;                  % 窗长 5ms (包含10个信号周期)
window_len = round(win_duration * fs); 
noverlap = round(window_len * 0.8);    % 80% 重叠
nfft = 4096; 

%% 4. 绘图展示 - Figure 1: 时域与频域对比
wave_idx_map = [1, 2, 5, 6]; 
fft_idx_map  = [3, 4, 7, 8];

figure('Name', 'Figure 1: Waveforms (Left) & Spectra (Right)', 'Color','w', 'Position', [50, 50, 1400, 700]);

for i = 1:N_scenarios
    CurrentSignal = signals(i,:);
    gap_ms = gaps(i) * 1000;
    
    % --- 左半区：时域波形 ---
    subplot(2, 4, wave_idx_map(i)); 
    plot(t*1000, CurrentSignal, 'b'); 
    grid on; 
    % 调整显示范围：覆盖两个脉冲
    xlim([10 100]); 
    ylim([-2.2 2.2]); % 因为有叠加，幅度可能达到2
    title(['CW信号',num2str(i), ' 时域 (\Delta t=', num2str(gap_ms), 'ms)']);
    ylabel('幅度');
    if i > 2; xlabel('时间 (ms)'); end
    
    % --- 右半区：FFT 频谱 ---
    subplot(2, 4, fft_idx_map(i));
    L = length(CurrentSignal);
    Y = fft(CurrentSignal);
    P2 = abs(Y);
    P1 = P2(1:L/2+1);
    P1_dB = 20*log10(P1(1:floor(length(P1)/2)) + eps); 
    f_axis = fs*(0:(length(P1_dB)-1))/L;
    
    plot(f_axis, P1_dB, 'r'); % x轴单位 Hz
    grid on; 
    % 聚焦低频段 0 - 5000 Hz
    xlim([0 5000]); 
    ylim([-20 80]);
    title(['CW信号',num2str(i), ' FFT']);
    ylabel('dB');
    if i > 2; xlabel('频率 (Hz)'); end
    
    % 标注波纹频率 (仅当有波纹时)
    % 对于CW信号，波纹不如宽带LFM明显，但在重叠时幅度会变化
    ripple_hz = 1/gaps(i);
    text(4000, 60, ['~' num2str(ripple_hz) 'Hz'], 'BackgroundColor','w');
end

%% 5. 绘图展示 - Figure 2: STFT 时频图
figure('Name', 'Figure 2: STFT Spectrograms', 'Color','w', 'Position', [100, 100, 800, 700]);

for i = 1:N_scenarios
    CurrentSignal = signals(i,:);
    gap_ms = gaps(i) * 1000;
    
    subplot(2, 2, i);
    
    % 计算 STFT
    [s, f_stft, t_stft] = spectrogram(CurrentSignal, hann(window_len), noverlap, nfft, fs);
    S_dB = 20*log10(abs(s) + eps);
    
    % 绘制热力图
    imagesc(t_stft*1000, f_stft, S_dB); 
    axis xy; 
    colormap jet;
    
    % 统一视觉范围
    ylim([0 4000]);   % 频率 0-4000 Hz (聚焦基带)
    xlim([10 100]);   % 时间 10-100 ms
    clim([-60 10]);   % 动态范围
    colorbar;
    
    title(['CW信号',num2str(i), ' STFT (\Delta t=', num2str(gap_ms), 'ms)']);
    ylabel('频率 (Hz)');
    xlabel('时间 (ms)');
end