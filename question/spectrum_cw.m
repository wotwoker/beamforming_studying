%% 高频微观仿真：30kHz, 50周期脉宽 (布局优化版)
% Figure 1: 左侧放波形(2x2)，右侧放频谱(2x2)
% Figure 2: 放时频图(2x2)
clear; clc; close all;

%% 1. 全局参数设置
fs = 500e3;            % 采样率 500kHz
f0 = 30e3;             % 信号频率 30kHz
T_total = 0.2;          % 总时长 200ms
t = 0:1/fs:T_total-1/fs;% 时间轴

% 脉冲定义：50个周期 (约 1.67 ms)
period = 1/f0;
pulse_duration = 50 * period; 
t_start1 = 0.02;        % 第一个脉冲在 20ms 处开始

% 定义 4 种微小的毫秒级间隔 (Gaps)
gaps = [0.005, 0.010, 0.020, 0.040]; 
N_scenarios = length(gaps);

%% 2. 信号生成
% signals = zeros(N_scenarios, length(t));
% mask1 = (t >= t_start1) & (t < t_start1 + pulse_duration);
% 
% for i = 1:N_scenarios
%     gap = gaps(i);
%     t_start2 = t_start1 + pulse_duration + gap;
%     mask2 = (t >= t_start2) & (t < t_start2 + pulse_duration);
% 
%     total_mask = mask1 | mask2;
%     sig_pure = cos(2*pi*f0*t) .* total_mask;
%     signals(i, :) = sig_pure + 0.01 * randn(size(t)); 
% end

% 改进代码：给脉冲加一个包络（例如 Tukey 窗或高斯窗）
% 定义单脉冲的长度点数
N_pulse = round(pulse_duration * fs);
% 生成一个两头圆滑、中间平坦的窗 (Tukey窗，r=0.2表示边缘各占10%)
soft_envelope = tukeywin(N_pulse, 0.2)'; 

signals = zeros(N_scenarios, length(t));

for i = 1:N_scenarios
    gap = gaps(i);
    % 生成全零背景
    sig_temp = zeros(size(t));
    
    % 填入第一个脉冲 (带柔和边缘)
    idx1_start = round(t_start1 * fs) + 1;
    idx1_end = idx1_start + N_pulse - 1;
    sig_temp(idx1_start:idx1_end) = soft_envelope .* cos(2*pi*f0*t(idx1_start:idx1_end));
    
    % 填入第二个脉冲 (带柔和边缘)
    t_start2 = t_start1 + gap;
    idx2_start = round(t_start2 * fs) + 1;
    idx2_end = idx2_start + N_pulse - 1;
    sig_temp(idx2_start:idx2_end) = soft_envelope .* cos(2*pi*f0*t(idx2_start:idx2_end));
    
    signals(i, :) = sig_temp + 0.01 * randn(size(t));
end

%% 3. STFT 参数设计 (1.0ms 短窗)
win_duration = 0.0010;                 
window_len = round(win_duration * fs); 
noverlap = round(window_len * 0.8);    
nfft = 4096; 

%% 4. 绘图展示 - Figure 1: 时域与频域对比
% 布局策略：总共 2行4列
% 时域图占左边两列：位置索引为 [1, 2, 5, 6]
% 频域图占右边两列：位置索引为 [3, 4, 7, 8]
wave_idx_map = [1, 2, 5, 6]; 
fft_idx_map  = [3, 4, 7, 8];

figure('Name', 'Figure 1: Waveforms (Left) & Spectra (Right)', 'Color','w', 'Position', [50, 50, 1400, 700]);

for i = 1:N_scenarios
    CurrentSignal = signals(i,:);
    gap_ms = gaps(i) * 1000;
    
    % --- 左半区：时域波形 ---
    subplot(2, 4, wave_idx_map(i)); 
    plot(t*1000, CurrentSignal); 
    grid on; 
    xlim([15 65]); % 聚焦显示 15-65 ms
    ylim([-1.5 1.5]);
    title(['CW',num2str(i), ' 时域 (\Delta t=', num2str(gap_ms), 'ms)']);
    ylabel('幅度');
    xlabel('时间 (ms)')
    % if i > 2; xlabel('时间 (ms)'); end % 只在第二行显示X轴标签
    
    % --- 右半区：FFT 频谱 ---
    subplot(2, 4, fft_idx_map(i));
    L = length(CurrentSignal);
    Y = fft(CurrentSignal);
    P2 = abs(Y);
    P1 = P2(1:L/2+1);
    P1_dB = 20*log10(P1(1:floor(length(P1)/2)) + eps); 
    f_axis = fs*(0:(length(P1_dB)-1))/L;
    
    plot(f_axis/1000, P1_dB, 'b'); 
    grid on; 
    xlim([20 40]); % 聚焦显示 20kHz - 40kHz
    ylim([-20 70]);
    title(['CW信号',num2str(i), ' FFT']);
    ylabel('dB');
    xlabel('频率 (kHz)');
    % if i > 2; xlabel('频率 (kHz)'); end
end

%% 5. 绘图展示 - Figure 2: STFT 时频图
% 布局策略：2行2列
figure('Name', 'Figure 2: STFT Spectrograms', 'Color','w', 'Position', [100, 100, 800, 700]);

for i = 1:N_scenarios
    CurrentSignal = signals(i,:);
    gap_ms = gaps(i) * 1000;
    
    subplot(2, 2, i);
    
    % [s, f_stft, t_stft] = spectrogram(CurrentSignal, hann(window_len), noverlap, nfft, fs);
    % S_dB = 20*log10(abs(s) + eps);
    % imagesc(t_stft*1000, f_stft/1000, S_dB); 
    % 直接调用，不要等号左边的 [s,f,t]
    spectrogram(CurrentSignal, hann(window_len), noverlap, nfft, fs, 'yaxis','power');
    axis xy; 
    colormap jet;

    
    % 统一视觉范围
    ylim([0 60]);   % 频率 0-60 kHz
    xlim([15 65]);  % 时间 15-65 ms (与时域图对齐)
    colorbar;
    clim([-100 0]);  % 动态范围
    
    title(['CW信号',num2str(i), ' STFT (\Delta t=', num2str(gap_ms), 'ms)']);
    ylabel('频率 (kHz)');
    xlabel('时间 (ms)');
end