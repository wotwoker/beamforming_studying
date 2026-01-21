%% 信号与系统仿真：时频分析对比
% 仿真目标：生成两个时域不同、但幅频特性相同的信号，并通过STFT进行区分
% 信号A：先 50Hz，后 200Hz
% 信号B：先 200Hz，后 50Hz

clear; clc; close all;

%% 1. 参数设置
fs = 5000;              % 采样率 5000Hz (足够分辨200Hz信号)
T = 0.6;                % 总时长 0.6秒
t = 0:1/fs:T-1/fs;      % 时间向量

% 定义脉冲出现的时间段
t_pulse1 = (t >= 0.05) & (t <= 0.15); % 第一个脉冲时间窗 (0.05s - 0.15s)
t_pulse2 = (t >= 0.45) & (t <= 0.55); % 第二个脉冲时间窗 (0.45s - 0.55s)

% 定义脉冲的包络 (使用汉宁窗模拟图中的梭形包络)
% 计算第一段的真实长度
len1 = sum(t_pulse1);
env1 = hann(len1)'; % 生成对应长度的窗
% 计算第二段的真实长度 (它可能比第一段少1个点)
len2 = sum(t_pulse2);
env2 = hann(len2)'; % 生成对应长度的窗

%% 2. 信号生成
% 信号 A: 先 50Hz (低频), 后 200Hz (高频)
sigA = zeros(size(t));
sigA(t_pulse1) = 1.0 * env1 .* cos(2*pi*50*t(t_pulse1));
sigA(t_pulse2) = 1.0 * env2 .* cos(2*pi*200*t(t_pulse2));

% 信号 B: 先 200Hz (高频), 后 50Hz (低频)
sigB = zeros(size(t));
sigB(t_pulse1) = 1.0 * env1 .* cos(2*pi*200*t(t_pulse1));
sigB(t_pulse2) = 1.0 * env2 .* cos(2*pi*50*t(t_pulse2));

%% 3. 计算频谱 (FFT)
L = length(t);
f = fs*(0:(L/2))/L; % 频率轴，频率分辨率乘索引值，映射到真实频率

% 计算信号 A 的 FFT
Y_A = fft(sigA); % 双边谱，如有L个采样点，会输出L个频率点
P2_A = abs(Y_A/L); % 归一化，
P1_A = P2_A(1:L/2+1); % 截取前一半，正频率的部分
P1_A(2:end-1) = 2*P1_A(2:end-1); % 将负频率的能量加回来
P1_A_dB = 20*log10(P1_A + eps); % 转换为 dB，加 eps 防止 log(0)

% 计算信号 B 的 FFT
Y_B = fft(sigB);
P2_B = abs(Y_B/L);
P1_B = P2_B(1:L/2+1);
P1_B(2:end-1) = 2*P1_B(2:end-1);
P1_B_dB = 20*log10(P1_B + eps);

%% 4. 绘图展示
figure('Color','w', 'Position', [100, 100, 1000, 700]);

% --- 第一行：时域波形 ---
subplot(3,2,1);
plot(t, sigA, 'LineWidth', 1.2); title('信号 A 时域 (先50Hz 后200Hz)');
xlabel('时间 (s)'); ylabel('幅度'); grid on; axis([0 0.6 -1 1]);

subplot(3,2,2);
plot(t, sigB, 'LineWidth', 1.2); title('信号 B 时域 (先200Hz 后50Hz)');
xlabel('时间 (s)'); ylabel('幅度'); grid on; axis([0 0.6 -1 1]);

% --- 第二行：频谱图 (FFT) ---
subplot(3,2,3);
plot(f, P1_A_dB, 'b', 'LineWidth', 1); 
title('信号 A 幅度谱 (FFT)');
xlabel('频率 (Hz)'); ylabel('幅度 (dB)'); grid on; xlim([0 500]); ylim([-80 0]);
text(50, -10, '50Hz', 'HorizontalAlignment','center');
text(200, -10, '200Hz', 'HorizontalAlignment','center');

subplot(3,2,4);
plot(f, P1_B_dB, 'b', 'LineWidth', 1); 
title('信号 B 幅度谱 (FFT)');
xlabel('频率 (Hz)'); ylabel('幅度 (dB)'); grid on; xlim([0 500]); ylim([-80 0]);
text(50, -10, '50Hz', 'HorizontalAlignment','center');
text(200, -10, '200Hz', 'HorizontalAlignment','center');
% 注意：你会发现上面两张图几乎一模一样！

% --- 第三行：语谱图 (Spectrogram) ---
% 设置 STFT 参数
% 窗长选择：64ms
% 理由：50Hz周期为20ms，64ms包含了约3.2个周期，频率分辨率足够

% 动态计算窗参数
win_duration = 0.064;           % 设定窗长为 64ms
window_len = round(win_duration * fs); % 在 fs=5000 时，这里是 320 点
noverlap = round(window_len * 0.8);    % 80% 重叠，让图像更平滑
nfft = 4096;  % FFT点数设大一点，频率轴更密（但物理频率分辨率还是1/win_ms)

subplot(3,2,5);
% 注意：Spectrogram 如果指定了 fs，输出单位默认是 Hz，不需自己换算
spectrogram(sigA, hann(window_len), noverlap, nfft, fs, 'yaxis');
title('信号 A 时频图 (STFT)'); 
ylim([0 0.3]); % 显示 0 - 300 Hz (这里的单位是 kHz，所以是 0.3)
colormap jet;

subplot(3,2,6);
[s, f, t] = spectrogram(sigB, hann(window_len), noverlap, nfft, fs);
S_dB = 20*log10(abs(s) + eps);
imagesc(t, f, S_dB);
axis xy; 
title('信号 B 时频图 (STFT)');
ylabel('频率 (Hz)');
xlabel('时间 (s)');
ylim([0 300]);            % 同样改为 300
colormap jet;
colorbar;
clim([-150 0])