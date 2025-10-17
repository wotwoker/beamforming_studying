%% 1. 定义阵列和信号参数
clear; clc; close all;
N = 64;                 % 阵元数量
d = 0.5;                % 阵元间距 (m)
c = 1500;             % 声速 (m/s)
fs = 5000;            % 采样频率 (Hz)
T = 0.1;                % 信号时长 (s)
t = 0:1/fs:(T-1/fs);    % 时间向量 (1 x M)
M = length(t);          % 信号长度
signal_doa_deg = -30;% 信号的真实入射角度
signal_doa_rad = deg2rad(signal_doa_deg); % 转换为弧度
f_start = 1500;         % HFM起始频率 (Hz)
f_end = 500;            % HFM终止频率 (Hz)
f_center = (f_start + f_end) / 2; % 中心频率，用于波束形成

%% 2. 模拟HFM接收信号
k = (f_start / T) * (f_start / f_end - 1);
phi = (2 * pi * f_start^2 / k) * log(1 + (k / f_start) * t);
hfm_signal_base = exp(1j * phi);
element_pos = (0:N-1)' * d;
tau = element_pos * sin(signal_doa_rad) / c;
received_signals = zeros(N, M, 'like', 1j);
time_matrix = t - tau;
for i = 1:N
    received_signals(i, :) = interp1(t, hfm_signal_base, time_matrix(i, :), 'linear', 0);
end

noise = 0.1 * (randn(N, M) + 1j * randn(N, M));
received_signals = received_signals + noise;

%% 3. 频域宽带常规波束形成
% 对每一行（每个阵元）进行FFT，将接收信号转换到频域
fft_received_signals = fft(received_signals, [], 2);
freq_axis = (0:M-1) * fs / M;% 创建频率轴

% 找到HFM信号实际占用的频率范围对应的FFT索引
freq_indices = find(freq_axis >= f_end & freq_axis <= f_start);
num_freq_bins = length(freq_indices);

% 定义扫描角度范围
scan_angles_deg = -90:1:90; 
scan_angles_rad = deg2rad(scan_angles_deg);

beam_output_power = zeros(length(scan_angles_rad), 1);% 预分配内存
% 外层循环扫描角度
for i = 1:length(scan_angles_rad)
    theta = scan_angles_rad(i);
    total_power = 0; % 用于累加所有频率的功率
    
    % 内层循环遍历所有频率点
    for k = 1:num_freq_bins
        % 获取当前频率点的索引和频率值
        current_idx = freq_indices(k);
        current_freq = freq_axis(current_idx);
        
        % 如果频率为0则跳过，避免除零
        if current_freq == 0
            continue;
        end
        
        % 当前频率的导向矢量
        wavelen_k = c / current_freq;
        steering_vector_k = exp(-1j * 2 * pi * element_pos * sin(theta) / wavelen_k);
        
        % 提取该频率点的快照（一个N x 1的复数向量）
        snapshot_k = fft_received_signals(:, current_idx);
        % 在该频率点上进行窄带波束形成
        beam_output_k = steering_vector_k' * snapshot_k;
        % 将该频率点的功率累加到总功率上
        total_power = total_power + abs(beam_output_k)^2;
    end
    beam_output_power(i) = total_power;
end

% 归一化功率
beam_output_power_norm = beam_output_power / max(beam_output_power);
beam_output_power_db = 10 * log10(beam_output_power_norm);

%% 4.计算并打印-3dB波束宽度 (这部分及以后代码无需改动)
[~, peak_idx] = max(beam_output_power_norm);
left_idx = peak_idx;
while left_idx > 1 && beam_output_power_db(left_idx) > -3
    left_idx = left_idx - 1;
end
left_angle = scan_angles_deg(left_idx);
right_idx = peak_idx;
while right_idx < length(scan_angles_deg) && beam_output_power_db(right_idx) > -3
    right_idx = right_idx + 1;
end
right_angle = scan_angles_deg(right_idx);
hpbw = right_angle - left_angle;
fprintf('主瓣峰值位于: %d°\n', scan_angles_deg(peak_idx));
fprintf('-3dB点位于: %d° 和 %d°\n', left_angle, right_angle);
fprintf('波束宽度 (HPBW) 约为: %d°\n', hpbw);

%% 5. 绘制原始方向图
figure(1);
plot(scan_angles_deg, beam_output_power_db, 'DisplayName', 'Broadband CBF');
hold on;
plot([left_angle, right_angle], [-3, -3], 'k', 'LineWidth', 2, 'DisplayName', sprintf('-3dB宽度: %d°', hpbw));
grid on;
title('宽带常规波束形成图 (HFM信号)');
xlabel('角度(°)');
ylabel('归一化功率(dB)');
ylim([-50, 5]);
true_doa_label = sprintf('真实信号DOA (%d°)', signal_doa_deg);
plot([signal_doa_deg, signal_doa_deg], [-50, 5], 'r--', 'DisplayName', true_doa_label);
hold off;
legend('show');

%% 6：计算并绘制波束空间的时间序列 (其导向矢量是基于中心频率的)
wavelen_center = c/f_center;
beamspace_matrix = zeros(length(scan_angles_deg), M, 'like', 1j);
for i = 1:length(scan_angles_rad)
    theta = scan_angles_rad(i);
    steering_vector = exp(-1j * 2 * pi * element_pos * sin(theta) / wavelen_center);
    beam_timeseries_row = steering_vector' * received_signals;
    beamspace_matrix(i, :) = beam_timeseries_row;
end

figure(2);
imagesc(t, scan_angles_deg, abs(beamspace_matrix)); % 这里用abs()来观察能量
set(gca, 'YDir', 'normal'); 
colorbar;
title('波束空间的时间序列 (幅度)');
xlabel('时间 (s)');
ylabel('扫描角度 (°)');