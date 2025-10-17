% CBF for CW和HFM信号
%% 1. 定义阵列和信号参数
clear; clc; close all;

N = 64;                 % 阵元数量
d = 0.5;                % 阵元间距 (m)
c = 1500;             % 声速 (m/s)
f = 1500;             % 信号频率 (Hz),小于1500
fs = 5000;            % 采样频率 (Hz)
T = 0.1;                % 信号时长 (s)
t = 0:1/fs:(T-1/fs);    % 时间向量 (1 x M)
L = length(t);          % 采样点数（快拍数）

signal_doa_deg = -30;% 信号的真实入射角度
signal_doa_rad = deg2rad(signal_doa_deg); % 转换为弧度

% 派生量
lambda = c / f;  % d/λ=0.5，正好半波长，不会栅瓣
k0 = 2*pi / lambda;      % 波数(空间角频率，rad/m)
element_pos = (0:N-1).' * d;      % 阵元位置 (N x 1)
% 扫描角设置（用于方向图/波束空间）
scan_angles_deg = -90:1:90;
scan_angles_rad = deg2rad(scan_angles_deg);
K = numel(scan_angles_deg); % K=181 个角度

%% 2. 模拟CW接收信号
tau = element_pos * sin(signal_doa_rad) / c; % 每个阵元相对于第一个阵元的时间延迟
% 生成每个阵元接收到的信号 (N x M 矩阵)
time_matrix = t - tau; % 第 n 个阵元在全局时间为t(m)的那一刻，所接收到的信号
received_signals = exp(1j * 2 * pi * f * time_matrix); % 窄带复指数

% 加入一些噪声
noise = 0.1 * (randn(N, L) + 1j * randn(N, L));
received_signals = received_signals + noise;

%% 3. 相移波束形成
% 取 t=0 时刻的快照 (N x 1 向量)
% S = [a(theta1), a(theta2), ...] (N x K) —— 阵列流形矩阵
% a(theta) = exp(-j*k0*element_pos*sin(theta))
S = exp(-1j * k0 * (element_pos * sin(scan_angles_rad)));  % N x K，省去单个导向矩阵循环扫描
Rxx = (received_signals * received_signals') / L; % (N x N) 样本协方差，比单个snapshot更平均
P = real( sum( conj(S) .* (Rxx * S), 1 ).' );  % (K x 1)，方向图：P(θ) = a^H Rxx a

% 归一化功率
to_dB = @(p) 10*log10( p / max(p(:)) + eps );% 定义归一化转dB函数
P_db = to_dB(P);% 转换为dB

%% 4.计算并打印-3dB波束宽度
% 找到峰值索引
[~, peak_idx] = max(P_db);
% 从峰值向左寻找-3dB点
left_idx = peak_idx;
while left_idx > 1 && P_db(left_idx) > -3
    left_idx = left_idx - 1;
end
left_angle = scan_angles_deg(left_idx);
% 从峰值向右寻找-3dB点
right_idx = peak_idx;
while right_idx < length(scan_angles_deg) && P_db(right_idx) > -3
    right_idx = right_idx + 1;
end
right_angle = scan_angles_deg(right_idx);
% 计算波束宽度
hpbw = right_angle - left_angle;
fprintf('主瓣峰值位于: %d°\n', scan_angles_deg(peak_idx));
fprintf('-3dB点位于: %d° 和 %d°\n', left_angle, right_angle);
fprintf('波束宽度 (HPBW) 约为: %d°\n', hpbw);

%% 5. 绘制原始方向图
figure(1);
plot(scan_angles_deg, P_db, 'DisplayName', 'CBF');
hold on;
plot([left_angle, right_angle], [-3, -3], 'k', 'LineWidth', 2, 'DisplayName', sprintf('-3dB宽度: %d°', hpbw));
grid on;
title('常规波束形成图(CW)');
xlabel('角度(°)'); ylabel('归一化功率(dB)');
xlim([-90, 90]); ylim([-50, 5]);
true_doa_label = sprintf('真实信号DOA (%d°)', signal_doa_deg);
plot([signal_doa_deg, signal_doa_deg], [-50, 5], 'r--', 'DisplayName', true_doa_label);
hold off;
legend('show'); 

%% 6：计算并绘制波束空间的时间序列
B_phase = (conj(S).') * received_signals; % 相移法，每一行是一条角度的时序
%B_phase_pow = abs(B_phase).^2;
% B_phase_db  = to_dB(B_phase_pow);

figure(2);
imagesc(t, scan_angles_deg, abs(B_phase)); % imagesc绘制颜色图(X轴, Y轴, 矩阵数据）
set(gca, 'YDir', 'normal'); % 将y轴方向设置为正常（-90在下，90在上），美化图表
colorbar; % 颜色条
title('波束空间的时间序列(相移法，幅度)');
xlabel('时间 (s)');
ylabel('扫描角度 (°)');






%% HFM：宽带延时求和波束形成 
% 参数
f_start  = 1500;                     % HFM 起始频率 (Hz)
f_end    = 500;                      % HFM 终止频率 (Hz)
f_center = (f_start + f_end)/2;      % 仅作参考，不用于相移
% 生成 HFM 基带 (单位幅度的复指数)
k_hfm = (f_start / T) * (f_start / f_end - 1);                 % 超曲调频参数
phi = (2*pi*f_start^2/k_hfm) * log(1 + (k_hfm/f_start)*t);           % 相位
hfm_signal_base = exp(1j * phi);

% 阵列接收（按真实 DOA 的几何延时）
tau = element_pos * sin(signal_doa_rad) / c;            % N×1
time_matrix = t - tau;                                  % N×M
received_signals_hfm = zeros(N, L);  % 复数矩阵
for n = 1:N
    % 对于每个阵元 n，其接收信号是在延迟后的时间点time_matrix(n,:) 对原始hfm_signal_base进行采样的结果
    received_signals_hfm(n,:) = interp1(t, hfm_signal_base, time_matrix(n,:), 'linear', 0);
end
% 加噪
received_signals_hfm = received_signals_hfm + 0.1*(randn(N,L) + 1j*randn(N,L));

%% 宽带延时求和：按角度扫描对齐并求功率方向图
B_das = zeros(K, L);  % K×L，每一行是某扫描角的时间序列
for kk = 1:K
    theta = scan_angles_rad(kk);
    tau_scan = element_pos * sin(theta) / c; % N×1，扫描角的补偿延时
    y_sum = zeros(1, L);
    for n = 1:N
        % 延时求和，对第 n 个阵元做 +tau_scan(n) 的时间对齐（若扫描角=真实角，则对齐到同相）
        y_n = interp1(t, received_signals_hfm(n,:), t + tau_scan(n), 'linear', 0);
        y_sum = y_sum + y_n;
    end
    B_das(kk,:) = y_sum / N;  % 均衡加权
end

% 方向图（把每个角度的时间序列做平均功率）
P_hfm = mean(abs(B_das).^2, 2);      % K×1，mean(,2)在时间T上求平均
P_db_hfm = 10*log10(P_hfm / max(P_hfm) + eps);

% 计算并打印 -3 dB 波束宽度
[~, peak_idx_hfm] = max(P_db_hfm);
left_idx = peak_idx_hfm;
while left_idx > 1 && P_db_hfm(left_idx) > -3
    left_idx = left_idx - 1;
end
right_idx = peak_idx_hfm;
while right_idx < length(scan_angles_deg) && P_db_hfm(right_idx) > -3
    right_idx = right_idx + 1;
end
left_angle_hfm  = scan_angles_deg(left_idx);
right_angle_hfm = scan_angles_deg(right_idx);
hpbw_hfm = right_angle_hfm - left_angle_hfm;

fprintf('[HFM] 主瓣峰值位于: %d°\n', scan_angles_deg(peak_idx_hfm));
fprintf('[HFM] -3dB点位于: %d° 和 %d°\n', left_angle_hfm, right_angle_hfm);
fprintf('[HFM] 波束宽度 (HPBW) 约为: %d°\n', hpbw_hfm);

%% HFM 宽带延时求和方向图
figure(3);
plot(scan_angles_deg, P_db_hfm, 'DisplayName', 'DAS (HFM)');
hold on;
plot([left_angle_hfm, right_angle_hfm], [-3, -3], 'k', 'LineWidth', 2, ...
    'DisplayName', sprintf('-3dB宽度: %d°', hpbw_hfm));
plot([signal_doa_deg, signal_doa_deg], [-50, 5], 'r--', 'DisplayName', ...
    sprintf('真实信号DOA (%d°)', signal_doa_deg));
grid on; xlim([-90,90]); ylim([-50, 5]);
title('HFM 宽带延时求和方向图');
xlabel('角度(°)'); ylabel('归一化功率(dB)');
legend('show'); hold off;

%% HFM 波束空间的时间序列（延时求和）
figure(4);
imagesc(t, scan_angles_deg, abs(B_das));     % 幅度（也可改成 10*log10(abs(B_das).^2)）
set(gca, 'YDir', 'normal');
colorbar;
title('HFM 波束空间的时间序列（延时求和，幅度）');
xlabel('时间 (s)'); ylabel('扫描角度 (°)');

%% 可视化生成的HFM基带信号
% 为了绘图，我们需要脚本第一部分定义的变量: t, hfm_signal_base, f_start, f_end, T, phi, fs
% --- 创建新图形窗口 ---
figure(5);
% --- 1. 绘制信号的实部波形 ---
subplot(2, 1, 1);
plot(t, real(hfm_signal_base),'DisplayName', 'HFM实部');
grid on;
title('HFM时域波形 (信号实部)');
xlabel('时间 (s)');ylabel('幅度');legend('show');


% --- 2. 绘制瞬时频率 ---
% 瞬时角频率是相位的导数 w(t) = d(phi)/dt
% 瞬时频率 f(t) = w(t) / (2*pi)
% 用差分来近似求导
inst_freq = diff(unwrap(phi)) ./ (2 * pi * diff(t)); % diff(t)的每个值都是1/fs

subplot(2, 1, 2);
% diff会使向量长度减1，所以时间轴也要对应地去掉最后一个点
plot(t(1:end-1), inst_freq,'DisplayName', 'HFM瞬时频率'); 
grid on;
title('HFM瞬时频率');
xlabel('时间 (s)');ylabel('频率 (Hz)');legend('show');
ylim([f_end-100, f_start+100]); % 设置一个合适的Y轴范围


% --- 3. (推荐) 绘制谱图 (Spectrogram)，最直观的显示方式 ---
% 需要 Signal Processing Toolbox
figure(6);
spectrogram(hfm_signal_base, kaiser(128, 5), 120, 256, fs, 'yaxis');
title('HFM信号的谱图（时频分析）');