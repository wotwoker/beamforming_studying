% 1. 定义阵列和信号参数
clear; clc; close all;
N = 64;                 % 阵元数量
d = 0.5;                % 阵元间距 (m)
c = 1500;             % 声速 (m/s)
fs = 5000;            % 采样频率 (Hz)
T = 0.1;                % 信号时长 (s)
t = 0:1/fs:(T-1/fs);    % 时间向量 (1 x M)
M = length(t);          % 信号长度

% --- 2. 模拟接收信号 (一个目标 + 一个强干扰) ---
% 目标信号 (期望信号)
signal_doa_deg_target = -30;
signal_rad_target = deg2rad(signal_doa_deg_target);
f_target = 1000; % 目标频率
s_target = exp(1j * 2 * pi * f_target * t); % 目标信号波形

% 干扰信号 (不期望的信号)
signal_doa_deg_jammer = 10;
signal_rad_jammer = deg2rad(signal_doa_deg_jammer);
f_jammer = 1200; % 干扰频率
% 干扰的功率比目标强100倍 (幅度强10倍)
s_jammer = 10 * exp(1j * 2 * pi * f_jammer * t); 

% 计算各自的接收信号
element_pos = (0:N-1)' * d;
tau_target = element_pos * sin(signal_rad_target) / c;
received_target = exp(1j * 2 * pi * f_target * (t - tau_target));

tau_jammer = element_pos * sin(signal_rad_jammer) / c;
received_jammer = 10 * exp(1j * 2 * pi * f_jammer * (t - tau_jammer));

% 加入背景噪声
noise = 0.5 * (randn(N, M) + 1j * randn(N, M));

% 将目标、干扰和噪声相加，得到最终的接收信号
received_signals = received_target + received_jammer + noise;

% --- 3. 波束形成处理 ---
fprintf('--- 开始波束形成处理 ---\n');
% 计算协方差矩阵 R_xx (CBF和MVDR都需要)
R_xx = (received_signals * received_signals') / M;

% 定义扫描角度
scan_angles_deg = -90:0.5:90; % 使用0.5度步长以获得更精细的图像
scan_angles_rad = deg2rad(scan_angles_deg);

% 预分配内存
power_cbf = zeros(length(scan_angles_rad), 1);
power_mvdr = zeros(length(scan_angles_rad), 1);

% --- 3.1 MVDR算法 ---
fprintf('正在计算MVDR谱...\n');
% MVDR的核心：对协方差矩阵求逆
% 为防止矩阵病态，可以加入微小的对角线加载，但这里我們先直接求逆
R_inv = inv(R_xx); 
wavelen_target = c / f_target; % MVDR导向矢量应使用目标频率

for i = 1:length(scan_angles_rad)
    theta = scan_angles_rad(i);
    % 导向矢量 a(theta)
    a = exp(-1j * 2 * pi * element_pos * sin(theta) / wavelen_target);
    % MVDR功率谱计算公式 P = 1 / (a' * R_inv * a)
    power_mvdr(i) = 1 / (a' * R_inv * a);
end

% --- 3.2 CBF算法 (用于对比) ---
fprintf('正在计算CBF谱 (用于对比)...\n');
for i = 1:length(scan_angles_rad)
    theta = scan_angles_rad(i);
    a = exp(-1j * 2 * pi * element_pos * sin(theta) / wavelen_target);
    % CBF功率谱计算公式 P = a' * R * a
    power_cbf(i) = a' * R_xx * a;
end

% --- 4. 绘图对比 ---
% 归一化处理
power_cbf_norm = abs(power_cbf) / max(abs(power_cbf));
power_cbf_db = 10*log10(power_cbf_norm);

power_mvdr_norm = abs(power_mvdr) / max(abs(power_mvdr));
power_mvdr_db = 10*log10(power_mvdr_norm);

figure;
plot(scan_angles_deg, power_cbf_db, 'b-', 'LineWidth', 1, 'DisplayName', 'CBF');
hold on;
plot(scan_angles_deg, power_mvdr_db, 'r-', 'LineWidth', 2, 'DisplayName', 'MVDR');
grid on;
title('CBF vs. MVDR 方位估计性能对比');
xlabel('角度 (°)');
ylabel('归一化功率 (dB)');
ylim([-60, 5]);

% 绘制目标和干扰的真实方向
plot([signal_doa_deg_target, signal_doa_deg_target], ylim, 'r--', 'DisplayName', '真实目标 DOA');
plot([signal_doa_deg_jammer, signal_doa_deg_jammer], ylim, 'k--', 'DisplayName', '真实干扰 DOA');
legend('show');