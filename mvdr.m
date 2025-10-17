%% MVDR (Minimum Variance Distortionless Response) 波束形成
clear; clc; close all;

%% 1. 定义阵列和信号参数 (与之前统一)
N = 64;                 % 阵元数量
d = 0.5;                % 阵元间距 (m), 假设为半波长
c = 1500;               % 声速 (m/s)
f = 1500;               % 信号频率 (Hz)
fs = 5000;              % 采样频率 (Hz)
T = 0.1;                % 信号时长 (s)
t = (0:1/fs:T-1/fs).';  % 时间向量 (L x 1), 转置为列向量
L = length(t);          % 采样点数（快拍数）
signal_doa_deg = 35;   % 信号的真实入射角度
SNR_dB = 0;             % 单个阵元的信噪比 (dB)

% 派生量
lambda = c / f;         % 波长
% d = lambda / 2;       % 严格按半波长设计可以这样写
k0 = 2*pi / lambda;     % 波数

%% 2. 模拟CW接收信号
% 定义导向矢量函数 a(theta)
% a(theta) = exp(-j*k0*d*[0:N-1]'*sind(theta))
steering_vector = @(theta_deg) exp(-1j * k0 * d * (0:N-1).' * sind(theta_deg));

% 生成基带信号 s(t)
s = exp(1j * 2 * pi * f * t); % CW (连续波)信号 (L x 1)

% 根据信噪比设置信号幅度
signal_power = 10^(SNR_dB / 10);% 噪声功率已归一化为1，所以信号功率就是 SNR
As = sqrt(signal_power); % 幅度是功率的平方根

% 生成无噪声的信号矩阵 X = a * s^T
a_true = steering_vector(signal_doa_deg); % 真实方向的导向矢量 (N x 1)
X_clean = As * (a_true * s.'); % (N x 1) * (1 x L) -> (N x L)
% 生成功率为1的复高斯白噪声
noise = (randn(N,L) + 1j*randn(N,L)) / sqrt(2);

% 最终接收信号 = 干净信号 + 噪声
received_signals = X_clean + noise;

%% 3. 估算协方差矩阵 R
% R = E[X * X^H]
Rxx = (received_signals * received_signals') / L; % (N x N) 样本协方差矩阵

% 对角加载 (Diagonal Loading)一个很小的正数，以提高稳健性
% 防止Rxx在快拍数不足或信号相干时出现奇异或病态
epsilon = 1e-3; % 加载因子 
Rxx = Rxx + epsilon * trace(Rxx)/N * eye(N); % eye(N)是N维单位矩阵

%% 4. MVDR 波束形成与扫描
% 扫描角度设置
scan_angles_deg = -90:1:90;
K = numel(scan_angles_deg); % K=181 个角度

% 构建扫描角度对应的阵列流形矩阵 S 
S = steering_vector(scan_angles_deg); % (N x K)

% --- MVDR 谱计算 ---
% P_mvdr(theta) = 1 / (a(theta)' * inv(R) * a(theta))
R_inv_S = Rxx \ S; % 反斜杠高效计算 Rxx^{-1} * S
den = sum(conj(S) .* R_inv_S, 1); % 计算分母 a' * inv(R) * a ，sum(,1)指按列求和

B_mvdr_cw = zeros(K, L); %% 初始化波束输出矩阵
% 为每个角度计算最优权重w，并应用到时序信号上
for k = 1:K
    % a) 计算当前角度k的最优权重向量 w_k
    % w_k = (R_inv * a_k) / (a_k' * R_inv * a_k)
    w_k = R_inv_S(:, k) / den(k);
    
    % b) 应用权重向量到整个时间序列信号上 (w' * X)
    % w_k' 是 1xN, received_signals 是 NxL, 结果是 1xL
    B_mvdr_cw(k, :) = w_k' * received_signals;
end

P_mvdr = real(1 ./ den); % MVDR 功率谱
% 归一化为dB
P_mvdr_db = 10*log10(P_mvdr / max(P_mvdr));

% --- CBF 谱计算 (用于对比) ---
% P_cbf(theta) = a(theta)' * R * a(theta)
P_cbf = real(sum(conj(S) .* (Rxx * S), 1));
P_cbf_db = 10*log10(P_cbf / max(P_cbf));

% 估计DOA
[~, idx] = max(P_mvdr);
doa_estimate_mvdr = scan_angles_deg(idx);
fprintf('MVDR 估计角度: %d°\n', doa_estimate_mvdr);

%% 5. 结果可视化
figure(1); 
plot(scan_angles_deg, P_mvdr_db, 'LineWidth', 1, 'DisplayName', 'MVDR');
hold on;
plot(scan_angles_deg, P_cbf_db, '-.', 'LineWidth', 1,'DisplayName', 'CBF (Bartlett)');
xline(signal_doa_deg, '--r', 'LineWidth', 1, 'DisplayName', '真实DOA');
grid on;
xlim([-90, 90]);
ylim([-20, 0]);
title('MVDR 与 CBF 空间谱对比');
xlabel('角度 (°)');
ylabel('归一化功率 (dB)');
legend('show', 'Location', 'best');

figure(2);
imagesc(t, scan_angles_deg, abs(B_mvdr_cw));
set(gca, 'YDir', 'normal');
colorbar;
title('窄带(CW)信号的MVDR波束空间时间序列');
xlabel('时间 (s)');
ylabel('扫描角度 (°)');
fprintf('完成!\n');










%% 6. HFM 信号生成 (宽带场景)
% HFM信号参数
f_start = 1500; % 起始频率
f_end = 500;    % 终止频率

% 生成 HFM 基带信号 (与之前HFM代码相同)
k_hfm = (f_start / T) * (f_start / f_end - 1);
phi = (2*pi*f_start^2/k_hfm) * log(1 + (k_hfm/f_start)*t);
hfm_signal_base = exp(1j * phi);

% 模拟阵列接收 HFM 信号
a_true_hfm_center_freq = exp(-1j * 2*pi*((f_start+f_end)/2)/c * d * (0:N-1).' * sind(signal_doa_deg)); % 仅为概念演示
tau = (d * (0:N-1).' * sind(signal_doa_deg)) / c; % 时延
time_matrix = t.' - tau; % L x N
received_signals_hfm = zeros(N, L);
for n = 1:N
    % interp1(线性插值)函数，在延迟后的时间点time_matrix(n,:)对原始信号 hfm_signal_base 进行重新采样
    received_signals_hfm(n,:) = interp1(t, hfm_signal_base, time_matrix(n,:), 'linear', 0);
end
X_clean_hfm = As * received_signals_hfm;
noise_hfm = (randn(N,L) + 1j*randn(N,L)) / sqrt(2);
received_signals_hfm = X_clean_hfm + noise_hfm;

%% 7. 宽带MVDR处理 (频域非相干平均)
% FFT 参数
N_fft = L; % FFT点数，这里简单取为快拍数
freq_bins = (0:N_fft-1) * fs / N_fft; % 计算每个频箱对应的频率(映射到真实频率)

% 选择处理的频率范围 (只处理信号所在的频带)
freq_indices = find(freq_bins >= f_end & freq_bins <= f_start);
num_freq_to_process = length(freq_indices);
fprintf('将处理 %d 个频箱...\n', num_freq_to_process);

% 对接收信号做FFT
X_fft = fft(received_signals_hfm, N_fft, 2); % 沿时间维(维度2)进行FFT

P_mvdr_hfm_total = zeros(1, K);% 初始化总的MVDR谱
B_mvdr_hfm_freq_domain = zeros(K, N_fft);% 初始化一个复数矩阵，用于存储每个角度在每个频点的复数输出
% 循环遍历每个频箱，执行窄带MVDR，然后累加
for i = 1:num_freq_to_process
    fi = freq_indices(i);
    f_fi = freq_bins(fi); % 当前处理的频率
    
    % a) 获取该频率的快照向量 (N x 1)
    X_fi = X_fft(:, fi);
    
    % b) 估算该频率的协方差矩阵 (CSDM)
    % 在此简单实现中，每个频箱只有一个快照，所以Rxx_fi是秩为1的
    % 故对角加载在这里至关重要
    Rxx_fi = X_fi * X_fi';
    Rxx_fi = Rxx_fi + epsilon * trace(Rxx_fi)/N * eye(N);
    
    % c) 计算当前频率 f_k 对应的导向矢量矩阵
    k0_fi = 2*pi*f_fi / c;
    S_fi = exp(-1j * k0_fi * d * (0:N-1).' * sind(scan_angles_deg));
    
    % d) 计算该频率下，所有角度的MVDR复数输出
    % y(theta, f_fi) = w' * X_fi = (a' * R_inv * X_fi) / (a' * R_inv * a)

    den_fi = sum(conj(S_fi) .* (Rxx_fi \ S_fi), 1).'; % 计算分母
    num_fi = S_fi' * (Rxx_fi \ X_fi); % 计算分子
    Y_fi = num_fi ./ den_fi;% 计算所有角度在该频率下的复数输出 Y_fi
    B_mvdr_hfm_freq_domain(:, fi) = Y_fi; % 存储到频域波束矩阵中

end
% 对每个角度的完整频谱(每一行)进行IFFT，合成时域信号
B_mvdr_hfm_time_domain = ifft(B_mvdr_hfm_freq_domain, N_fft, 2);

% 方向图（把每个角度的时间序列做平均功率）
P_mvdr_hfm = mean(abs(B_mvdr_hfm_time_domain).^2, 2);      % K×1，mean(,2)在时间T上求平均
P_mvdr_hfm_db = 10*log10(P_mvdr_hfm / max(P_mvdr_hfm));

% 估计DOA
[~, idx_hfm] = max(P_mvdr_hfm);
doa_estimate_mvdr_hfm = scan_angles_deg(idx_hfm);
fprintf('[HFM] MVDR 估计角度: %.d°\n', doa_estimate_mvdr_hfm);

%% 8. HFM 结果可视化
figure(3);
plot(scan_angles_deg, P_mvdr_hfm_db, 'LineWidth', 1.5, 'DisplayName', 'MVDR (HFM)');
hold on;
xline(signal_doa_deg, '--r', 'LineWidth', 1.5, 'DisplayName', '真实DOA');
grid on; xlim([-90, 90]); ylim([-42, 0]);
title('宽带(HFM)信号的 MVDR 空间谱 (频域平均法)');
xlabel('角度 (°)'); ylabel('归一化功率 (dB)');
legend('show', 'Location', 'best');

figure(4);
imagesc(t, scan_angles_deg, abs(B_mvdr_hfm_time_domain));
set(gca, 'YDir', 'normal');
colorbar;
title('宽带(HFM)信号的MVDR波束空间时间序列 (频域合成法)');
xlabel('时间 (s)');
ylabel('扫描角度 (°)');