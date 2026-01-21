%% 基于 LFM 信号的均匀水平线阵 (HLA) 宽带观测模型构建
clear; clc; close all;

%% 1. 系统参数设置 ==================
fs = 200e3;              % 采样频率 (200kHz)
T = 0.1;                 % 脉冲宽度 (100ms)
fc = 30e3;               % 中心频率 (30kHz)
B = 20e3;                % 带宽 (20kHz, 即 20k-40k Hz)
mu = B/T;                % 调频斜率
c = 1500;                % 声速 (m/s)

M = 12;                  % 阵元数
d = c/(2*40e3);          % 阵元间距 (按最高频率的半波长布阵)
theta = 15;              % 入射方位角 (度)
snr = -0;                % 信噪比 (dB)

%%  2. 时域 LFM 信号生成 ==================
t = 0:1/fs:T-1/fs;
% 发射信号复包络
s = exp(1j * 2 * pi * (fc * t + 0.5 * mu * t.^2)); 

%% 3. 阵列接收信号构建 (时域) ==================
tau = (0:M-1)' * d * sin(deg2rad(theta)) / c; % 相对时延向量
N_samples = length(t);
X = zeros(M, N_samples);

for m = 1:M
    % 对每个阵元添加时延 (模拟远场宽带信号传播)
    % 这一步模拟了物理上的波前到达延迟
    X(m, :) = exp(-1j * 2 * pi * fc * tau(m)) * s .* ...
              exp(-1j * 1 * pi * mu * tau(m) * t); 
end

% 注入加性高斯白噪声
X = awgn(X, snr, 'measured');

%% 4. 子带分解与 SCM 估计 ==================
% 采用短时傅里叶变换 (STFT) 模拟多快拍
N_fft = 1024;            % FFT 点数
N_overlap = 512;         % 重叠点数

% 获取频率向量 F，用于确定子带索引
% 注意：此处只获取 F 和 T，不需要 P
[~, F, ~] = stft(X(1,:), fs, 'Window', hamming(N_fft), 'OverlapLength', N_overlap);

% 筛选 LFM 带宽内的有效频点 (20kHz - 40kHz)
idx_freq = find(F >= 20e3 & F <= 40e3); 
K = length(idx_freq);    % 子带总数

% 存储各子带的协方差矩阵 Rk
Rk_cell = cell(K, 1);    

fprintf('正在处理子带分解与 SCM 估计...\n');

for k = 1:K
    fk = F(idx_freq(k)); % 第 k 个子带的中心频率
    
    % 初始化当前子带的快拍数据矩阵 (M x N_snapshots)
    % 我们需要知道 stft 后的时间点数，先对第一个阵元做一次来获取尺寸
    if k == 1
         [P_temp, ~, ~] = stft(X(1,:), fs, 'Window', hamming(N_fft), 'OverlapLength', N_overlap);
         N_snapshots = size(P_temp, 2);
         Xk = zeros(M, N_snapshots);
    end
    
    % 提取所有阵元在当前频率 fk 处的观测数据
    for m = 1:M
        % 对第 m 个阵元做 STFT，取第 1 个输出参数 (复数矩阵)
        [P_m, ~, ~] = stft(X(m,:), fs, 'Window', hamming(N_fft), 'OverlapLength', N_overlap);
        
        % 取出对应频率索引的数据行，放入观测矩阵的第 m 行
        Xk(m, :) = P_m(idx_freq(k), :);
    end
    
    % 估计样本协方差矩阵 (SCM)
    % 公式: Rk = (1/N) * Xk * Xk'
    Rk_cell{k} = (Xk * Xk') / N_snapshots; 
end

fprintf('模型构建完成！\n');
fprintf('子带总数 K = %d\n', K);
fprintf('每个子带的快拍数 (Snapshots) = %d\n', N_snapshots);

%% 5. 简单验证 ==================
% 对第一个子带做特征分解，检查是否存在主特征值
[U, D] = eig(Rk_cell{1});
eigenvalues = diag(D);
[sorted_eig, sort_idx] = sort(eigenvalues, 'descend');

disp('------------------------------------------------');
disp(['第一个子带 (' num2str(F(idx_freq(1))/1000) ' kHz) 的特征值分布(前5个):']);
disp(sorted_eig(1:5));
disp('说明：若第一个特征值显著大于其他，说明信号子空间建立成功。');


%% 6. 基础 ISSM 算法实现 ==================
fprintf('正在执行 ISSM 宽带测向...\n');

% 1. 定义扫描角度范围 (Search Grid)
scan_angles = -90:0.1:90;  % 从 -90度 扫到 90度，步长 0.1度
N_scan = length(scan_angles);

% 初始化融合谱 (用于累加所有子带的谱)
P_ISSM_sum = zeros(1, N_scan); 

% 2. 循环处理每个子带
for k = 1:K
    % 获取当前子带的频率和协方差矩阵
    fk = F(idx_freq(k));
    Rk = Rk_cell{k};
    
    % --- (A) 特征分解 (EVD) ---
    [U, D] = eig(Rk);
    eig_vals = diag(D);
    [~, I] = sort(eig_vals, 'descend');
    U_sorted = U(:, I);
    
    % 估计信号源个数 (此处假设已知源个数为1，或者通过 MDL/AIC 准则估计)
    n_sources = 1; 
    
    % 提取噪声子空间 Un (第 n_sources+1 到 M 列)
    Un = U_sorted(:, n_sources+1:end);
    
    % --- (B) 计算当前子带的空间谱 P_k(\theta) ---
    % MUSIC 公式: P(theta) = 1 / (a^H * Un * Un^H * a)
    P_k = zeros(1, N_scan);
    
    % 预计算 Un * Un' 以加速
    UnUnH = Un * Un';
    
    for i_ang = 1:N_scan
        theta_scan = scan_angles(i_ang);
        
        % 构造当前频率 fk 下的导向矢量 a(fk, theta)
        % 注意：导向矢量是频率 fk 的函数！这是宽带的关键。
        tau_scan = (0:M-1)' * d * sin(deg2rad(theta_scan)) / c;
        a_vec = exp(-1j * 2 * pi * fk * tau_scan);
        
        % 计算谱值
        denom = a_vec' * UnUnH * a_vec;
        P_k(i_ang) = 1 / abs(denom);
    end
    
    % --- (C) 累加到总谱中 (等权融合) ---
    % 在实际 ISSM 中，通常直接累加
    P_ISSM_sum = P_ISSM_sum + P_k;
end

% 3. 计算平均谱并归一化
P_ISSM = P_ISSM_sum / K;
P_ISSM_norm = 10 * log10(P_ISSM / max(P_ISSM)); % 转为 dB 并归一化

% ================== 7. 绘图与结果展示 ==================
figure;
plot(scan_angles, P_ISSM_norm, 'LineWidth', 2,'DisplayName', 'ISSM-MUSIC');
grid on;
xline(theta, '--k', 'LineWidth', 1, 'DisplayName', '真实角度');
legend('show', 'Location', 'best');
xlabel('方位角 (Degree)');
ylabel('空间谱幅度 (dB)');
title(['ISSM 宽带测向结果 (LFM信号, 真实角度=' num2str(theta) '^\circ, SNR=' num2str(snr) 'dB)']);
xlim([-90 90]);
%ylim([-60 0]);

% 找峰值
[~, idx_peak] = max(P_ISSM_norm);
est_theta = scan_angles(idx_peak);
fprintf('真实角度: %.2f 度\n', theta);
fprintf('ISSM 估计角度: %.2f 度\n', est_theta);
fprintf('误差: %.2f 度\n', abs(theta - est_theta));


%% 7. W-ISSM 加权融合实现 ==================
% 目标：根据子带的“质量”分配权重，而不是像上面那样直接相加

fprintf('正在执行 W-ISSM (基于能量加权)...\n');

% 1. 计算每个子带的权重 w_k
weights = zeros(K, 1);

for k = 1:K
    % 获取第 k 个子带的协方差矩阵
    Rk = Rk_cell{k};
    
    % --- 权重策略 A: 基于信号子空间能量 (开题报告公式 4) ---
    % 逻辑：能量越大的子带，信噪比可能越高，给它更大的发言权
    % P_signal = trace(Rk) - M * noise_power (简化版直接用 trace)
    
    % 这里我们用最大特征值来代表信号强度 (比 trace 更抗噪)
    eigs_k = sort(eig(Rk), 'descend');
    signal_power = eigs_k(1); % 主特征值
    
    weights(k) = signal_power; % 暂存权重
end

% 归一化权重，使 sum(weights) = 1
weights = weights / sum(weights);

% 2. 绘制权重分布 (这一步对论文很重要，展示你的权重是否合理)
figure;
plot(F(idx_freq)/1000, weights, 'o-', 'LineWidth', 1.5);
xlabel('频率 (kHz)'); ylabel('权重值');
title('各子带权重分布 (基于最大特征值)');
grid on;

% 3. 执行加权融合
P_WISSM_sum = zeros(1, N_scan); 

for k = 1:K
    fk = F(idx_freq(k));
    Rk = Rk_cell{k};
    wk = weights(k); % 获取当前子带的权重
    
    % --- 重复 ISSM 的计算步骤 ---
    [U, D] = eig(Rk);
    [~, I] = sort(diag(D), 'descend');
    Un = U(:, I(2:end)); % 噪声子空间 (假设单目标)
    UnUnH = Un * Un';
    
    % 计算当前子带谱 P_k
    P_k = zeros(1, N_scan);
    for i_ang = 1:N_scan
        theta_scan = scan_angles(i_ang);
        tau_scan = (0:M-1)' * d * sin(deg2rad(theta_scan)) / c;
        a_vec = exp(-1j * 2 * pi * fk * tau_scan);
        
        denom = a_vec' * UnUnH * a_vec;
        P_k(i_ang) = 1 / abs(denom);
    end
    
    % --- 关键点：加权累加 ---
    % 以前是: P_sum = P_sum + P_k
    % 现在是: P_sum = P_sum + wk * P_k
    P_WISSM_sum = P_WISSM_sum + wk * P_k;
end

% 4. 结果归一化与对比
P_WISSM_norm = 10 * log10(P_WISSM_sum / max(P_WISSM_sum));

figure;
plot(scan_angles, P_ISSM_norm, '--', 'LineWidth', 1.5, 'DisplayName', '传统等权 ISSM');
hold on;
plot(scan_angles, P_WISSM_norm, 'r-', 'LineWidth', 2, 'DisplayName', '加权 W-ISSM (能量)');
grid on;
xline(theta, '--k', 'Label', '真实角度', 'LineWidth', 1.5, 'LabelVerticalAlignment', 'bottom', 'DisplayName', '真实角度');
legend('show', 'Location', 'best');
xlabel('方位角 (Degree)');
ylabel('空间谱 (dB)');
title('ISSM vs W-ISSM 性能对比');
xlim([theta-5, theta+5]); % 放大看峰值附近

% 找峰值
[~, idx_peak] = max(P_WISSM_norm);
est_theta = scan_angles(idx_peak);
fprintf('真实角度: %.2f 度\n', theta);
fprintf('WISSM 估计角度: %.2f 度\n', est_theta);
fprintf('误差: %.2f 度\n', abs(theta - est_theta));