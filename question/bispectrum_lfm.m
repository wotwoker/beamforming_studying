%% LFM 信号综合分析：时域、FFT 与 真正的双谱 (低频基带版)
clear; clc; close all;

%%  1. 全局参数设置 (适配低频)
fs = 20e3;              % 采样率：20kHz 
T_total = 1.0;          % 总时长 (加长时长以获得更好的频率分辨率)
t = 0:1/fs:T_total-1/fs;% 时间轴

% LFM 信号参数 (1kHz -> 4kHz)
f_start = 1000;         
f_end   = 4000;         
T_pulse = 0.02;         % 脉宽：50ms (低频信号脉宽设长一点，特征更明显)
B = f_end - f_start;    % 带宽：3kHz
K = B / T_pulse;        % 调频斜率

% 定义 4 种时延间隔 (低频周期长，间隔需要设大一点才能看到条纹)
% 10ms, 20ms, 40ms, 80ms
gaps = [0.010, 0.020, 0.040, 0.080]; 
N_scenarios = length(gaps);

% 定义窗函数 (Tukey窗)
N_pulse_pts = round(T_pulse * fs);
win = tukeywin(N_pulse_pts, 0.1)'; 

% 预分配信号矩阵
signals = zeros(N_scenarios, length(t));

%%  2. 信号生成
fprintf('正在生成信号...\n');
for i = 1:N_scenarios
    gap = gaps(i);
    
    % --- 脉冲 1 (固定在 0.1s 处开始) ---
    t_p1 = 0.1; 
    idx1_start = round(t_p1 * fs) + 1;
    idx1_end = idx1_start + N_pulse_pts - 1;
    
    % 生成 LFM 波形
    t_local = (0:N_pulse_pts-1)/fs;
    chirp_wave = cos(2*pi*(f_start * t_local + 0.5 * K * t_local.^2));
    
    % 填入脉冲 1
    signals(i, idx1_start:idx1_end) = win .* chirp_wave;
    
    % --- 脉冲 2 (真正的时延逻辑) ---
    % 时延逻辑: t_start2 = t_start1 + gap
    t_p2 = t_p1 + gap;
    idx2_start = round(t_p2 * fs) + 1;
    idx2_end = idx2_start + N_pulse_pts - 1;
    
    if idx2_end <= length(t)
        signals(i, idx2_start:idx2_end) = win .* chirp_wave;
    end
    
    % 加一点微量噪声
    signals(i, :) = signals(i, :) + 0.001 * randn(size(t));
end

%% 3. Figure 1: 时域与 FFT 频谱 (2行4列布局)
fprintf('正在绘制 Figure 1...\n');
figure('Name', 'Figure 1: Time Domain (Left) & FFT Spectrum (Right)', 'Color','w', 'Position', [50, 50, 1400, 700]);

% 子图位置映射
wave_pos = [1, 2, 5, 6];
fft_pos  = [3, 4, 7, 8];

for i = 1:N_scenarios
    sig = signals(i, :);
    gap_ms = gaps(i) * 1000;
    
    % --- A. 时域波形 ---
    subplot(2, 4, wave_pos(i));
    plot(t*1000, sig);
    grid on;
    % 自动聚焦到波形区域
    xlim([90, 100 + gap_ms + 60]); 
    ylim([-1.2 1.2]);
    
    title(['LFM信号', num2str(i), ' 时域 (\Delta t=', num2str(gap_ms), 'ms)']);
    ylabel('幅度');
    xlabel('时间 (ms)');
    % if i > 2; xlabel('时间 (ms)'); end
    
    % --- B. FFT 频谱 ---
    subplot(2, 4, fft_pos(i));
    Nfft = length(sig); 
    Y = fft(sig);
    f_axis = (0:Nfft-1)*(fs/Nfft);
    
    % 只画 0 - 5kHz 范围
    f_idx_max = round(5000 / (fs/Nfft));
    
    Y_segment = abs(Y(1:f_idx_max));
    f_segment = f_axis(1:f_idx_max);
    
    % 对数刻度
    Y_dB = 20*log10(Y_segment + eps);
    
    plot(f_segment, Y_dB, 'b'); % x轴单位 Hz
    grid on;
    xlim([500 4500]); % 聚焦 0.5k - 4.5k
    ylim([-15 45]); 
    
    title(['LFM信号', num2str(i), ' FFT']);
    ylabel('dB');
    xlabel('频率 (Hz)')
    % if i > 2; xlabel('频率 (Hz)'); end
    
    % 标注波纹频率
    ripple_hz = 1/gaps(i);
    text(4000, 40, ['Ripple: ~' num2str(ripple_hz) 'Hz'], ...
        'HorizontalAlignment','right', 'FontSize', 8);
end

%% 4. Figure 2: 双谱分析 (含局部放大)

fprintf('正在计算并绘制 Figure 2 (带局部放大)... \n');
figure('Name', 'Figure 2: LFM True Bispectrum with Inset', 'Color','w', 'Position', [100, 50, 1000, 800]);

% 定义想要放大的“特写” (Zoom Region)
zoom_start = 1600;
zoom_end   = 1900;

for i = 1:N_scenarios
    sig = signals(i, :);
    gap = gaps(i);
    
    % 1. 计算双谱
    Nfft = length(sig);
    X = fft(sig);
    f_axis = (0:Nfft-1)*(fs/Nfft);
    
    f_roi_max = 4000;
    idx_max = find(f_axis <= f_roi_max, 1, 'last');
    
    step = 2; 
    roi_indices = 1 : step : idx_max;
    freq_roi = f_axis(roi_indices);
    
    [I, J] = meshgrid(roi_indices, roi_indices);
    K_indices = I + J - 1;
    mask_valid = K_indices <= ceil(Nfft/2);
    
    X_f1 = X(I);
    X_f2 = X(J);
    X_f3 = zeros(size(K_indices));
    X_f3(mask_valid) = X(K_indices(mask_valid));
    
    B_spec = X_f1 .* X_f2 .* conj(X_f3);
    B_dB = 20*log10(abs(B_spec) + eps);
    B_dB = B_dB - max(B_dB(:)); 
    
    % 2. 绘制主图 (Main Plot)
    h_main = subplot(2, 2, i); 
    imagesc(freq_roi, freq_roi, B_dB);
    axis xy; colormap jet; 
    
    clim([-40 0]); 
    xlim([1000 3000]); 
    ylim([1000 3000]);
    colorbar; 
    
    title(['LFM信号', num2str(i), ' 双谱 (\Delta t=', num2str(gap*1000), 'ms)']);
    xlabel('f_1 (Hz)'); ylabel('f_2 (Hz)');
    
    % 3. 绘制局部放大图 (Inset Plot) 
    
    % A. 获取主图在 Figure 中的位置 [left, bottom, width, height]
    % 注意：此时获取的位置是已经被 colorbar 挤压后的位置
    main_pos = get(h_main, 'Position');
    
    % B. 定义小图的大小和位置
    % 修改：稍微改小一点比例 (0.3)，防止太大
    inset_scale = 0.3; 
    inset_width  = main_pos(3) * inset_scale; 
    inset_height = main_pos(4) * inset_scale;
    
    % 修改：定义一个安全边距 (Margin)，防止贴着色柱
    margin_right = 0.08; % 距离右边界 8%
    margin_top   = 0.02; % 距离上边界 
    
    % 计算小图左下角坐标：
    % 左边 = 主图左边 + 主图宽 - 小图宽 - 右边距
    inset_left = main_pos(1) + main_pos(3) - inset_width - margin_right;
    % 下边 = 主图下边 + 主图高 - 小图高 - 上边距
    inset_bottom = main_pos(2) + main_pos(4) - inset_height - margin_top;
    
    % C. 新建坐标轴
    h_inset = axes('Position', [inset_left, inset_bottom, inset_width, inset_height]);
    
    % D. 重画数据
    imagesc(freq_roi, freq_roi, B_dB);
    axis xy; colormap jet; 
    clim([-40 0]);         
    
    % E. 锁定放大区域
    xlim([zoom_start zoom_end]);
    ylim([zoom_start zoom_end]);
    
    % F. 美化小图 (关键修改：显示刻度)
    box on; 
    % 设置坐标轴颜色为白色(在深蓝背景下可见)
    set(gca, 'XColor', 'w', 'YColor', 'w');
    
    % 如果觉得刻度数字挡住了图，可以把刻度标签放在上面/右边，或者保持默认
    set(gca, 'XAxisLocation', 'top', 'YAxisLocation', 'right'); 
    
    % % 加个半透明标题说明
    % title('Zoom','Color', 'w', 'FontSize', 8, 'FontWeight', 'bold');
    text(0.5, -0.01, 'Zoom Region', ...
        'Units', 'normalized', ...           % 使用相对坐标(0-1)
        'HorizontalAlignment', 'center', ... % 水平居中
        'VerticalAlignment', 'top', ...      % 文字顶端对齐坐标点
        'Color', 'w', ...
        'FontSize', 8, ...
        'FontWeight', 'bold');
end

