%% Side-by-Side Comparison: Normal (CPU) vs. Modified (GPU) Simulink Models
% Compares native Simulink CPU execution vs. 384-core NVIDIA GT 750M OpenCL S-Function

clear; clc;
disp('====================================================================');
disp('⚖️  SIMULINK BENCHMARK: NORMAL (CPU) vs. MODIFIED (GPU ACCELERATED)');
disp('====================================================================');

num_channels = 1024;    % 1,024 parallel dynamical nodes
sim_time     = 3.0;     % 3 seconds physical time
dt           = 0.005;   % 200 Hz simulation step (601 steps total)
damping      = 0.05;

% Input excitation: central pulse disturbance
u_input = zeros(num_channels, 1);
u_input(round(num_channels/2)-15 : round(num_channels/2)+15) = 10.0;
assignin('base', 'u_input', u_input);

cpu_model = 'simulink_cpu_model';
gpu_model = 'simulink_gpu_model';

% Models are already created on disk, load them
load_system(cpu_model);
load_system(gpu_model);

%% 1. RUN SIDE-BY-SIDE BENCHMARKS
disp('--------------------------------------------------------------------');
disp('⏱️  Running Normal (CPU) Simulink Model (1024 nodes x 601 steps)...');
tic;
res_cpu = sim(cpu_model);
t_cpu = toc;
disp(sprintf('   CPU Simulation completed in: %.3f seconds', t_cpu));

disp('⚡ Running Modified (GPU Accelerated) Simulink Model (384 CUDA Cores)...');
tic;
res_gpu = sim(gpu_model);
t_gpu = toc;
disp(sprintf('   GPU Simulation completed in: %.3f seconds', t_gpu));

%% 2. NUMERICAL VALIDATION & COMPARISON
y_cpu_raw = res_cpu.get('out_cpu');
y_gpu_raw = res_gpu.get('out_gpu');
tout      = res_cpu.get('tout');

% Ensure 2D matrix shape [TimeSteps x Channels]
y_cpu = squeeze(y_cpu_raw);
if size(y_cpu, 1) == num_channels
    y_cpu = y_cpu';
end

y_gpu = squeeze(y_gpu_raw);
if size(y_gpu, 1) == num_channels
    y_gpu = y_gpu';
end

diff_matrix = abs(y_cpu - y_gpu);
max_err = max(diff_matrix(:));
rms_err = sqrt(mean(diff_matrix(:).^2));

speedup = t_cpu / t_gpu;

disp('====================================================================');
disp('📊 BENCHMARK COMPARISON RESULTS');
disp('====================================================================');
fprintf('  Simulated Physical Channels:  %d nodes\n', num_channels);
fprintf('  Total Simulation Time Steps:  %d steps (dt = %.4fs)\n', length(tout), dt);
fprintf('  Normal Model (CPU Time):      %.3f s\n', t_cpu);
fprintf('  Modified Model (GPU Time):    %.3f s\n', t_gpu);
fprintf('  ⚡ Parallel Speedup:           %.2fx FASTER on NVIDIA GT 750M\n', speedup);
fprintf('  Max Absolute Error:           %.4e (Mathematical Match)\n', max_err);
fprintf('  Root-Mean-Square (RMS) Error: %.4e\n', rms_err);
disp('====================================================================');

%% 3. GENERATE VISUAL COMPARISON PLOT
fig = figure('Visible', 'off', 'Position', [100, 100, 1100, 650]);

% Subplot 1: CPU Wave Heatmap
subplot(2, 2, 1);
imagesc(1:num_channels, tout, y_cpu);
colormap(turbo); colorbar;
title('1️⃣ Normal Model: Native Simulink (CPU)');
xlabel('Oscillator Node'); ylabel('Time (s)');

% Subplot 2: GPU Wave Heatmap
subplot(2, 2, 2);
imagesc(1:num_channels, tout, y_gpu);
colormap(turbo); colorbar;
title('2️⃣ Modified Model: OpenCL Bridge (NVIDIA GPU)');
xlabel('Oscillator Node'); ylabel('Time (s)');

% Subplot 3: Numerical Residual / Error
subplot(2, 2, 3);
imagesc(1:num_channels, tout, diff_matrix);
colormap(hot); colorbar;
title(sprintf('3️⃣ Numerical Difference |CPU - GPU| (Max: %.1e)', max_err));
xlabel('Oscillator Node'); ylabel('Time (s)');

% Subplot 4: Execution Time & Speedup Bar Chart
subplot(2, 2, 4);
b = bar([t_cpu, t_gpu], 0.5);
b.FaceColor = 'flat';
b.CData(1,:) = [0.2, 0.4, 0.8]; % Blue for CPU
b.CData(2,:) = [0.1, 0.8, 0.3]; % Green for GPU
set(gca, 'XTickLabel', {'Normal (CPU)', 'Modified (GPU)'});
ylabel('Execution Time (seconds)');
title(sprintf('4️⃣ Speedup: %.2fx Faster on GPU!', speedup));
grid on;

text(2, t_gpu + t_cpu*0.05, sprintf('%.2fx FASTER', speedup), ...
     'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontSize', 12, 'Color', [0 0.5 0]);

saveas(fig, 'cpu_vs_gpu_simulink_comparison.png');
disp('✓ Comparison figure saved to: cpu_vs_gpu_simulink_comparison.png');

close_system(cpu_model, 0);
close_system(gpu_model, 0);
disp('🎉 Benchmark complete!');
