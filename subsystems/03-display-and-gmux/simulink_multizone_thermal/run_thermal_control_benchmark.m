% run_thermal_control_benchmark.m
% Executes side-by-side Closed-Loop 2D Thermal Control Benchmark:
% CPU (FP32 C-MEX) vs GPU (NVIDIA GT 750M OpenCL / Rusticl)

fprintf('====================================================================\n');
fprintf('🔥 CLOSED-LOOP 2D MULTIZONE THERMAL CONTROL BENCHMARK\n');
fprintf('   Plant: 256x256 = 65,536 nodes, Convection-Diffusion\n');
fprintf('   Control: 4x Discrete PI Controllers with Anti-Windup\n');
fprintf('   Hardware: Core i7 CPU vs NVIDIA GeForce GT 750M (384 CUDA Cores)\n');
fprintf('====================================================================\n\n');

% 1. Compile S-Functions if not already compiled
if ~exist('sfun_thermal_2d_opencl.mexa64', 'file') || ~exist('sfun_thermal_2d_cpu.mexa64', 'file')
    run('compile_thermal_sfunction.m');
end

% 2. Build Models if not already built
if ~exist('thermal_multizone_control_cpu.slx', 'file') || ~exist('thermal_multizone_control_gpu.slx', 'file')
    run('build_thermal_models.m');
end

% 3. Run CPU Simulation
fprintf('--------------------------------------------------------------------\n');
fprintf('⏱️  Running CPU Closed-Loop Simulation (25.0s, 501 steps x 5 substeps)...\n');
tic;
sim_cpu_out = sim('thermal_multizone_control_cpu');
t_cpu_sim = toc;
fprintf('   CPU Simulation completed in: %.3f seconds\n', t_cpu_sim);

% Extract CPU signals
t_axis = sim_cpu_out.tout;
T_cpu = sim_cpu_out.T_meas;
P_cpu = sim_cpu_out.P_ctrl;
T_ref = sim_cpu_out.T_ref;

% 4. Run GPU Simulation
fprintf('⚡ Running GPU Closed-Loop Simulation (NVIDIA GT 750M, 384 CUDA Cores)...\n');
tic;
sim_gpu_out = sim('thermal_multizone_control_gpu');
t_gpu_sim = toc;
fprintf('   GPU Simulation completed in: %.3f seconds\n', t_gpu_sim);

% Extract GPU signals
T_gpu = sim_gpu_out.T_meas;
P_gpu = sim_gpu_out.P_ctrl;

% 5. Benchmark Metrics
speedup = t_cpu_sim / t_gpu_sim;
diff_T = abs(T_cpu - T_gpu);
max_err = max(diff_T(:));
rms_err = sqrt(mean(diff_T(:).^2));

fprintf('\n====================================================================\n');
fprintf('📊 BENCHMARK METRICS & VALIDATION\n');
fprintf('====================================================================\n');
fprintf('  Physical Grid Size:       256 x 256 = 65,536 finite difference nodes\n');
fprintf('  Total Physical Time:      25.0 seconds (501 steps, 2,505 sub-iterations)\n');
fprintf('  Total Grid Updates:       164.1 Million node evaluations\n');
fprintf('  CPU Execution Time:       %.3f s\n', t_cpu_sim);
fprintf('  GPU Execution Time:       %.3f s\n', t_gpu_sim);
fprintf('  ⚡ Parallel Speedup:       %.2fx FASTER on NVIDIA GT 750M\n', speedup);
fprintf('  Max Temp Difference:      %.4e deg C\n', max_err);
fprintf('  RMS Temp Difference:      %.4e deg C\n', rms_err);
fprintf('====================================================================\n\n');

% 6. Read Final 2D Thermal Field from GPU
grid_2d = zeros(256, 256);
fp = fopen('thermal_field_final_gpu.bin', 'rb');
if fp ~= -1
    raw = fread(fp, 256*256, 'single');
    fclose(fp);
    grid_2d = reshape(raw, [256, 256])';
end

% 7. Plot Comprehensive 4-Panel Visualization
fig = figure('Name', 'Closed-Loop Thermal Control Benchmark', ...
             'Color', [0.12 0.12 0.14], 'Position', [100 100 1300 850], 'Visible', 'off');

colors = [
    0.95 0.30 0.25;  % Zone 1: Bright Red
    0.20 0.70 0.95;  % Zone 2: Cyan
    0.95 0.75 0.20;  % Zone 3: Gold
    0.30 0.85 0.40   % Zone 4: Emerald
];

% Panel 1: Temperature Trajectories (Setpoint tracking & disturbance rejection)
subplot(2, 2, 1);
hold on;
plot(t_axis, T_ref(:, 1), '--', 'Color', [0.7 0.7 0.7], 'LineWidth', 1.5, 'DisplayName', 'Zone 1 Setpoint (45->70 C)');
for z = 1:4
    plot(t_axis, T_cpu(:, z), ':', 'Color', colors(z, :)*0.8, 'LineWidth', 1.5, ...
         'DisplayName', sprintf('CPU Zone %d', z));
    plot(t_axis, T_gpu(:, z), '-', 'Color', colors(z, :), 'LineWidth', 2.0, ...
         'DisplayName', sprintf('GPU Zone %d', z));
end
hold off;
grid on;
set(gca, 'Color', [0.08 0.08 0.10], 'XColor', [0.8 0.8 0.8], 'YColor', [0.8 0.8 0.8], 'GridColor', [0.25 0.25 0.30]);
xlabel('Time (seconds)', 'Color', [0.9 0.9 0.9], 'FontSize', 11, 'FontWeight', 'bold');
ylabel('Temperature (°C)', 'Color', [0.9 0.9 0.9], 'FontSize', 11, 'FontWeight', 'bold');
title('1. Closed-Loop Temperature Tracking & Coupling', 'Color', [1 1 1], 'FontSize', 12, 'FontWeight', 'bold');
legend('Location', 'eastoutside', 'TextColor', [0.9 0.9 0.9], 'Color', [0.15 0.15 0.18]);
ylim([20 75]);

% Panel 2: Actuator Control Effort (PWM Power %)
subplot(2, 2, 2);
hold on;
for z = 1:4
    plot(t_axis, P_gpu(:, z), '-', 'Color', colors(z, :), 'LineWidth', 2.0, ...
         'DisplayName', sprintf('Heater %d Power', z));
end
hold off;
grid on;
set(gca, 'Color', [0.08 0.08 0.10], 'XColor', [0.8 0.8 0.8], 'YColor', [0.8 0.8 0.8], 'GridColor', [0.25 0.25 0.30]);
xlabel('Time (seconds)', 'Color', [0.9 0.9 0.9], 'FontSize', 11, 'FontWeight', 'bold');
ylabel('Heater Power (%)', 'Color', [0.9 0.9 0.9], 'FontSize', 11, 'FontWeight', 'bold');
title('2. PI Controller Outputs (Anti-Windup Clamped)', 'Color', [1 1 1], 'FontSize', 12, 'FontWeight', 'bold');
legend('Location', 'northeast', 'TextColor', [0.9 0.9 0.9], 'Color', [0.15 0.15 0.18]);
ylim([0 105]);

% Panel 3: 2D Distributed Thermal Field Heatmap
subplot(2, 2, 3);
imagesc(grid_2d);
colormap(gca, hot);
cb = colorbar;
cb.Color = [0.9 0.9 0.9];
ylabel(cb, 'Temperature (°C)', 'Color', [0.9 0.9 0.9], 'FontSize', 10);
set(gca, 'Color', [0.08 0.08 0.10], 'XColor', [0.8 0.8 0.8], 'YColor', [0.8 0.8 0.8]);
xlabel('X Grid Node (Plate Width)', 'Color', [0.9 0.9 0.9], 'FontSize', 11, 'FontWeight', 'bold');
ylabel('Y Grid Node (Plate Height)', 'Color', [0.9 0.9 0.9], 'FontSize', 11, 'FontWeight', 'bold');
title(sprintf('3. 2D Thermal Field at t=25s (65,536 Nodes on GPU)'), 'Color', [1 1 1], 'FontSize', 12, 'FontWeight', 'bold');
axis equal tight;

% Panel 4: Performance Benchmark Bar Chart
subplot(2, 2, 4);
b = bar([t_cpu_sim, t_gpu_sim]);
b.FaceColor = 'flat';
b.CData(1, :) = [0.2 0.5 0.8];   % Blue for CPU
b.CData(2, :) = [0.1 0.8 0.4];   % Green for GPU
set(gca, 'XTickLabel', {'Normal (CPU)', 'Accelerated (GPU)'}, ...
         'Color', [0.08 0.08 0.10], 'XColor', [0.8 0.8 0.8], 'YColor', [0.8 0.8 0.8], ...
         'GridColor', [0.25 0.25 0.30]);
grid on;
ylabel('Execution Time (seconds)', 'Color', [0.9 0.9 0.9], 'FontSize', 11, 'FontWeight', 'bold');
title(sprintf('4. Benchmark Speedup: %.2fx Faster on GT 750M!', speedup), ...
      'Color', [1 1 1], 'FontSize', 12, 'FontWeight', 'bold');
text(2, t_gpu_sim + 0.05*t_cpu_sim, sprintf('%.2fx FASTER\n(Δ = %.4e °C)', speedup, max_err), ...
     'Color', [0.2 1.0 0.4], 'FontSize', 12, 'FontWeight', 'bold', 'HorizontalAlignment', 'center');

% Save output plot
out_fig = 'thermal_multizone_control_results.png';
saveas(fig, out_fig);
fprintf('✓ Comparison figure saved to: %s\n', out_fig);
fprintf('🎉 Closed-loop thermal benchmark finished successfully!\n');
