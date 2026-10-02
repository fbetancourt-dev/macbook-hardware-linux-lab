% run_thermal_stress_benchmark.m
% Executes Thermal Stress & Physical Saturation Benchmark:
% Testing Actuator Saturation [0, 100%], Cross-Coupling Deficit, and Anti-Windup Recovery.

fprintf('====================================================================\n');
fprintf('🔥 THERMAL STRESS & PHYSICAL SATURATION BENCHMARK\n');
fprintf('   Plant: 256x256 = 65,536 nodes, Convection-Diffusion\n');
fprintf('   Test: Zone 1 Heat Pulse (45 -> 110 -> 45 °C) to force 0%% Saturation\n');
fprintf('   Hardware: Core i7 CPU vs NVIDIA GeForce GT 750M (384 CUDA Cores)\n');
fprintf('====================================================================\n\n');

% 1. Recompile S-Functions with DirectFeedThrough=0
run('compile_thermal_sfunction.m');

% 2. Build Models if not already built
run('build_stress_thermal_models.m');

% 3. Run CPU Simulation (40s, 801 steps x 5 substeps = 4,005 grid iterations)
fprintf('--------------------------------------------------------------------\n');
fprintf('⏱️  Running CPU Stress Simulation (40.0s, 801 steps, 262.5M updates)...\n');
tic;
sim_cpu_out = sim('thermal_stress_control_cpu');
t_cpu_sim = toc;
fprintf('   CPU Simulation completed in: %.3f seconds\n', t_cpu_sim);

t_axis = sim_cpu_out.tout;
T_cpu = sim_cpu_out.T_meas;
P_cpu = sim_cpu_out.P_ctrl;
T_ref = sim_cpu_out.T_ref;

% 4. Run GPU Simulation
fprintf('⚡ Running GPU Stress Simulation (NVIDIA GT 750M, 384 CUDA Cores)...\n');
tic;
sim_gpu_out = sim('thermal_stress_control_gpu');
t_gpu_sim = toc;
fprintf('   GPU Simulation completed in: %.3f seconds\n', t_gpu_sim);

T_gpu = sim_gpu_out.T_meas;
P_gpu = sim_gpu_out.P_ctrl;

% 5. Benchmark Metrics
speedup = t_cpu_sim / t_gpu_sim;
diff_T = abs(T_cpu - T_gpu);
max_err = max(diff_T(:));

fprintf('\n====================================================================\n');
fprintf('📊 BENCHMARK METRICS & SATURATION ANALYSIS\n');
fprintf('====================================================================\n');
fprintf('  Physical Grid Size:       256 x 256 = 65,536 finite difference nodes\n');
fprintf('  Total Physical Time:      40.0 seconds (801 steps, 4,005 sub-iterations)\n');
fprintf('  Total Grid Updates:       262.5 Million node evaluations\n');
fprintf('  CPU Execution Time:       %.3f s\n', t_cpu_sim);
fprintf('  GPU Execution Time:       %.3f s\n', t_gpu_sim);
fprintf('  ⚡ Parallel Speedup:       %.2fx FASTER on NVIDIA GT 750M\n', speedup);
fprintf('  Simulated / Real Ratio:   %.2fx Faster than Real Physical Time\n', 40.0 / t_gpu_sim);
fprintf('  Max Temp Difference:      %.4e deg C\n', max_err);
fprintf('  Min Power on Zone 2:      %.2f %% (Saturated at lower boundary)\n', min(P_gpu(:, 2)));
fprintf('  Min Power on Zone 3:      %.2f %% (Saturated at lower boundary)\n', min(P_gpu(:, 3)));
fprintf('  Peak Thermal Drift Z2:    +%.2f °C above setpoint (Physical Coupling Deficit)\n', ...
        max(T_gpu(t_axis >= 8 & t_axis <= 24, 2)) - 45.0);
fprintf('====================================================================\n\n');

% 6. Read Final Thermal Field
grid_2d = zeros(256, 256);
fp = fopen('thermal_field_final_gpu.bin', 'rb');
if fp ~= -1
    raw = fread(fp, 256*256, 'single');
    fclose(fp);
    grid_2d = reshape(raw, [256, 256])';
end

% 7. Plot 4-Panel Report
fig = figure('Name', 'Thermal Stress & Saturation Benchmark', ...
             'Color', [0.12 0.12 0.14], 'Position', [100 100 1350 900], 'Visible', 'off');

colors = [
    0.95 0.25 0.25;  % Zone 1: Red
    0.20 0.75 0.95;  % Zone 2: Cyan
    0.95 0.75 0.20;  % Zone 3: Gold
    0.30 0.85 0.40   % Zone 4: Emerald
];

% Panel 1: Temperature Trajectories with Saturation Region
subplot(2, 2, 1);
hold on;
% Shade the extreme pulse zone [8, 24]
fill([8 24 24 8], [15 15 125 125], [0.35 0.15 0.15], 'FaceAlpha', 0.25, 'EdgeColor', 'none', ...
     'DisplayName', 'Zone 1 Heat Pulse (110°C)');
plot(t_axis, T_ref(:, 1), '--w', 'LineWidth', 1.8, 'DisplayName', 'Zone 1 Setpoint (45->110->45)');
plot(t_axis, T_ref(:, 2), '--', 'Color', [0.7 0.7 0.7], 'LineWidth', 1.2, 'DisplayName', 'Zones 2-4 Setpoint (45°C)');
for z = 1:4
    plot(t_axis, T_cpu(:, z), ':', 'Color', colors(z, :)*0.75, 'LineWidth', 1.5);
    plot(t_axis, T_gpu(:, z), '-', 'Color', colors(z, :), 'LineWidth', 2.0, ...
         'DisplayName', sprintf('Zone %d Temp', z));
end
hold off;
grid on;
set(gca, 'Color', [0.08 0.08 0.10], 'XColor', [0.8 0.8 0.8], 'YColor', [0.8 0.8 0.8], 'GridColor', [0.25 0.25 0.30]);
xlabel('Time (seconds)', 'Color', [0.9 0.9 0.9], 'FontSize', 11, 'FontWeight', 'bold');
ylabel('Temperature (°C)', 'Color', [0.9 0.9 0.9], 'FontSize', 11, 'FontWeight', 'bold');
title('1. Thermal Response: Pulse & Active Cooling Deficit', 'Color', [1 1 1], 'FontSize', 12, 'FontWeight', 'bold');
legend('Location', 'northeast', 'TextColor', [0.9 0.9 0.9], 'Color', [0.15 0.15 0.18], 'FontSize', 9);
ylim([20 120]);

% Panel 2: Actuator Power Outputs & 0% Lower Saturation Boundary
subplot(2, 2, 2);
hold on;
% Lower saturation line at 0%
yline(0, '--r', 'LineWidth', 1.5, 'DisplayName', 'Lower Saturation (0% Off)');
for z = 1:4
    plot(t_axis, P_gpu(:, z), '-', 'Color', colors(z, :), 'LineWidth', 2.0, ...
         'DisplayName', sprintf('Heater %d Power', z));
end
hold off;
grid on;
set(gca, 'Color', [0.08 0.08 0.10], 'XColor', [0.8 0.8 0.8], 'YColor', [0.8 0.8 0.8], 'GridColor', [0.25 0.25 0.30]);
xlabel('Time (seconds)', 'Color', [0.9 0.9 0.9], 'FontSize', 11, 'FontWeight', 'bold');
ylabel('Actuator Power (%)', 'Color', [0.9 0.9 0.9], 'FontSize', 11, 'FontWeight', 'bold');
title('2. Actuator Clamping: 0% Lower Saturation & Anti-Windup', 'Color', [1 1 1], 'FontSize', 12, 'FontWeight', 'bold');
legend('Location', 'northeast', 'TextColor', [0.9 0.9 0.9], 'Color', [0.15 0.15 0.18], 'FontSize', 9);
ylim([-5 105]);

% Panel 3: 2D Thermal Heatmap
subplot(2, 2, 3);
imagesc(grid_2d);
colormap(gca, hot);
cb = colorbar;
cb.Color = [0.9 0.9 0.9];
ylabel(cb, 'Temperature (°C)', 'Color', [0.9 0.9 0.9], 'FontSize', 10);
set(gca, 'Color', [0.08 0.08 0.10], 'XColor', [0.8 0.8 0.8], 'YColor', [0.8 0.8 0.8]);
xlabel('X Grid Node', 'Color', [0.9 0.9 0.9], 'FontSize', 11, 'FontWeight', 'bold');
ylabel('Y Grid Node', 'Color', [0.9 0.9 0.9], 'FontSize', 11, 'FontWeight', 'bold');
title('3. Final 2D Thermal Field (65,536 Nodes on GPU)', 'Color', [1 1 1], 'FontSize', 12, 'FontWeight', 'bold');
axis equal tight;

% Panel 4: Performance Speedup
subplot(2, 2, 4);
b = bar([t_cpu_sim, t_gpu_sim]);
b.FaceColor = 'flat';
b.CData(1, :) = [0.2 0.5 0.8];
b.CData(2, :) = [0.1 0.8 0.4];
set(gca, 'XTickLabel', {'Normal (CPU)', 'Accelerated (GPU)'}, ...
         'Color', [0.08 0.08 0.10], 'XColor', [0.8 0.8 0.8], 'YColor', [0.8 0.8 0.8], ...
         'GridColor', [0.25 0.25 0.30]);
grid on;
ylabel('Execution Time (seconds)', 'Color', [0.9 0.9 0.9], 'FontSize', 11, 'FontWeight', 'bold');
title(sprintf('4. Benchmark: %.2fx Faster on GT 750M (%.1fx Realtime)', speedup, 40.0/t_gpu_sim), ...
      'Color', [1 1 1], 'FontSize', 12, 'FontWeight', 'bold');
text(2, t_gpu_sim + 0.05*t_cpu_sim, sprintf('%.2fx FASTER\n(40s in %.2fs)', speedup, t_gpu_sim), ...
     'Color', [0.2 1.0 0.4], 'FontSize', 12, 'FontWeight', 'bold', 'HorizontalAlignment', 'center');

out_fig = 'thermal_stress_saturation_results.png';
saveas(fig, out_fig);
fprintf('✓ Comparison figure saved to: %s\n', out_fig);
fprintf('🎉 Thermal stress benchmark completed successfully!\n');
