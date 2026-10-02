% run_pendulum_benchmark.m
% Executes 1,024-Pendulum Parallel Ensemble Simulation:
% Host CPU vs NVIDIA GeForce GT 750M (384 Kepler CUDA Cores via OpenCL)

fprintf('====================================================================\n');
fprintf('🎪 INVERTED PENDULUM ENSEMBLE BENCHMARK (1,024 SYSTEMS IN PARALLEL)\n');
fprintf('   Dynamics: Nonlinear Coupled Cart-Pole ODEs (RK4, 5 substeps)\n');
fprintf('   Control: Energy Swing-Up + Full-State LQR Stabilization\n');
fprintf('   Hardware: Core i7 CPU vs NVIDIA GeForce GT 750M (384 CUDA Cores)\n');
fprintf('====================================================================\n\n');

% 1. Compile S-Functions
run('compile_pendulum_sfunction.m');

% 2. Build Simulink Models
run('build_pendulum_models.m');

% 3. Run CPU Simulation (12s, 1201 steps x 5 substeps x 1024 pendulums)
fprintf('--------------------------------------------------------------------\n');
fprintf('⏱️  Running CPU Simulation (1,024 pendulums x 12.0s = 12,288 simulated seconds)...\n');
tic;
sim_cpu_out = sim('pendulum_cpu_ensemble_model', 'StopTime', '12.0');
t_cpu_sim = toc;
fprintf('   CPU Simulation completed in: %.3f seconds\n', t_cpu_sim);

t_axis = sim_cpu_out.tout;
ref_cpu = sim_cpu_out.ref_states;
stats_cpu = sim_cpu_out.ensemble_stats;

% 4. Run GPU Simulation on 384 CUDA Cores
fprintf('⚡ Running GPU Simulation on 384 CUDA Cores (NVIDIA GT 750M)...\n');
tic;
sim_gpu_out = sim('pendulum_gpu_ensemble_model', 'StopTime', '12.0');
t_gpu_sim = toc;
fprintf('   GPU Simulation completed in: %.3f seconds\n', t_gpu_sim);

ref_gpu = sim_gpu_out.ref_states;
stats_gpu = sim_gpu_out.ensemble_stats;

% 5. Benchmark Analytics
speedup = t_cpu_sim / t_gpu_sim;
throughput_gpu = (1024 * 12.0) / t_gpu_sim; % simulated pendulum-seconds per real second
final_stab_rate = stats_gpu(end, 1);

fprintf('\n====================================================================\n');
fprintf('📊 BENCHMARK METRICS & ENSEMBLE TELEMETRY\n');
fprintf('====================================================================\n');
fprintf('  Parallel Systems:         1,024 Cart-Pole pendulums\n');
fprintf('  Total Simulation Time:    12.0 seconds (1,201 steps, 6,005 RK4 steps)\n');
fprintf('  Total ODE Evaluations:    24.6 Million state derivative calculations\n');
fprintf('  CPU Execution Time:       %.3f s\n', t_cpu_sim);
fprintf('  GPU Execution Time:       %.3f s\n', t_gpu_sim);
fprintf('  ⚡ Parallel Speedup:       %.2fx FASTER on 384 CUDA Cores\n', speedup);
fprintf('  GPU Throughput:           %.1f pendulum-seconds / second\n', throughput_gpu);
fprintf('  Ensemble Stability Rate:  %.1f %% of all 1,024 systems stabilized!\n', final_stab_rate);
fprintf('====================================================================\n\n');

% 6. Read Final Ensemble States for Basin of Attraction Map
basin_grid = zeros(32, 32);
fp = fopen('pendulum_ensemble_final.bin', 'rb');
if fp ~= -1
    % 1024 states x 4 floats [x, x_dot, theta, theta_dot]
    raw_data = fread(fp, 1024 * 4, 'single');
    fclose(fp);
    states_all = reshape(raw_data, [4, 1024])';
    
    for i = 1:1024
        a_idx = mod(i-1, 32) + 1;
        v_idx = floor((i-1)/32) + 1;
        final_th = abs(states_all(i, 3));
        final_x = abs(states_all(i, 1));
        if final_th < 0.15 && final_x < 1.0
            basin_grid(v_idx, a_idx) = 1.0; % Stable (LQR locked)
        elseif final_th < 0.5
            basin_grid(v_idx, a_idx) = 0.5; % Near upright
        else
            basin_grid(v_idx, a_idx) = 0.0; % Unstable / fallen
        end
    end
end

% 7. Generate 4-Panel Publication Dashboard
fig = figure('Name', 'Inverted Pendulum GPU Benchmark', ...
             'Color', [0.12 0.12 0.14], 'Position', [100 100 1350 900], 'Visible', 'off');

% Panel 1: Reference Cart Swing-Up & Stabilization (Theta and X)
subplot(2, 2, 1);
yyaxis left;
plot(t_axis, ref_gpu(:, 3) * 180 / pi, '-', 'Color', [0.20 0.85 0.40], 'LineWidth', 2.2);
ylabel('Pole Angle θ (degrees)', 'Color', [0.20 0.85 0.40], 'FontSize', 11, 'FontWeight', 'bold');
ylim([-200 200]);
yline(0, '--', 'Color', [0.5 0.5 0.5]);

yyaxis right;
plot(t_axis, ref_gpu(:, 1), '-', 'Color', [0.20 0.70 0.95], 'LineWidth', 2.0);
ylabel('Cart Position x (m)', 'Color', [0.20 0.70 0.95], 'FontSize', 11, 'FontWeight', 'bold');
ylim([-1.5 1.5]);

grid on;
set(gca, 'Color', [0.08 0.08 0.10], 'XColor', [0.8 0.8 0.8], 'GridColor', [0.25 0.25 0.30]);
xlabel('Time (seconds)', 'Color', [0.9 0.9 0.9], 'FontSize', 11, 'FontWeight', 'bold');
title('1. Reference Cart: Swing-Up (180° -> 0°) & LQR Centering', 'Color', [1 1 1], 'FontSize', 12, 'FontWeight', 'bold');

% Panel 2: Phase Portrait (Theta vs Theta_dot)
subplot(2, 2, 2);
plot(ref_gpu(:, 3) * 180 / pi, ref_gpu(:, 4) * 180 / pi, '-', 'Color', [0.95 0.75 0.20], 'LineWidth', 2.0);
hold on;
plot(ref_gpu(1, 3)*180/pi, ref_gpu(1, 4)*180/pi, 'go', 'MarkerSize', 8, 'MarkerFaceColor', 'g', 'DisplayName', 'Start (180°)');
plot(ref_gpu(end, 3)*180/pi, ref_gpu(end, 4)*180/pi, 'ro', 'MarkerSize', 8, 'MarkerFaceColor', 'r', 'DisplayName', 'Equilibrium (0,0)');
hold off;
grid on;
set(gca, 'Color', [0.08 0.08 0.10], 'XColor', [0.8 0.8 0.8], 'YColor', [0.8 0.8 0.8], 'GridColor', [0.25 0.25 0.30]);
xlabel('Pole Angle θ (degrees)', 'Color', [0.9 0.9 0.9], 'FontSize', 11, 'FontWeight', 'bold');
ylabel('Angular Velocity dθ/dt (deg/s)', 'Color', [0.9 0.9 0.9], 'FontSize', 11, 'FontWeight', 'bold');
title('2. Phase Portrait: Energy Pump Limit Cycle -> LQR Attractor', 'Color', [1 1 1], 'FontSize', 12, 'FontWeight', 'bold');
legend('Location', 'northeast', 'TextColor', [0.9 0.9 0.9], 'Color', [0.15 0.15 0.18]);

% Panel 3: Basin of Attraction Heatmap (1,024 Systems)
subplot(2, 2, 3);
imagesc(linspace(-180, 180, 32), linspace(-4, 4, 32), basin_grid);
colormap(gca, [0.15 0.15 0.20; 0.95 0.50 0.20; 0.15 0.85 0.40]); % Black=Unstable, Orange=Near, Green=Stable
cb = colorbar;
cb.Ticks = [0, 0.5, 1.0];
cb.TickLabels = {'Unstable', 'Transitional', 'Stabilized (LQR)'};
cb.Color = [0.9 0.9 0.9];
set(gca, 'Color', [0.08 0.08 0.10], 'XColor', [0.8 0.8 0.8], 'YColor', [0.8 0.8 0.8], 'YDir', 'normal');
xlabel('Initial Pole Angle θ₀ (degrees)', 'Color', [0.9 0.9 0.9], 'FontSize', 11, 'FontWeight', 'bold');
ylabel('Initial Angular Velocity ω₀ (rad/s)', 'Color', [0.9 0.9 0.9], 'FontSize', 11, 'FontWeight', 'bold');
title(sprintf('3. Basin of Attraction (1,024 Parallel Initial Conditions)'), 'Color', [1 1 1], 'FontSize', 12, 'FontWeight', 'bold');

% Panel 4: Performance Speedup & Throughput
subplot(2, 2, 4);
b = bar([t_cpu_sim, t_gpu_sim]);
b.FaceColor = 'flat';
b.CData(1, :) = [0.2 0.5 0.8];
b.CData(2, :) = [0.1 0.85 0.4];
set(gca, 'XTickLabel', {'Host CPU (FP32)', '384 CUDA Cores (GPU)'}, ...
         'Color', [0.08 0.08 0.10], 'XColor', [0.8 0.8 0.8], 'YColor', [0.8 0.8 0.8], ...
         'GridColor', [0.25 0.25 0.30]);
grid on;
ylabel('Execution Time (seconds)', 'Color', [0.9 0.9 0.9], 'FontSize', 11, 'FontWeight', 'bold');
title(sprintf('4. GPU Speedup: %.2fx Faster on 384 Kepler Cores', speedup), ...
      'Color', [1 1 1], 'FontSize', 12, 'FontWeight', 'bold');
text(2, t_gpu_sim + 0.08*t_cpu_sim, ...
     sprintf('%.2fx FASTER\n(%.0f pend-sec/s)', speedup, throughput_gpu), ...
     'Color', [0.2 1.0 0.4], 'FontSize', 11, 'FontWeight', 'bold', 'HorizontalAlignment', 'center');

out_fig = 'pendulum_gpu_benchmark_results.png';
saveas(fig, out_fig);
fprintf('✓ Comparison figure saved to: %s\n', out_fig);
fprintf('🎉 Inverted Pendulum GPU benchmark completed successfully!\n');
