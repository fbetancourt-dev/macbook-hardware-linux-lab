%% Automated Simulink Model Creation & Simulation with OpenCL GPU Acceleration
% Targets 384 Kepler CUDA Cores on NVIDIA GT 750M via Mesa Rusticl

clear; clc;
disp('====================================================================');
disp('🚀 SIMULINK GPU ACCELERATION DEMO (OpenCL S-Function Bridge)');
disp('====================================================================');

model_name = 'gpu_oscillator_array_model';
num_channels = 512; % 512 coupled oscillators simulated simultaneously on GPU
damping = 0.05;
sim_time = 2.0; % 2.0 seconds of physical time

% Close model if already open
if bdIsLoaded(model_name)
    close_system(model_name, 0);
end

disp('1. Creating programmatic Simulink model...');
new_system(model_name);
set_param(model_name, 'Solver', 'FixedStepDiscrete', 'FixedStep', '0.005', 'StopTime', num2str(sim_time));

% Add Input Chirp/Sine Source: Vector of 512 channels
add_block('simulink/Sources/Constant', [model_name '/Excitation']);
% Pulse in the middle 20 channels, 0 elsewhere
init_u = zeros(num_channels, 1);
init_u(round(num_channels/2)-10 : round(num_channels/2)+10) = 5.0; 
set_param([model_name '/Excitation'], 'Value', 'excitation_vector', 'Position', [100, 100, 180, 140]);

% Add S-Function Block (Our OpenCL Bridge!)
add_block('simulink/User-Defined Functions/S-Function', [model_name '/GPU_Plant_SFunction']);
set_param([model_name '/GPU_Plant_SFunction'], ...
          'FunctionName', 'sfun_opencl_parallel', ...
          'Parameters', sprintf('%d, %f', num_channels, damping), ...
          'Position', [260, 85, 420, 155]);

% Add To Workspace Sink
add_block('simulink/Sinks/To Workspace', [model_name '/ToWorkspace']);
set_param([model_name '/ToWorkspace'], ...
          'VariableName', 'sim_out_states', ...
          'SaveFormat', 'Array', ...
          'Position', [500, 100, 580, 140]);

% Connect lines
add_line(model_name, 'Excitation/1', 'GPU_Plant_SFunction/1');
add_line(model_name, 'GPU_Plant_SFunction/1', 'ToWorkspace/1');

% Save model
save_system(model_name);
disp(['✓ Simulink model saved: ' model_name '.slx']);

% Assign variable to base workspace
assignin('base', 'excitation_vector', init_u);

% Run Simulation!
disp(['2. Running Simulink simulation on GPU (' num2str(num_channels) ' channels x ' num2str(sim_time) 's)...']);
tic;
sim_res = sim(model_name);
elapsed_sim = toc;

disp(sprintf('✓ Simulink simulation completed in %.3f seconds!', elapsed_sim));

% Extract results
states = sim_res.get('sim_out_states');
tout   = sim_res.get('tout');

fprintf('   - Total Simulation Steps: %d\n', length(tout));
fprintf('   - State Matrix Size:      %d time-steps x %d channels\n', size(states, 1), size(states, 2));

% Save visual figure
fig = figure('Visible', 'off', 'Position', [100, 100, 900, 500]);
subplot(1, 2, 1);
imagesc(1:num_channels, tout, states);
colormap(turbo);
colorbar;
xlabel('Coupled Channel / Oscillator Index');
ylabel('Time (s)');
title(sprintf('GPU Wave Propagation (%d Channels)', num_channels));

subplot(1, 2, 2);
plot(tout, states(:, round(num_channels/2)), 'LineWidth', 1.5);
hold on;
plot(tout, states(:, round(num_channels/2) + 50), 'LineWidth', 1.2);
grid on;
xlabel('Time (s)');
ylabel('State Position y(t)');
legend('Center Node (Driven)', 'Node +50 (Propagated)');
title('Simulink S-Function Time Response');

saveas(fig, 'simulink_gpu_simulation_result.png');
disp('✓ Simulation visualization saved to: simulink_gpu_simulation_result.png');

% Clean up model
close_system(model_name, 0);
disp('====================================================================');
disp('🎉 SUCCESS: Simulink OpenCL Bridge is operational!');
disp('====================================================================');
