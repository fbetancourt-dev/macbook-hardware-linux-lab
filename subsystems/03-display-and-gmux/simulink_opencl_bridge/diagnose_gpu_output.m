clear; clc;
disp('--- Diagnosing CPU and GPU models ---');

u_input = zeros(1024, 1);
u_input(500:520) = 10.0;
assignin('base', 'u_input', u_input);

load_system('simulink_cpu_model');
load_system('simulink_gpu_model');

disp('1. Running CPU model...');
rc = sim('simulink_cpu_model');
yc = rc.get('out_cpu');
fprintf('CPU out size: [%s], hasNaN: %d, min: %f, max: %f\n', ...
    num2str(size(yc)), any(isnan(yc(:))), min(yc(:)), max(yc(:)));

disp('2. Running GPU model...');
rg = sim('simulink_gpu_model');
yg = rg.get('out_gpu');
fprintf('GPU out size: [%s], hasNaN: %d, min: %f, max: %f\n', ...
    num2str(size(yg)), any(isnan(yg(:))), min(yg(:)), max(yg(:)));

disp('Sample first 5 nodes at step 10:');
disp('CPU:'); disp(yc(10, 500:505));
disp('GPU:'); disp(yg(10, 500:505));

close_system('simulink_cpu_model', 0);
close_system('simulink_gpu_model', 0);
disp('--- Done Diagnosis ---');
