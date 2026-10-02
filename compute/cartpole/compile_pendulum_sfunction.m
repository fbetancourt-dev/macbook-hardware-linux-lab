% compile_pendulum_sfunction.m
% Compiles both GPU (OpenCL) and CPU (FP32) C-MEX S-Functions for Pendulum Ensemble

fprintf('====================================================\n');
fprintf('🛠️  Compiling Inverted Pendulum S-Functions...\n');
fprintf('====================================================\n');

% Compile GPU OpenCL MEX
fprintf('Compiling sfun_pendulum_ensemble_opencl.c (OpenCL / 384 CUDA Cores)... ');
try
    mex -O -lOpenCL sfun_pendulum_ensemble_opencl.c
    fprintf('SUCCESS [sfun_pendulum_ensemble_opencl.mexa64]\n');
catch ME
    fprintf('FAILED:\n%s\n', ME.message);
    rethrow(ME);
end

% Compile CPU FP32 MEX
fprintf('Compiling sfun_pendulum_ensemble_cpu.c (CPU Reference)... ');
try
    mex -O sfun_pendulum_ensemble_cpu.c
    fprintf('SUCCESS [sfun_pendulum_ensemble_cpu.mexa64]\n');
catch ME
    fprintf('FAILED:\n%s\n', ME.message);
    rethrow(ME);
end

fprintf('🎉 All Pendulum S-Functions compiled successfully!\n');
