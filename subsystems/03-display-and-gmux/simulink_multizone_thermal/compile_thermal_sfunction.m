% compile_thermal_sfunction.m
% Compiles both GPU (OpenCL) and CPU (FP32) C-MEX S-Functions for Simulink

fprintf('====================================================\n');
fprintf('🛠️  Compiling Thermal 2D Plant C-MEX S-Functions...\n');
fprintf('====================================================\n');

% Compile GPU OpenCL MEX
fprintf('Compiling sfun_thermal_2d_opencl.c (OpenCL / Rusticl)... ');
try
    mex -O -lOpenCL sfun_thermal_2d_opencl.c
    fprintf('SUCCESS [sfun_thermal_2d_opencl.mexa64]\n');
catch ME
    fprintf('FAILED:\n%s\n', ME.message);
    rethrow(ME);
end

% Compile CPU FP32 MEX
fprintf('Compiling sfun_thermal_2d_cpu.c (CPU FP32 Reference)... ');
try
    mex -O sfun_thermal_2d_cpu.c
    fprintf('SUCCESS [sfun_thermal_2d_cpu.mexa64]\n');
catch ME
    fprintf('FAILED:\n%s\n', ME.message);
    rethrow(ME);
end

fprintf('🎉 All thermal S-Functions compiled successfully!\n');
