%% Compilation script for Simulink OpenCL S-Function Bridge
% Targets NVIDIA Kepler GT 750M via Mesa Rusticl (OpenCL 3.0)

disp('=== Compiling Simulink OpenCL S-Function Bridge ===');
current_dir = fileparts(mfilename('fullpath'));
cd(current_dir);

try
    mex -O CFLAGS='$CFLAGS -O3 -Wall' -lOpenCL sfun_opencl_parallel.c
    disp('✅ S-Function successfully compiled: sfun_opencl_parallel.mexa64');
catch ME
    fprintf('❌ Compilation failed:\n%s\n', ME.message);
end
