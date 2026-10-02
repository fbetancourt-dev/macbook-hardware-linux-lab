% build_pendulum_models.m
% Generates Simulink models for 1,024-system Inverted Pendulum ensemble simulation.

fprintf('====================================================\n');
fprintf('🏗️  Building Inverted Pendulum Simulink Models...\n');
fprintf('====================================================\n');

bdclose('all');

models = {'pendulum_gpu_ensemble_model', 'pendulum_cpu_ensemble_model'};
sfuncs = {'sfun_pendulum_ensemble_opencl', 'sfun_pendulum_ensemble_cpu'};

for m_idx = 1:length(models)
    mdl = models{m_idx};
    sfunc_name = sfuncs{m_idx};
    
    if bdIsLoaded(mdl)
        close_system(mdl, 0);
    end
    if exist([mdl '.slx'], 'file')
        delete([mdl '.slx']);
    end
    
    new_system(mdl);
    load_system(mdl);
    
    set_param(mdl, 'StopTime', '12.0');
    set_param(mdl, 'SolverType', 'Fixed-step');
    set_param(mdl, 'Solver', 'FixedStepDiscrete');
    set_param(mdl, 'FixedStep', '0.01');
    
    % 1. Plant S-Function Block
    plant_blk = [mdl '/Pendulum_Ensemble'];
    add_block('simulink/User-Defined Functions/S-Function', plant_blk);
    set_param(plant_blk, 'FunctionName', sfunc_name);
    set_param(plant_blk, 'Position', [280, 100, 420, 180]);
    
    % 2. Disturbance Force Input Source
    dist_blk = [mdl '/Disturbance_Force'];
    add_block('simulink/Sources/Constant', dist_blk);
    set_param(dist_blk, 'Value', '0.0');
    set_param(dist_blk, 'SampleTime', '0.01');
    set_param(dist_blk, 'Position', [120, 130, 160, 150]);
    add_line(mdl, 'Disturbance_Force/1', 'Pendulum_Ensemble/1');
    
    % 3. Log Reference Cart States [x, x_dot, theta, theta_dot, F]
    log_ref = [mdl '/ToWorkspace_RefStates'];
    add_block('simulink/Sinks/To Workspace', log_ref);
    set_param(log_ref, 'VariableName', 'ref_states');
    set_param(log_ref, 'SaveFormat', 'Array');
    set_param(log_ref, 'SampleTime', '0.01');
    set_param(log_ref, 'Position', [500, 85, 590, 115]);
    add_line(mdl, 'Pendulum_Ensemble/1', 'ToWorkspace_RefStates/1');
    
    % 4. Log Ensemble Statistics [% stabilized, mean_err, max_err]
    log_stats = [mdl '/ToWorkspace_Stats'];
    add_block('simulink/Sinks/To Workspace', log_stats);
    set_param(log_stats, 'VariableName', 'ensemble_stats');
    set_param(log_stats, 'SaveFormat', 'Array');
    set_param(log_stats, 'SampleTime', '0.01');
    set_param(log_stats, 'Position', [500, 145, 590, 175]);
    add_line(mdl, 'Pendulum_Ensemble/2', 'ToWorkspace_Stats/1');
    
    save_system(mdl, [mdl '.slx']);
    close_system(mdl);
    fprintf('✓ Created model: %s.slx (Plant: %s)\n', mdl, sfunc_name);
end

fprintf('🎉 Inverted Pendulum models generated successfully!\n');
