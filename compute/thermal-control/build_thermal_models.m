% build_thermal_models.m
% Programmatically generates both CPU and GPU Simulink models for
% the 2D Multizone Thermal Control Benchmark.

fprintf('====================================================\n');
fprintf('🏗️  Building Closed-Loop Simulink Thermal Models...\n');
fprintf('====================================================\n');

models = {'thermal_multizone_control_cpu', 'thermal_multizone_control_gpu'};
sfuncs = {'sfun_thermal_2d_cpu', 'sfun_thermal_2d_opencl'};

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
    open_system(mdl);
    
    set_param(mdl, 'StopTime', '25.0');
    set_param(mdl, 'SolverType', 'Fixed-step');
    set_param(mdl, 'Solver', 'FixedStepDiscrete');
    set_param(mdl, 'FixedStep', '0.05');
    
    % 1. Plant S-Function Block
    plant_blk = [mdl '/Thermal_Plant_2D'];
    add_block('simulink/User-Defined Functions/S-Function', plant_blk);
    set_param(plant_blk, 'FunctionName', sfunc_name);
    set_param(plant_blk, 'Position', [450, 160, 580, 240]);
    
    % 2. Mux for 4 control inputs -> Plant
    mux_u = [mdl '/Mux_Control'];
    add_block('simulink/Signal Routing/Mux', mux_u);
    set_param(mux_u, 'Inputs', '4');
    set_param(mux_u, 'Position', [390, 150, 400, 250]);
    add_line(mdl, 'Mux_Control/1', 'Thermal_Plant_2D/1');
    
    % 3. Demux for 4 plant outputs -> Feedback
    demux_y = [mdl '/Demux_Sensors'];
    add_block('simulink/Signal Routing/Demux', demux_y);
    set_param(demux_y, 'Outputs', '4');
    set_param(demux_y, 'Position', [630, 150, 640, 250]);
    add_line(mdl, 'Thermal_Plant_2D/1', 'Demux_Sensors/1');
    
    % 4. Build 4 Control Channels (Ref, Sum, PI Controller)
    % Zone 1 has a step from 45C to 70C at t = 10s. Zones 2, 3, 4 stay at 45C.
    mux_ref = [mdl '/Mux_Ref'];
    add_block('simulink/Signal Routing/Mux', mux_ref);
    set_param(mux_ref, 'Inputs', '4');
    set_param(mux_ref, 'Position', [170, 30, 180, 110]);
    
    y_offsets = [100, 180, 260, 340];
    
    for z = 1:4
        y_pos = y_offsets(z);
        
        % Setpoint source
        ref_blk = [mdl sprintf('/Ref_Z%d', z)];
        if z == 1
            add_block('simulink/Sources/Step', ref_blk);
            set_param(ref_blk, 'Time', '10.0');
            set_param(ref_blk, 'Before', '45.0');
            set_param(ref_blk, 'After', '70.0');
            set_param(ref_blk, 'SampleTime', '0.05');
        else
            add_block('simulink/Sources/Constant', ref_blk);
            set_param(ref_blk, 'Value', '45.0');
            set_param(ref_blk, 'SampleTime', '0.05');
        end
        set_param(ref_blk, 'Position', [30, y_pos, 70, y_pos+25]);
        
        % Connect to Mux_Ref for logging
        add_line(mdl, sprintf('Ref_Z%d/1', z), sprintf('Mux_Ref/%d', z));
        
        % Error sum junction: e = Ref - Y
        sum_blk = [mdl sprintf('/Sum_e%d', z)];
        add_block('simulink/Math Operations/Sum', sum_blk);
        set_param(sum_blk, 'Inputs', '+-');
        set_param(sum_blk, 'Position', [110, y_pos, 130, y_pos+25]);
        add_line(mdl, sprintf('Ref_Z%d/1', z), sprintf('Sum_e%d/1', z));
        
        % Feedback line from Demux_Sensors
        add_line(mdl, sprintf('Demux_Sensors/%d', z), sprintf('Sum_e%d/2', z));
        
        % PI Controller implementation:
        % Kp Gain
        kp_blk = [mdl sprintf('/Kp_Z%d', z)];
        add_block('simulink/Math Operations/Gain', kp_blk);
        set_param(kp_blk, 'Gain', '2.5');
        set_param(kp_blk, 'Position', [160, y_pos-15, 200, y_pos+10]);
        add_line(mdl, sprintf('Sum_e%d/1', z), sprintf('Kp_Z%d/1', z));
        
        % Ki Gain
        ki_blk = [mdl sprintf('/Ki_Z%d', z)];
        add_block('simulink/Math Operations/Gain', ki_blk);
        set_param(ki_blk, 'Gain', '0.35');
        set_param(ki_blk, 'Position', [160, y_pos+20, 200, y_pos+45]);
        add_line(mdl, sprintf('Sum_e%d/1', z), sprintf('Ki_Z%d/1', z));
        
        % Discrete Integrator with Anti-Windup Clamping [0, 100]
        integ_blk = [mdl sprintf('/Integ_Z%d', z)];
        add_block('simulink/Discrete/Discrete-Time Integrator', integ_blk);
        set_param(integ_blk, 'IntegratorMethod', 'Forward Euler');
        set_param(integ_blk, 'SampleTime', '0.05');
        set_param(integ_blk, 'LimitOutput', 'on');
        set_param(integ_blk, 'UpperSaturationLimit', '100.0');
        set_param(integ_blk, 'LowerSaturationLimit', '0.0');
        set_param(integ_blk, 'Position', [225, y_pos+18, 265, y_pos+47]);
        add_line(mdl, sprintf('Ki_Z%d/1', z), sprintf('Integ_Z%d/1', z));
        
        % PI Sum (P + I)
        pi_sum = [mdl sprintf('/PI_Sum%d', z)];
        add_block('simulink/Math Operations/Sum', pi_sum);
        set_param(pi_sum, 'Inputs', '++');
        set_param(pi_sum, 'Position', [290, y_pos, 310, y_pos+25]);
        add_line(mdl, sprintf('Kp_Z%d/1', z), sprintf('PI_Sum%d/1', z));
        add_line(mdl, sprintf('Integ_Z%d/1', z), sprintf('PI_Sum%d/2', z));
        
        % Final Saturation [0, 100]
        sat_blk = [mdl sprintf('/Sat_Z%d', z)];
        add_block('simulink/Discontinuities/Saturation', sat_blk);
        set_param(sat_blk, 'UpperLimit', '100.0');
        set_param(sat_blk, 'LowerLimit', '0.0');
        set_param(sat_blk, 'Position', [330, y_pos, 355, y_pos+25]);
        add_line(mdl, sprintf('PI_Sum%d/1', z), sprintf('Sat_Z%d/1', z));
        
        % Connect to Mux_Control
        add_line(mdl, sprintf('Sat_Z%d/1', z), sprintf('Mux_Control/%d', z));
    end
    
    % 5. Data Logging to Workspace
    % Temperatures log
    log_T = [mdl '/ToWorkspace_T'];
    add_block('simulink/Sinks/To Workspace', log_T);
    set_param(log_T, 'VariableName', 'T_meas');
    set_param(log_T, 'SaveFormat', 'Array');
    set_param(log_T, 'SampleTime', '0.05');
    set_param(log_T, 'Position', [680, 185, 760, 215]);
    add_line(mdl, 'Thermal_Plant_2D/1', 'ToWorkspace_T/1');
    
    % Actuator power log
    log_P = [mdl '/ToWorkspace_P'];
    add_block('simulink/Sinks/To Workspace', log_P);
    set_param(log_P, 'VariableName', 'P_ctrl');
    set_param(log_P, 'SaveFormat', 'Array');
    set_param(log_P, 'SampleTime', '0.05');
    set_param(log_P, 'Position', [450, 90, 530, 120]);
    add_line(mdl, 'Mux_Control/1', 'ToWorkspace_P/1');
    
    % Setpoint log
    log_R = [mdl '/ToWorkspace_Ref'];
    add_block('simulink/Sinks/To Workspace', log_R);
    set_param(log_R, 'VariableName', 'T_ref');
    set_param(log_R, 'SaveFormat', 'Array');
    set_param(log_R, 'SampleTime', '0.05');
    set_param(log_R, 'Position', [220, 55, 300, 85]);
    add_line(mdl, 'Mux_Ref/1', 'ToWorkspace_Ref/1');
    
    % Save model
    save_system(mdl, [mdl '.slx']);
    close_system(mdl);
    fprintf('✓ Created model: %s.slx (S-Function: %s)\n', mdl, sfunc_name);
end

fprintf('🎉 Both closed-loop models generated successfully!\n');
