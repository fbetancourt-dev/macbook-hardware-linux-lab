% build_antiwindup_comparison_models.m
% Generates two GPU-accelerated Simulink models:
% 1. thermal_antiwindup_on_gpu.slx  (Discrete PI with Clamping Anti-Windup)
% 2. thermal_antiwindup_off_gpu.slx (Standard Discrete PI without Anti-Windup)

fprintf('====================================================\n');
fprintf('🏗️  Building Anti-Windup Comparison Models...\n');
fprintf('====================================================\n');

modes = {'thermal_antiwindup_on_gpu', 'thermal_antiwindup_off_gpu'};
aw_flags = {'on', 'off'};

for m_idx = 1:length(modes)
    mdl = modes{m_idx};
    aw_mode = aw_flags{m_idx};
    
    if bdIsLoaded(mdl)
        close_system(mdl, 0);
    end
    if exist([mdl '.slx'], 'file')
        delete([mdl '.slx']);
    end
    
    new_system(mdl);
    load_system(mdl);
    
    set_param(mdl, 'StopTime', '70.0');
    set_param(mdl, 'SolverType', 'Fixed-step');
    set_param(mdl, 'Solver', 'FixedStepDiscrete');
    set_param(mdl, 'FixedStep', '0.05');
    
    % 1. Plant S-Function Block (GPU OpenCL)
    plant_blk = [mdl '/Thermal_Plant_2D'];
    add_block('simulink/User-Defined Functions/S-Function', plant_blk);
    set_param(plant_blk, 'FunctionName', 'sfun_thermal_2d_opencl');
    set_param(plant_blk, 'Position', [480, 160, 610, 240]);
    
    % 2. Mux for 4 control inputs -> Plant
    mux_u = [mdl '/Mux_Control'];
    add_block('simulink/Signal Routing/Mux', mux_u);
    set_param(mux_u, 'Inputs', '4');
    set_param(mux_u, 'Position', [420, 150, 430, 250]);
    add_line(mdl, 'Mux_Control/1', 'Thermal_Plant_2D/1');
    
    % 3. Demux for 4 plant outputs -> Feedback
    demux_y = [mdl '/Demux_Sensors'];
    add_block('simulink/Signal Routing/Demux', demux_y);
    set_param(demux_y, 'Outputs', '4');
    set_param(demux_y, 'Position', [660, 150, 670, 250]);
    add_line(mdl, 'Thermal_Plant_2D/1', 'Demux_Sensors/1');
    
    % 4. Build 4 Control Channels
    mux_ref = [mdl '/Mux_Ref'];
    add_block('simulink/Signal Routing/Mux', mux_ref);
    set_param(mux_ref, 'Inputs', '4');
    set_param(mux_ref, 'Position', [200, 20, 210, 100]);
    
    % Mux for internal Integrator States logging
    mux_integ = [mdl '/Mux_Integ'];
    add_block('simulink/Signal Routing/Mux', mux_integ);
    set_param(mux_integ, 'Inputs', '4');
    set_param(mux_integ, 'Position', [315, 20, 325, 100]);
    
    y_offsets = [110, 190, 270, 350];
    
    % Zone 1 Reference: Pulse from 45 -> 110 C at t=8s, back to 45 C at t=24s
    c_base = [mdl '/Base_Z1'];
    add_block('simulink/Sources/Constant', c_base);
    set_param(c_base, 'Value', '45.0');
    set_param(c_base, 'SampleTime', '0.05');
    set_param(c_base, 'Position', [10, 50, 40, 70]);
    
    step_up = [mdl '/StepUp_Z1'];
    add_block('simulink/Sources/Step', step_up);
    set_param(step_up, 'Time', '8.0');
    set_param(step_up, 'Before', '0.0');
    set_param(step_up, 'After', '65.0');
    set_param(step_up, 'SampleTime', '0.05');
    set_param(step_up, 'Position', [10, 75, 40, 95]);
    
    step_down = [mdl '/StepDown_Z1'];
    add_block('simulink/Sources/Step', step_down);
    set_param(step_down, 'Time', '24.0');
    set_param(step_down, 'Before', '0.0');
    set_param(step_down, 'After', '-65.0');
    set_param(step_down, 'SampleTime', '0.05');
    set_param(step_down, 'Position', [10, 100, 40, 120]);
    
    sum_ref1 = [mdl '/Sum_Ref1'];
    add_block('simulink/Math Operations/Sum', sum_ref1);
    set_param(sum_ref1, 'Inputs', '+++');
    set_param(sum_ref1, 'Position', [60, 70, 80, 100]);
    add_line(mdl, 'Base_Z1/1', 'Sum_Ref1/1');
    add_line(mdl, 'StepUp_Z1/1', 'Sum_Ref1/2');
    add_line(mdl, 'StepDown_Z1/1', 'Sum_Ref1/3');
    add_line(mdl, 'Sum_Ref1/1', 'Mux_Ref/1');
    
    for z = 1:4
        y_pos = y_offsets(z);
        
        sum_e = [mdl sprintf('/Sum_e%d', z)];
        add_block('simulink/Math Operations/Sum', sum_e);
        set_param(sum_e, 'Inputs', '+-');
        set_param(sum_e, 'Position', [140, y_pos, 160, y_pos+25]);
        
        if z == 1
            add_line(mdl, 'Sum_Ref1/1', 'Sum_e1/1');
        else
            ref_z = [mdl sprintf('/Ref_Z%d', z)];
            add_block('simulink/Sources/Constant', ref_z);
            set_param(ref_z, 'Value', '45.0');
            set_param(ref_z, 'SampleTime', '0.05');
            set_param(ref_z, 'Position', [60, y_pos, 100, y_pos+25]);
            add_line(mdl, sprintf('Ref_Z%d/1', z), sprintf('Mux_Ref/%d', z));
            add_line(mdl, sprintf('Ref_Z%d/1', z), sprintf('Sum_e%d/1', z));
        end
        
        % Feedback from plant
        add_line(mdl, sprintf('Demux_Sensors/%d', z), sprintf('Sum_e%d/2', z));
        
        % Proportional Gain Kp = 2.5
        kp_blk = [mdl sprintf('/Kp_Z%d', z)];
        add_block('simulink/Math Operations/Gain', kp_blk);
        set_param(kp_blk, 'Gain', '2.5');
        set_param(kp_blk, 'Position', [190, y_pos-15, 230, y_pos+10]);
        add_line(mdl, sprintf('Sum_e%d/1', z), sprintf('Kp_Z%d/1', z));
        
        % Integral Gain Ki = 0.35
        ki_blk = [mdl sprintf('/Ki_Z%d', z)];
        add_block('simulink/Math Operations/Gain', ki_blk);
        set_param(ki_blk, 'Gain', '0.35');
        set_param(ki_blk, 'Position', [190, y_pos+20, 230, y_pos+45]);
        add_line(mdl, sprintf('Sum_e%d/1', z), sprintf('Ki_Z%d/1', z));
        
        % Discrete Integrator: Anti-Windup Clamping vs Free Integration
        integ_blk = [mdl sprintf('/Integ_Z%d', z)];
        add_block('simulink/Discrete/Discrete-Time Integrator', integ_blk);
        set_param(integ_blk, 'IntegratorMethod', 'Forward Euler');
        set_param(integ_blk, 'SampleTime', '0.05');
        set_param(integ_blk, 'LimitOutput', aw_mode);
        if strcmp(aw_mode, 'on')
            set_param(integ_blk, 'UpperSaturationLimit', '100.0');
            set_param(integ_blk, 'LowerSaturationLimit', '0.0');
        end
        set_param(integ_blk, 'Position', [255, y_pos+18, 295, y_pos+47]);
        add_line(mdl, sprintf('Ki_Z%d/1', z), sprintf('Integ_Z%d/1', z));
        
        % Connect to Mux_Integ for state logging
        add_line(mdl, sprintf('Integ_Z%d/1', z), sprintf('Mux_Integ/%d', z));
        
        % PI Sum
        pi_sum = [mdl sprintf('/PI_Sum%d', z)];
        add_block('simulink/Math Operations/Sum', pi_sum);
        set_param(pi_sum, 'Inputs', '++');
        set_param(pi_sum, 'Position', [320, y_pos, 340, y_pos+25]);
        add_line(mdl, sprintf('Kp_Z%d/1', z), sprintf('PI_Sum%d/1', z));
        add_line(mdl, sprintf('Integ_Z%d/1', z), sprintf('PI_Sum%d/2', z));
        
        % Hardware Power Actuator Saturation [0, 100%]
        sat_blk = [mdl sprintf('/Sat_Z%d', z)];
        add_block('simulink/Discontinuities/Saturation', sat_blk);
        set_param(sat_blk, 'UpperLimit', '100.0');
        set_param(sat_blk, 'LowerLimit', '0.0');
        set_param(sat_blk, 'Position', [360, y_pos, 385, y_pos+25]);
        add_line(mdl, sprintf('PI_Sum%d/1', z), sprintf('Sat_Z%d/1', z));
        
        add_line(mdl, sprintf('Sat_Z%d/1', z), sprintf('Mux_Control/%d', z));
    end
    
    % 5. Data Logging to Workspace
    log_T = [mdl '/ToWorkspace_T'];
    add_block('simulink/Sinks/To Workspace', log_T);
    set_param(log_T, 'VariableName', 'T_meas');
    set_param(log_T, 'SaveFormat', 'Array');
    set_param(log_T, 'SampleTime', '0.05');
    set_param(log_T, 'Position', [710, 185, 790, 215]);
    add_line(mdl, 'Thermal_Plant_2D/1', 'ToWorkspace_T/1');
    
    log_P = [mdl '/ToWorkspace_P'];
    add_block('simulink/Sinks/To Workspace', log_P);
    set_param(log_P, 'VariableName', 'P_ctrl');
    set_param(log_P, 'SaveFormat', 'Array');
    set_param(log_P, 'SampleTime', '0.05');
    set_param(log_P, 'Position', [480, 90, 560, 120]);
    add_line(mdl, 'Mux_Control/1', 'ToWorkspace_P/1');
    
    log_I = [mdl '/ToWorkspace_I'];
    add_block('simulink/Sinks/To Workspace', log_I);
    set_param(log_I, 'VariableName', 'I_state');
    set_param(log_I, 'SaveFormat', 'Array');
    set_param(log_I, 'SampleTime', '0.05');
    set_param(log_I, 'Position', [350, 45, 430, 75]);
    add_line(mdl, 'Mux_Integ/1', 'ToWorkspace_I/1');
    
    log_R = [mdl '/ToWorkspace_Ref'];
    add_block('simulink/Sinks/To Workspace', log_R);
    set_param(log_R, 'VariableName', 'T_ref');
    set_param(log_R, 'SaveFormat', 'Array');
    set_param(log_R, 'SampleTime', '0.05');
    set_param(log_R, 'Position', [230, 45, 300, 75]);
    add_line(mdl, 'Mux_Ref/1', 'ToWorkspace_Ref/1');
    
    save_system(mdl, [mdl '.slx']);
    close_system(mdl);
    fprintf('✓ Created model: %s.slx (Anti-Windup: %s)\n', mdl, upper(aw_mode));
end

fprintf('🎉 Anti-Windup comparison models generated successfully!\n');
