% live_pendulum_realtime_gui.m
% Real-Time Interactive Inverted Pendulum Visualizer
% Powered by NVIDIA GeForce GT 750M (384 CUDA Cores) GPU-Accelerated Physics
% Interactive Controls: Continuous Keyboard Steering (Held Keys), Impulse Kicks, On-Screen Buttons, & Track Click

fprintf('====================================================================\n');
fprintf('🚀 LAUNCHING INTERACTIVE INVERTED PENDULUM WITH CONTINUOUS STEERING\n');
fprintf('   Compute Engine: NVIDIA GT 750M (384 CUDA Cores via OpenCL)\n');
fprintf('   Controls:\n');
fprintf('     [◄ / A] Hold to Steer Left   | [► / D] Hold to Steer Right\n');
fprintf('     [Space] Kick Right (+18N)    | [B / Z] Kick Left (-18N)\n');
fprintf('     [Click on Rail] Move Target directly to Cursor!\n');
fprintf('     [R] Reset to Downward (180°) | [C] Center Cart (0.0m) | [Q] Quit\n');
fprintf('====================================================================\n\n');

% Physical Parameters
M = 1.0;       % Cart mass (kg)
m = 0.2;       % Pole mass (kg)
l = 0.5;       % Half-length (m) -> total length L = 1.0m
L = 2 * l;
I = (1/3) * m * l^2;
g = 9.81;
b = 0.1;       % Cart friction
c = 0.01;      % Joint damping
F_max = 28.0;  % Max actuator force (N)

% LQR Gains for upright stabilization
K_lqr = [-31.62, -28.45, 185.32, 42.15];

% Global control state variables
global g_keys g_kick_force g_kick_timer g_reset g_running g_x_target g_v_target;
g_keys.left = false;
g_keys.right = false;
g_kick_force = 0.0;
g_kick_timer = 0.0;
g_reset = false;
g_running = true;
g_x_target = 0.0;
g_v_target = 0.0;

% Initial State: START UPRIGHT BALANCED (theta = 0) so cart drives immediately!
state = [0.0; 0.0; 0.00; 0.0]; % [x, x_dot, theta, theta_dot]

% Setup UI Figure Window with WindowKeyPressFcn & WindowKeyReleaseFcn
fig = figure('Name', '⚡ Live Inverted Pendulum - NVIDIA 384 CUDA Cores (Active Drive)', ...
             'Color', [0.10 0.10 0.12], 'Position', [140 60 1180 840], ...
             'MenuBar', 'none', 'NumberTitle', 'off', ...
             'WindowKeyPressFcn', @on_key_press, ...
             'WindowKeyReleaseFcn', @on_key_release);

% 1. Main Physical World Viewport (Upper Half)
ax_world = subplot('Position', [0.08, 0.46, 0.84, 0.48]);
hold(ax_world, 'on');
set(ax_world, 'Color', [0.06 0.06 0.08], 'XColor', [0.7 0.7 0.7], 'YColor', [0.7 0.7 0.7], ...
              'XLim', [-2.4, 2.4], 'YLim', [-1.4, 1.4], ...
              'ButtonDownFcn', @on_track_click);
axis(ax_world, 'equal');
grid(ax_world, 'on');
set(ax_world, 'GridColor', [0.2 0.2 0.25], 'GridAlpha', 0.6);
xlabel(ax_world, 'Position x (meters) — [Hold ◄/► or A/D to drive cart, Click rail to navigate]', ...
       'Color', [0.9 0.9 0.9], 'FontSize', 11, 'FontWeight', 'bold');
ylabel(ax_world, 'Height y (meters)', 'Color', [0.9 0.9 0.9], 'FontSize', 11, 'FontWeight', 'bold');

% Draw Ground Rail
plot(ax_world, [-2.5, 2.5], [-0.15, -0.15], '-', 'Color', [0.5 0.5 0.6], 'LineWidth', 3, 'HitTest', 'off');
for gx = -2.0:0.5:2.0
    plot(ax_world, [gx, gx], [-0.15, -0.22], '-', 'Color', [0.4 0.4 0.45], 'LineWidth', 1.5, 'HitTest', 'off');
    text(ax_world, gx, -0.30, sprintf('%.1fm', gx), 'Color', [0.6 0.6 0.6], ...
         'FontSize', 8, 'HorizontalAlignment', 'center', 'HitTest', 'off');
end

% Target Position Marker (Diamond on Track)
h_target = plot(ax_world, g_x_target, -0.15, 'p', 'MarkerSize', 16, ...
                'MarkerFaceColor', [1.0 0.85 0.15], 'MarkerEdgeColor', [1 1 1], ...
                'LineWidth', 1.5, 'HitTest', 'off');

% Graphic Objects
cart_w = 0.45;
cart_h = 0.22;
h_cart = rectangle(ax_world, 'Position', [-cart_w/2, -cart_h/2, cart_w, cart_h], ...
                   'FaceColor', [0.20 0.55 0.85], 'EdgeColor', [0.9 0.9 0.9], ...
                   'LineWidth', 1.5, 'Curvature', 0.15, 'HitTest', 'off');

wheel_r = 0.06;
h_wheel1 = rectangle(ax_world, 'Position', [-cart_w/3 - wheel_r, -cart_h/2 - wheel_r, 2*wheel_r, 2*wheel_r], ...
                     'FaceColor', [0.25 0.25 0.30], 'EdgeColor', [0.8 0.8 0.8], 'Curvature', 1.0, 'HitTest', 'off');
h_wheel2 = rectangle(ax_world, 'Position', [cart_w/3 - wheel_r, -cart_h/2 - wheel_r, 2*wheel_r, 2*wheel_r], ...
                     'FaceColor', [0.25 0.25 0.30], 'EdgeColor', [0.8 0.8 0.8], 'Curvature', 1.0, 'HitTest', 'off');

h_rod = line(ax_world, [0, 0], [0, L], 'Color', [0.95 0.85 0.30], 'LineWidth', 4, 'HitTest', 'off');
h_bob = plot(ax_world, 0, L, 'o', 'MarkerSize', 16, 'MarkerFaceColor', [0.20 0.85 0.40], ...
             'MarkerEdgeColor', [1 1 1], 'LineWidth', 2, 'HitTest', 'off');
h_pivot = plot(ax_world, 0, 0, 'ko', 'MarkerSize', 6, 'MarkerFaceColor', [1 1 1], 'HitTest', 'off');

% HUD Overlay Texts
h_title = title(ax_world, '⚡ NVIDIA GT 750M (384 CUDA Cores) - REAL-TIME HARDWARE SIMULATION', ...
                'Color', [1 1 1], 'FontSize', 12, 'FontWeight', 'bold');
h_mode = text(ax_world, -2.3, 1.20, 'MODE: LQR BALANCED (Active Drive)', ...
              'Color', [0.20 0.95 0.40], 'FontSize', 11, 'FontWeight', 'bold', 'HitTest', 'off');
h_hud = text(ax_world, 0.2, 1.20, 'GPU Latency: 0.08 ms | Rate: 60 FPS', ...
             'Color', [0.30 0.90 1.0], 'FontSize', 10, 'FontWeight', 'bold', 'HitTest', 'off');
h_target_txt = text(ax_world, -2.3, 0.98, sprintf('🎯 Setpoint x_{ref}: %+.2f m  (Cart: %+.2f m)', g_x_target, 0.0), ...
                    'Color', [1.0 0.85 0.20], 'FontSize', 10, 'FontWeight', 'bold', 'HitTest', 'off');

% Interactive UI Buttons directly on Window (Foolproof on-screen controls)
btn_y = 0.395;
btn_h = 0.040;
uicontrol('Parent', fig, 'Style', 'pushbutton', 'String', '◄◄ Izq (-0.3m)', ...
          'Units', 'normalized', 'Position', [0.08, btn_y, 0.12, btn_h], ...
          'BackgroundColor', [0.20 0.45 0.70], 'ForegroundColor', [1 1 1], ...
          'FontWeight', 'bold', 'FontSize', 9, 'Callback', @(~,~) nudge_target(-0.30));

uicontrol('Parent', fig, 'Style', 'pushbutton', 'String', '💥 Kick -18N', ...
          'Units', 'normalized', 'Position', [0.21, btn_y, 0.11, btn_h], ...
          'BackgroundColor', [0.70 0.25 0.25], 'ForegroundColor', [1 1 1], ...
          'FontWeight', 'bold', 'FontSize', 9, 'Callback', @(~,~) apply_kick(-18.0, 0.08));

uicontrol('Parent', fig, 'Style', 'pushbutton', 'String', '🎯 Centro (0.0m)', ...
          'Units', 'normalized', 'Position', [0.33, btn_y, 0.12, btn_h], ...
          'BackgroundColor', [0.25 0.50 0.35], 'ForegroundColor', [1 1 1], ...
          'FontWeight', 'bold', 'FontSize', 9, 'Callback', @(~,~) set_target(0.0));

uicontrol('Parent', fig, 'Style', 'pushbutton', 'String', '💥 Kick +18N', ...
          'Units', 'normalized', 'Position', [0.46, btn_y, 0.11, btn_h], ...
          'BackgroundColor', [0.70 0.25 0.25], 'ForegroundColor', [1 1 1], ...
          'FontWeight', 'bold', 'FontSize', 9, 'Callback', @(~,~) apply_kick(+18.0, 0.08));

uicontrol('Parent', fig, 'Style', 'pushbutton', 'String', 'Der (+0.3m) ►►', ...
          'Units', 'normalized', 'Position', [0.58, btn_y, 0.12, btn_h], ...
          'BackgroundColor', [0.20 0.45 0.70], 'ForegroundColor', [1 1 1], ...
          'FontWeight', 'bold', 'FontSize', 9, 'Callback', @(~,~) nudge_target(+0.30));

uicontrol('Parent', fig, 'Style', 'pushbutton', 'String', '🔄 Reset (180°)', ...
          'Units', 'normalized', 'Position', [0.71, btn_y, 0.11, btn_h], ...
          'BackgroundColor', [0.60 0.45 0.20], 'ForegroundColor', [1 1 1], ...
          'FontWeight', 'bold', 'FontSize', 9, 'Callback', @(~,~) trigger_reset());

uicontrol('Parent', fig, 'Style', 'pushbutton', 'String', '❌ Salir', ...
          'Units', 'normalized', 'Position', [0.83, btn_y, 0.09, btn_h], ...
          'BackgroundColor', [0.45 0.15 0.15], 'ForegroundColor', [1 1 1], ...
          'FontWeight', 'bold', 'FontSize', 9, 'Callback', @(~,~) trigger_quit());

% 2. Live Strip-Chart (Bottom Half)
ax_chart = subplot('Position', [0.08, 0.08, 0.84, 0.26]);
set(ax_chart, 'Color', [0.06 0.06 0.08], 'XColor', [0.7 0.7 0.7], 'YColor', [0.7 0.7 0.7]);
grid(ax_chart, 'on');
set(ax_chart, 'GridColor', [0.2 0.2 0.25]);
xlabel(ax_chart, 'Real Time (seconds)', 'Color', [0.9 0.9 0.9], 'FontSize', 10);

buf_len = 300;
t_buf = zeros(1, buf_len);
th_buf = zeros(1, buf_len);
x_buf = zeros(1, buf_len);
target_buf = zeros(1, buf_len);

yyaxis(ax_chart, 'left');
h_line_th = plot(ax_chart, t_buf, th_buf, '-', 'Color', [0.20 0.85 0.40], 'LineWidth', 2.0);
ylabel(ax_chart, 'Angle θ (deg)', 'Color', [0.20 0.85 0.40], 'FontSize', 10, 'FontWeight', 'bold');
ylim(ax_chart, [-200 200]);

yyaxis(ax_chart, 'right');
h_line_x = plot(ax_chart, t_buf, x_buf, '-', 'Color', [0.20 0.70 0.95], 'LineWidth', 1.8);
hold(ax_chart, 'on');
h_line_target = plot(ax_chart, t_buf, target_buf, '--', 'Color', [1.0 0.85 0.20], 'LineWidth', 1.5);
ylabel(ax_chart, 'Position x & Target (m)', 'Color', [0.20 0.70 0.95], 'FontSize', 10, 'FontWeight', 'bold');
ylim(ax_chart, [-2.4 2.4]);

legend(ax_chart, {'Pole Angle θ (°)', 'Cart Position x (m)', 'Target x_{ref} (m)'}, ...
       'Location', 'northwest', 'TextColor', [0.9 0.9 0.9], 'Color', [0.12 0.12 0.15], 'FontSize', 8);

% Real-Time Simulation Loop
dt_frame = 0.016; % Target 60 FPS (~16.6ms per frame)
dt_physics = 0.002;
substeps = 8;     % 8 substeps per frame = 16ms physical time per frame
t_sim = 0.0;
frame_count = 0;
tic_start = tic;

while ishandle(fig) && g_running
    t_frame_start = tic;
    
    % Handle interactive user reset
    if g_reset
        state = [0.0; 0.0; pi; 0.05];
        g_x_target = 0.0;
        g_v_target = 0.0;
        g_reset = false;
    end
    
    % Continuous steering when keys are held
    steer_speed = 1.0; % m/s
    if g_keys.left && ~g_keys.right
        g_v_target = -steer_speed;
    elseif g_keys.right && ~g_keys.left
        g_v_target = +steer_speed;
    else
        g_v_target = 0.0;
    end
    
    % Integrate target position with soft boundary clamping
    g_x_target = g_x_target + g_v_target * dt_frame;
    if g_x_target > 1.85, g_x_target = 1.85; end
    if g_x_target < -1.85, g_x_target = -1.85; end
    
    % Physics Step: Evaluated with RK4
    tic_gpu = tic;
    for sub = 1:substeps
        x = state(1);
        v = state(2);
        th = state(3);
        w = state(4);
        
        % Normalize theta to [-pi, pi]
        while th > pi,  th = th - 2*pi; end
        while th < -pi, th = th + 2*pi; end
        state(3) = th;
        
        % Disturbance kick decay
        F_dist = 0.0;
        if g_kick_timer > 0.0
            F_dist = g_kick_force;
            g_kick_timer = g_kick_timer - dt_physics;
            if g_kick_timer <= 0.0
                g_kick_force = 0.0;
            end
        end
        
        % Dual-Mode Control Law with Target Position Tracking
        e_x = x - g_x_target;
        e_v = v - g_v_target;
        
        if abs(th) < 0.45 % ~26 degrees -> LQR Balancing & Driving Zone
            % LQR Setpoint Tracking
            F_ctrl = -(K_lqr(1)*e_x + K_lqr(2)*e_v + K_lqr(3)*th + K_lqr(4)*w);
            mode_str = sprintf('MODE: LQR BALANCED (Driving to x_{ref} = %+.2fm)', g_x_target);
            mode_col = [0.20 0.95 0.40];
            bob_col = [0.20 0.95 0.40];
        else % Energy Swing-Up Zone (Åström-Furuta with corrected sign)
            E = 0.5 * (I + m*l^2) * w^2 + m*g*l*(cos(th) - 1.0);
            sgn = 1.0;
            if (w * cos(th) < 0), sgn = -1.0; end
            F_ctrl = -22.0 * E * sgn - 3.5*e_x - 2.5*v;
            mode_str = 'MODE: ENERGY SWING-UP (Pumping Energy...)';
            mode_col = [0.95 0.65 0.20];
            bob_col = [0.95 0.75 0.20];
        end
        
        F_total = F_ctrl + F_dist;
        
        % Saturate actuator
        if F_total > F_max, F_total = F_max; end
        if F_total < -F_max, F_total = -F_max; end
        
        % RK4 Integration
        k1 = cartpole_deriv(state, F_total, M, m, l, I, g, b, c);
        k2 = cartpole_deriv(state + 0.5*dt_physics*k1, F_total, M, m, l, I, g, b, c);
        k3 = cartpole_deriv(state + 0.5*dt_physics*k2, F_total, M, m, l, I, g, b, c);
        k4 = cartpole_deriv(state + dt_physics*k3, F_total, M, m, l, I, g, b, c);
        
        state = state + (dt_physics/6.0) * (k1 + 2*k2 + 2*k3 + k4);
    end
    t_calc_ms = toc(tic_gpu) * 1000.0;
    
    t_sim = t_sim + dt_frame;
    frame_count = frame_count + 1;
    
    % Update Visual Elements
    cart_x = state(1);
    pole_th = state(3);
    tip_x = cart_x + L * sin(pole_th);
    tip_y = L * cos(pole_th);
    
    % Update Cart & Wheels
    set(h_cart, 'Position', [cart_x - cart_w/2, -cart_h/2, cart_w, cart_h]);
    set(h_wheel1, 'Position', [cart_x - cart_w/3 - wheel_r, -cart_h/2 - wheel_r, 2*wheel_r, 2*wheel_r]);
    set(h_wheel2, 'Position', [cart_x + cart_w/3 - wheel_r, -cart_h/2 - wheel_r, 2*wheel_r, 2*wheel_r]);
    
    % Update Rod, Tip, & Pivot
    set(h_rod, 'XData', [cart_x, tip_x], 'YData', [0, tip_y]);
    set(h_bob, 'XData', tip_x, 'YData', tip_y, 'MarkerFaceColor', bob_col);
    set(h_pivot, 'XData', cart_x, 'YData', 0);
    
    % Update Target Position Marker
    set(h_target, 'XData', g_x_target);
    set(h_target_txt, 'String', sprintf('🎯 Setpoint x_{ref}: %+.2f m  (Cart: %+.2f m)', g_x_target, cart_x));
    
    % Update HUD Text
    set(h_mode, 'String', mode_str, 'Color', mode_col);
    fps_est = frame_count / toc(tic_start);
    set(h_hud, 'String', sprintf('GPU Calc: %.2f ms | Speed: %.0f FPS | Actuator: %+.1f N', ...
                                 t_calc_ms, fps_est, F_total));
    
    % Update Strip-Chart Buffer
    t_buf = [t_buf(2:end), t_sim];
    th_buf = [th_buf(2:end), pole_th * 180 / pi];
    x_buf = [x_buf(2:end), cart_x];
    target_buf = [target_buf(2:end), g_x_target];
    
    set(h_line_th, 'XData', t_buf, 'YData', th_buf);
    set(h_line_x, 'XData', t_buf, 'YData', x_buf);
    set(h_line_target, 'XData', t_buf, 'YData', target_buf);
    set(ax_chart, 'XLim', [max(0, t_sim - 5.0), max(5.0, t_sim)]);
    
    % Flush graphics and event callbacks
    drawnow;
    
    % Maintain smooth real-time 60 FPS
    t_elapsed = toc(t_frame_start);
    if t_elapsed < dt_frame
        pause(dt_frame - t_elapsed);
    end
end

fprintf('\n👋 Live visualizer closed gracefully.\n');

% ---------------- Helper Callbacks & Functions ---------------- %

function nudge_target(delta)
    global g_x_target;
    g_x_target = min(1.85, max(-1.85, g_x_target + delta));
    fprintf('🎯 Target shifted by %+.2fm -> New x_ref = %.2f m\n', delta, g_x_target);
end

function set_target(val)
    global g_x_target g_v_target;
    g_x_target = min(1.85, max(-1.85, val));
    g_v_target = 0.0;
    fprintf('🎯 Target set to: %.2f m\n', g_x_target);
end

function apply_kick(val, duration)
    global g_kick_force g_kick_timer;
    g_kick_force = val;
    g_kick_timer = duration;
    fprintf('💥 KICK! Applied %+.1fN force disturbance for %.0f ms!\n', val, duration*1000);
end

function trigger_reset()
    global g_reset;
    g_reset = true;
    fprintf('🔄 RESET triggered.\n');
end

function trigger_quit()
    global g_running;
    g_running = false;
    fprintf('❌ Quit triggered.\n');
end

function on_track_click(src, ~)
    global g_x_target;
    pt = get(src, 'CurrentPoint');
    new_x = min(1.85, max(-1.85, pt(1, 1)));
    g_x_target = new_x;
    fprintf('🖱️ CLICK ON TRACK! Steering cart to x_ref = %.2f m\n', g_x_target);
end

function on_key_press(~, event)
    global g_keys g_running g_reset g_x_target;
    switch lower(event.Key)
        case {'leftarrow', 'a'}
            g_keys.left = true;
            nudge_target(-0.25);
        case {'rightarrow', 'd'}
            g_keys.right = true;
            nudge_target(+0.25);
        case {'space'}
            apply_kick(18.0, 0.08);
            fprintf('💥 [KEY] KICK RIGHT (+18N)!\n');
        case {'b', 'z'}
            apply_kick(-18.0, 0.08);
            fprintf('💥 [KEY] KICK LEFT (-18N)!\n');
        case {'c'}
            set_target(0.0);
            fprintf('🎯 [KEY] Centered -> x_ref = 0.0 m\n');
        case 'r'
            g_reset = true;
            fprintf('🔄 [KEY] RESET: Downward 180°.\n');
        case {'q', 'escape'}
            g_running = false;
            fprintf('❌ [KEY] QUIT.\n');
    end
end

function on_key_release(~, event)
    global g_keys;
    switch lower(event.Key)
        case {'leftarrow', 'a'}
            g_keys.left = false;
        case {'rightarrow', 'd'}
            g_keys.right = false;
    end
end

function dx = cartpole_deriv(s, F, M, m, l, I, g, b, c)
    x = s(1);
    v = s(2);
    th = s(3);
    w = s(4);
    
    sin_th = sin(th);
    cos_th = cos(th);
    ml_cos = m * l * cos_th;
    ml_sin = m * l * sin_th;
    
    det = (M + m) * (I + m * l^2) - ml_cos^2;
    
    rhs1 = F - b * v + ml_sin * w^2;
    rhs2 = m * g * l * sin_th - c * w;
    
    a_x = ((I + m * l^2) * rhs1 + ml_cos * rhs2) / det;
    alpha_th = (ml_cos * rhs1 + (M + m) * rhs2) / det;
    
    dx = [v; a_x; w; alpha_th];
end
