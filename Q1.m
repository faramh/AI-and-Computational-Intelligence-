% % Create figure
% fig = figure('Position', [100, 100, 800, 600]);
% 
% % Create sliders for W1, W2, and T
% uicontrol('Style', 'slider', 'Min', -5, 'Max', 5, 'Value', 1, ...
%     'Position', [50, 500, 120, 20], 'Callback', @update_plot, 'Tag', 'W1');
% uicontrol('Style', 'slider', 'Min', -5, 'Max', 5, 'Value', 1, ...
%     'Position', [50, 450, 120, 20], 'Callback', @update_plot, 'Tag', 'W2');
% uicontrol('Style', 'slider', 'Min', -5, 'Max', 5, 'Value', 0, ...
%     'Position', [50, 400, 120, 20], 'Callback', @update_plot, 'Tag', 'T');
% 
% % Create axes for the plot
% ax = axes('Parent', fig, 'Position', [0.3, 0.1, 0.6, 0.8]);
% 
% % Set initial values for W1, W2, and T
% setappdata(fig, 'W1', 1);
% setappdata(fig, 'W2', 1);
% setappdata(fig, 'T', 0);
% 
% % Update the plot
% update_plot(fig, ax);
initial_W1=1;
initial_W2=1;
initial_T=1;
Qbeta=1;
Z_plot(initial_W1,initial_W2,initial_T,Qbeta)


%% Q1-B 
fig = figure('Position', [100, 100, 800, 600]);

% Create axes for the plot
ax = axes('Parent', fig, 'Position', [0.3, 0.1, 0.6, 0.8]);
W1=1;
W2=1;
T=1;

    [X, Y] = meshgrid(linspace(-1, 1, 100));
    
    % Calculate z using the provided activation function
    b = 1; % You can adjust this value if needed
    fact = 1./(exp(-b*(W1.*X + W2.*Y - T)) + 1);
    
    % Plot z
    cla(ax); % Clear previous plot
    mesh(ax, X, Y, fact);
    xlabel(ax, 'X');
    ylabel(ax, 'Y');
    zlabel(ax, 'z');
    title(ax, 'Output z');



%% Q1-c
fig = figure('Position', [100, 100, 800, 600]);

% Create axes for the plot
ax = axes('Parent', fig, 'Position', [0.3, 0.1, 0.6, 0.8]);
W1=1;
W2=1;
T=1;

    [X, Y] = meshgrid(linspace(-1, 1, 100));
    
    % Calculate z using the provided activation function
    b = 1000; % You can adjust this value if needed
    fact = 1./(exp(-b*(W1.*X + W2.*Y - T)) + 1);
    
    % Plot z
    cla(ax); % Clear previous plot
    mesh(ax, X, Y, fact);
    xlabel(ax, 'X');
    ylabel(ax, 'Y');
    zlabel(ax, 'z');
    title(ax, 'Output z');




%% Q1-D
fig = figure('Position', [100, 100, 800, 600]);

% Create axes for the plot
ax = axes('Parent', fig, 'Position', [0.3, 0.1, 0.6, 0.8]);
w1=1;
w2=1;
T=1;

    [X, Y] = meshgrid(linspace(-1, 1, 100));
    
    % Calculate z using the provided activation function
    b = 0.01; % You can adjust this value if needed
    fact = 1./(exp(-b*(W1.*X + W2.*Y - T)) + 1);
    
    % Plot z
    cla(ax); % Clear previous plot
    mesh(ax, X, Y, fact);
    xlabel(ax, 'X');
    ylabel(ax, 'Y');
    zlabel(ax, 'z');
    title(ax, 'Output z');



%% Functions
function update_plot (src, event,new_w1, new_w2, new_T)
    global p % Access the global variable p
    dx = 0.01;
    dy = 0.01;
    beta = 0.5;

    x = -1:dx:1;
    y = -1:dy:1;
    [X,Y] = meshgrid(x,y);

    % Get the values of sliders for w1, w2, and T
    w1 = get(new_w1, 'Value'); 
    w2 = get(new_w2, 'Value'); 
    T = get(new_T, 'Value'); 
    
    % Calculate the activation function
    f_act = 1./(1 + exp(-1*beta.*(w1*X+w2*Y-T))); 
    
    % Update the plot with the new activation function values
    p.ZData = f_act; 

    % Print the current values of w1, w2, and T
    fprintf("w1= %d   ,   w2 = %d   ,   T = %d \n", w1, w2, T);
end

function Z_plot(init_w1, init_w2, init_T, beta)
    global p 
    dx = 0.01;
    dy = 0.01;

    x = -1:dx:1;
    y = -1:dy:1;
    [X,Y] = meshgrid(x,y);

    % Calculate the initial activation function values
    f_act = 1./(1 + exp(-1*beta.*(init_w1*X+init_w2*Y-init_T))); 

    f = figure;
    ax = axes('Parent',f,'position',[0.13 0.39  0.77 0.54]);

    % Create sliders for w1, w2, and T
    s_w1 = uicontrol('Parent',f,'Style','slider','Position',[81,110,419,23],...
                  'value',init_w1, 'min',-5, 'max',5);
    s_w2 = uicontrol('Parent',f,'Style','slider','Position',[81,70,419,23],...
                  'value',init_w2, 'min',-5, 'max',5);

    s_T = uicontrol('Parent',f,'Style','slider','Position',[81,30,419,23],...
                  'value',init_T, 'min',-5, 'max',5);

    % Create a 3D plot of the activation function
    p = mesh(ax, X, Y, f_act); 
    xlabel('X');
    ylabel('Y');
    colorbar;

    % Set callbacks for sliders to update the plot
    s_w1.Callback = {@update_plot, s_w1, s_w2, s_T};
    s_w2.Callback = {@update_plot, s_w1, s_w2, s_T};
    s_T.Callback = {@update_plot, s_w1, s_w2, s_T};
end






















