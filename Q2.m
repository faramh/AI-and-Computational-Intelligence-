%% Q2 -A
% Step 1: Load the Data
data = readtable('iris.csv'); % Assuming the file is in the same directory as your script

% Add column names
data.Properties.VariableNames = {'sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'iris_type'};

% Step 2: Separate Classes (Iris-setosa and Iris-versicolor)
setosa = data(strcmp(data.iris_type, 'Iris-setosa'), :);
versicolor = data(strcmp(data.iris_type, 'Iris-versicolor'), :);

% Step 3: Plot Features
features = {'sepal_length', 'sepal_width', 'petal_length', 'petal_width'};

% Create scatter plots for various feature pairs
for i = 1:length(features)
    for j = i+1:length(features)
        figure;
        scatter(setosa.(features{i}), setosa.(features{j}), 'b', 'filled', 'DisplayName', 'Iris-setosa');
        hold on;
        scatter(versicolor.(features{i}), versicolor.(features{j}), 'r', 'filled', 'DisplayName', 'Iris-versicolor');
        xlabel(features{i});
        ylabel(features{j});
        legend();
        hold off;
    end
end

% Step 4: Analyze Linearity



%% Q2 - B
% Randomly select 5 samples from each class
rng(43); % Set a seed for reproducibility
setosa_sample_indices = randperm(height(setosa), 5);
versicolor_sample_indices = randperm(height(versicolor), 5);

setosa_sample = setosa(setosa_sample_indices, :);
versicolor_sample = versicolor(versicolor_sample_indices, :);

% Plot petal length - petal width for the selected samples
figure;
scatter(setosa_sample.petal_length, setosa_sample.petal_width, 'b', 'filled', 'DisplayName', 'Iris-setosa');
hold on;
scatter(versicolor_sample.petal_length, versicolor_sample.petal_width, 'r', 'filled', 'DisplayName', 'Iris-versicolor');
xlabel('Petal Length');
ylabel('Petal Width');
legend();
hold off;



%% Q2 - C/D/E
clc;
clear;
% Importing dataset
data = readtable('iris.csv');

% Petal Length and Petal Width are selected as two linearly seperable
% features.
setosa = data(1:50, 3:5);  % Iris-setosa
versicolor = data(51:100, 3:5); % Iris-versicolor

train_samples_setosa = setosa(1:40, :);
test_samples_setosa = setosa(41:end, :);
train_samples_versicolor = versicolor(1:40, :);
test_samples_versicolor = versicolor(40:end, :);


% Merging training datasets
Data_attached = vertcat(train_samples_setosa, train_samples_versicolor);

% Online training implementation
% Initialize values for weights, threshold, and learning rate
online_weights = [randn(), randn()];
online_threshold = randn();
online_learning_rate = 0.001;
End_iteration = 300;
online_iteration = 0;
online_error = 1;

% Changes of weights and threshold values for batch & online
W1_variation_online = zeros(End_iteration, 1);
W2_variation_online = zeros(End_iteration, 1);
onlineThresholdChanges = zeros(End_iteration, 1);

% Loop for checking error and number of iterations
while online_error ~= 0 && online_iteration < End_iteration
    online_error = 0;
    online_iteration = online_iteration + 1; 

    for i = 1:size(Data_attached, 1)
        x = Data_attached(i, 1:2);
        target_label = Data_attached(i, 3);
        target_value = strcmp(target_label{1, 1}, 'Iris-versicolor');
        activation = x{1, 1} * online_weights(1) + x{1, 2} * online_weights(2) - online_threshold;
        
        if activation >= 0
            predicted_value = 1;
        else
            predicted_value = 0;
        end
        W1_variation_online(online_iteration) = online_weights(1);
        W2_variation_online(online_iteration) = online_weights(2);
        onlineThresholdChanges(online_iteration) = online_threshold;

        % Updating Process
        if(target_value ~= predicted_value)
            online_error = target_value - predicted_value;
            online_threshold = online_threshold - (online_learning_rate * online_error);
            online_weights = online_weights + online_learning_rate * (online_error) * [x{1, 1}, x{1, 2}];
        end
    end
    
end



% Scatter plot
figure;
scatter(train_samples_setosa{:,1}, train_samples_setosa{:,2}, 'r', 'filled', 'DisplayName', 'Iris-setosa');
hold on;
scatter(train_samples_versicolor{:,1}, train_samples_versicolor{:,2}, 'b', 'filled', 'DisplayName', 'Iris-versicolor');

% Decision boundary
x_values = min(Data_attached{:, 1}):0.1:max(Data_attached{:, 1});
y_values = (-online_weights(1) * x_values + online_threshold) / online_weights(2);
plot(x_values, y_values, 'm', 'LineWidth', 2, 'DisplayName', 'Decision Boundary');

xlabel('feature1');
ylabel('feature2');
title('Online Learning');
legend;
grid on;
hold off;

% Batch Learning
% Initialize values for weights, threshold, and learning rate

W1_variation_batch = zeros(End_iteration, 1);
W2_variation_batch = zeros(End_iteration, 1);
Threshold_variation_batch = zeros(End_iteration, 1);

batch_weights= [randn(), randn()];
batch_threshold= randn();
batch_learning_rate = 0.06;
batch_iteration = 0;
batch_error = 1;
% Loop for checking error and number of iterations
while batch_error ~= 0 && batch_iteration < End_iteration
    batch_error = 0;
    batch_iteration = batch_iteration + 1; 
    
    % Store updates for weights and threshold
    delta_weights = [0, 0]; 
    delta_threshold = 0;

    for i = 1:size(Data_attached, 1)
        x = Data_attached(i, 1:2);
        target_label = Data_attached(i, 3);
        target_value = strcmp(target_label{1, 1}, 'Iris-versicolor');
        activation = x{1, 1} * batch_weights(1) + x{1, 2} * batch_weights(2) - batch_threshold;
        
        % Updating Process
        if activation >= 0
            predicted_value = 1;
        else
            predicted_value = 0;
        end
        
        if target_value ~= predicted_value
            batch_error = target_value - predicted_value;
            delta_weights = delta_weights + batch_learning_rate * (batch_error) * [x{1, 1}, x{1, 2}];
            delta_threshold = delta_threshold - (batch_learning_rate * batch_error);
        end
    end

    W1_variation_batch(batch_iteration) = batch_weights(1);
    W2_variation_batch(batch_iteration) = batch_weights(2);
    Threshold_variation_batch(batch_iteration) = batch_threshold;

    % Batch update for weights and threshold
    batch_weights = batch_weights + delta_weights;
    batch_threshold = batch_threshold + delta_threshold;
end


% Scatter plot
figure;
scatter(train_samples_setosa{:,1}, train_samples_setosa{:,2}, 'r', 'filled', 'DisplayName', 'Iris-setosa');
hold on;
scatter(train_samples_versicolor{:,1}, train_samples_versicolor{:,2}, 'b', 'filled', 'DisplayName', 'Iris-versicolor');


% Decision boundary
x_values = min(Data_attached{:, 1}):0.1:max(Data_attached{:, 1});
y_values = (-batch_weights(1) * x_values + batch_threshold) / batch_weights(2);
plot(x_values, y_values, 'm', 'LineWidth', 2, 'DisplayName', 'Decision Boundary');

xlabel('feature1');
ylabel('feature2');
title('Batch Learning');
legend;
grid on;
hold off;
% online
figure;
plot(1:online_iteration+10, W1_variation_online(1:online_iteration+10), 'r', 'LineWidth', 1.5, 'DisplayName', 'W1 online');
hold on;
plot(1:online_iteration+10, W2_variation_online(1:online_iteration+10), 'b', 'LineWidth', 1.5, 'DisplayName', 'w2 online');
plot(1:online_iteration+10, onlineThresholdChanges(1:online_iteration+10), 'm', 'LineWidth', 1.5, 'DisplayName', 'Threshold online');
hold off;
xlabel('Epochs');
ylabel('Updating process');
title('Online Training ');
legend;
grid on;
% batch
figure;
plot(1:batch_iteration+10, W1_variation_batch(1:batch_iteration+10), 'r', 'LineWidth', 1.5, 'DisplayName', 'W1 batch');
hold on;
plot(1:batch_iteration+10, W2_variation_batch(1:batch_iteration+10), 'b', 'LineWidth', 1.5, 'DisplayName', 'W2 batch');
plot(1:batch_iteration+10, Threshold_variation_batch(1:batch_iteration+10), 'm', 'LineWidth', 1.5, 'DisplayName', 'threshold batch');
hold off;
xlabel('Epochs');
ylabel('updating process');
title('Batch Training');
legend;
grid on;
%% Q2-F

% test samples
% Evaluating accuracy using test samples
num_test_samples = size(test_samples_setosa, 1) + size(test_samples_versicolor, 1);
num_correct_predictions_batch = 0;
num_correct_predictions_online = 0;

for i = 1:size(test_samples_setosa, 1)
    x = test_samples_setosa(i, 1:2);

    batch_activation = x{1, 1} * batch_weights(1) + x{1, 2} * batch_weights(2) - batch_threshold;
    online_activation = x{1, 1} * online_weights(1) + x{1, 2} * online_weights(2) - online_threshold;
    
    if (batch_activation < 0)
        num_correct_predictions_batch = num_correct_predictions_batch + 1;
    end
    
    if (online_activation < 0)
        num_correct_predictions_online = num_correct_predictions_online + 1;
    end
end

for i = 1:size(test_samples_versicolor, 1)
    x = test_samples_versicolor(i, 1:2);
    
    batch_activation = x{1, 1} * batch_weights(1) + x{1, 2} * batch_weights(2) - batch_threshold;
    online_activation = x{1, 1} * online_weights(1) + x{1, 2} * online_weights(2) - online_threshold;
    
    if (batch_activation >= 0)
        num_correct_predictions_batch = num_correct_predictions_batch + 1;
    end
    
    if (online_activation >= 0)
        num_correct_predictions_online = num_correct_predictions_online + 1;
    end
end
%% accuracy

%accuracy computing
Batch_accuracy = (num_correct_predictions_batch / num_test_samples) * 100;
Online_accuracy = (num_correct_predictions_online / num_test_samples) * 100;

% Printing accuracy
disp(['Accuracy of Batch Learning: ', num2str(Batch_accuracy), '%']);
disp(['Accuracy of Online Learning: ', num2str(Online_accuracy), '%']);

%% Q2_G
% Step 1: Load the Data
data = readtable('iris.csv'); % Assuming the file is in the same directory as your script

% Add column names
data.Properties.VariableNames = {'sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'iris_type'};

% Step 2: Separate Classes
setosa = data(strcmp(data.iris_type, 'Iris-setosa'), :);
versicolor = data(strcmp(data.iris_type, 'Iris-versicolor'), :);
virginica = data(strcmp(data.iris_type, 'Iris-virginica'), :);

% Step 3: Plot Features in 3D
features = {'sepal_length', 'sepal_width', 'petal_length', 'petal_width'};

for i = 1:length(features)
    for j = i+1:length(features)
        for k = j+1:length(features)
            figure;
            scatter3(setosa.(features{i}), setosa.(features{j}), setosa.(features{k}), 'b', 'filled', 'DisplayName', 'Iris-setosa');
            hold on;
            scatter3(versicolor.(features{i}), versicolor.(features{j}), versicolor.(features{k}), 'r', 'filled', 'DisplayName', 'Iris-versicolor');
            scatter3(virginica.(features{i}), virginica.(features{j}), virginica.(features{k}), 'g', 'filled', 'DisplayName', 'Iris-virginica');
            xlabel(features{i});
            ylabel(features{j});
            zlabel(features{k});
            legend();
            hold off;
        end
    end
end









