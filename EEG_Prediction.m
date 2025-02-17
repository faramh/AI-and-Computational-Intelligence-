%AI and Computaational intelligence Final project 
%Mohamad Hosein Faramarzi - 99104095
%% Load data set 
clc
clear

% Load the data
DataSet=load('I:\Portal\Artificial Intelligence\Project\Cl_Project\Project_data.mat');

TrainData=DataSet.TrainData;
TrainLabels=DataSet.TrainLabels;
TestData=DataSet.TestData;
Frequency=DataSet.fs;
Channels=DataSet.Channels;

%% Plotting Some samples 


% Plotting the first channel of the first observation
Fs = 1000;
t = 1/Fs:1/Fs:5;
tiledlayout(2,1)

nexttile
plot(t, TrainData(1,:,1));
xlabel('t(s)');
title('The First Channel of The 1st Observation');
grid minor 

nexttile
plot(t, TrainData(1,:,3));
xlabel('t(s)');
title('The First Channel of The 3rd Observation');
grid minor 



%% Calculate Features

% Loop through each subject's EEG data
n = size(TrainData,3);
for m = 1:n
    for i = 1:size(TrainData,1)
        
        % Basic statistical features
        Mean(i,m) = mean(TrainData(i,:,m));
        Mode(i,m) = mode(TrainData(i,:,m));
        Var(i,m) = var(TrainData(i,:,m));
        Median(i,m) = median(TrainData(i,:,m));
        Skewness(i,m) = skewness(TrainData(i,:,m));
        Kurtosis(i,m) = kurtosis(TrainData(i,:,m));
        Max(i,m) = max(TrainData(i,:,m));
        [Min(i,m), MinTime(i,m)] = min(TrainData(i,:,m));
        
        % Energy and Power features
        Energy(i,m) = sum(TrainData(i,:,m).^2);
        AveragePower(i,m) = meansqr(TrainData(i,:,m));
        RMS(i,m) = rms(TrainData(i,:,m));
        
        % Correlation matrix
        for j = 1:i
            Correlation((i-1)*30+j,m) = corr(TrainData(i,:,m)', TrainData(j,:,m)');
        end
        
        % Other features
        Moment(i,m) = moment(TrainData(i,:,m),3);
        FormFactor(i,m) = sqrt(var(diff(TrainData(i,:,m)))^2) / ...
            (std(diff(diff(TrainData(i,:,m)))) * sqrt(var(TrainData(i,:,m))));
        FreqMean(i,m) = meanfreq(TrainData(i,:,m));
        FreqMedian(i,m) = medfreq(TrainData(i,:,m));
        Periodogram(i,m,:) = periodogram(TrainData(i,:,m));
        [PeakFrequency(i,m), MaxFreqIndex(i,m)] = max(Periodogram(i,m,:));
        aR = ar(TrainData(i,:,m),4);
        AR(i,m) = mean(aR.A);
        BandPower(i,m) = bandpower(TrainData(i,:,m));
    end
end

% Nonlinear Energy feature
for m = 1:n   
    for i = 2:size(TrainData,1)-1
        NonlinearEnergy(i,m) = sum(TrainData(i,:,m).^2 - TrainData(i+1,:,m).*TrainData(i-1,:,m));
    end
    NonlinearEnergy(30,:) = NonlinearEnergy(29,m);
end

% Combine time and frequency features
TimeFeatures = [Mean; Var; Mode; Median; Skewness; Kurtosis; Max; MaxTime; ...
    Min; MinTime; Energy; AveragePower; RMS; Correlation; Moment; FormFactor; ...
    NonlinearEnergy; AR];

FrequencyFeatures = [FreqMean; FreqMedian; PeakFrequency; MaxFreqIndex; BandPower];

% Normalize features
TimeFeatures = mapstd(TimeFeatures, 0, 1);
TimeFeatures = mapminmax(TimeFeatures);

FrequencyFeatures = mapstd(FrequencyFeatures, 0, 1);
FrequencyFeatures = mapminmax(FrequencyFeatures);

% Save features
save('FrequencyFeatures', 'FrequencyFeatures')
save('TimeFeatures', 'TimeFeatures')

%% Test Features

% Loop through each subject's test EEG data
Test_Features = [];
Num_Test = size(TestData,3);

for m = 1:Num_Test
    for i = 1:size(TestData,1)
        
        % Basic statistical features
        Mean_Test(i,m) = mean(TestData(i,:,m));
        Mode_Test(i,m) = mode(TestData(i,:,m));
        Var_Test(i,m) = var(TestData(i,:,m));
        Median_Test(i,m) = median(TestData(i,:,m));
        Skewness_Test(i,m) = skewness(TestData(i,:,m));
        Kurtosis_Test(i,m) = kurtosis(TestData(i,:,m));
        Max_Test(i,m) = max(TestData(i,:,m));
        [Min_Test(i,m), MinTime_Test(i,m)] = min(TestData(i,:,m));
        
        % Energy and Power features
        Energy_Test(i,m) = sum(TestData(i,:,m).^2);
        AveragePower_Test(i,m) = meansqr(TestData(i,:,m));
        RMS_Test(i,m) = rms(TestData(i,:,m));
        
        % Correlation matrix
        for j = 1:size(TestData,1)
            Correlation_Test((i-1)*30+j,m) = corr(TestData(i,:,m)', TestData(j,:,m)');
        end
        
        % Other features
        Moment_Test(i,m) = moment(TestData(i,:,m),3);
        FormFactor_Test(i,m) = sqrt(var(diff(TestData(i,:,m)))^2) / ...
            (std(diff(diff(TestData(i,:,m)))) * sqrt(var(TestData(i,:,m))));
        FreqMean_Test(i,m) = meanfreq(TestData(i,:,m));
        FreqMedian_Test(i,m) = medfreq(TestData(i,:,m));
        Periodogram_Test(i,m,:) = periodogram(TestData(i,:,m));
        [PeakFrequency_Test(i,m), MaxFreqIndex_Test(i,m)] = max(Periodogram_Test(i,m,:));
        aR_Test = ar(TestData(i,:,m),4);
        AR_Test(i,m) = mean(aR_Test.A);
        BandPower_Test(i,m) = bandpower(TestData(i,:,m));
    end
end

% Nonlinear Energy feature
for m = 1:Num_Test
    for i = 2:size(TestData,1)-1
        NonlinearEnergy_Test(i,m) = sum(TestData(i,:,m).^2 - ...
            TestData(i+1,:,m).*TestData(i-1,:,m));
    end
    NonlinearEnergy_Test(30,:) = NonlinearEnergy_Test(29,m);
end

% Combine time and frequency features
TimeFeatures_Test = [Mean_Test; Var_Test; Mode_Test; Median_Test; ...
    Skewness_Test; Kurtosis_Test; Max_Test; MaxTime_Test; ...
    Min_Test; MinTime_Test; Energy_Test; AveragePower_Test; ...
    RMS_Test; Correlation_Test; Moment_Test; FormFactor_Test; ...
    NonlinearEnergy_Test; AR_Test];

FrequencyFeatures_Test = [FreqMean_Test; FreqMedian_Test; PeakFrequency_Test; ...
    MaxFreqIndex_Test; BandPower_Test];

% Normalize features
TimeFeatures_Test = mapstd(TimeFeatures_Test, 0, 1);
TimeFeatures_Test = mapminmax(TimeFeatures_Test);

FrequencyFeatures_Test = mapstd(FrequencyFeatures_Test, 0, 1);
FrequencyFeatures_Test = mapminmax(FrequencyFeatures_Test);

% Save features
save('FrequencyFeatures_Test', 'FrequencyFeatures_Test')
save('TimeFeatures_Test', 'TimeFeatures_Test')

%% Feature Selection

% Load Time and Frequency Features
load('TimeFeatures');
load('FrequencyFeatures');
load('FrequencyFeatures_Test');
load('TimeFeatures_Test');

% Define Classes
Class1 = find(TrainLabels == 1);
Class0 = find(TrainLabels == -1);

% Calculate J for Time Features
for i = 1:size(TimeFeatures)
    u1 = mean(TimeFeatures(i, Class1));
    S1 = (TimeFeatures(i, Class1) - u1) * (TimeFeatures(i, Class1) - u1)';
    u2 = mean(TimeFeatures(i, Class0));
    S2 = (TimeFeatures(i, Class0) - u2) * (TimeFeatures(i, Class0) - u2)';
    Sw = S1 + S2;
    if Sw == 0
        Sw = 0.1;
    end
    u0 = mean(TimeFeatures(i, :));
    Sb = (u1 - u0)^2 + (u2 - u0)^2;

    J(i) = Sb / (Sw);
end

% Save position of best J (50 best)
[temp, originalpos] = sort(J, 'descend');
BestTimePositions = originalpos(1:50);

% Time Feature Extraction
for t = 1:size(BestTimePositions, 2)
    BestTimeFeatures(t, :) = TimeFeatures(BestTimePositions(t), :);
    BestTimeFeatures_Test(t, :) = TimeFeatures_Test(BestTimePositions(t), :);
end

% Save position of Frequency Features
for j = 1:size(FrequencyFeatures)
    u11 = mean(FrequencyFeatures(j, Class1));
    S11 = (FrequencyFeatures(j, Class1) - u11) * (TimeFeatures(j, Class1) - u11)';
    u22 = mean(FrequencyFeatures(j, Class0));
    S22 = (FrequencyFeatures(j, Class0) - u22) * (FrequencyFeatures(j, Class0) - u22)';
    Sww = S11 + S22;

    u00 = mean(FrequencyFeatures(j, :));
    Sbb = (u11 - u00)^2 + (u22 - u00)^2;

    JJ(j) = Sbb / Sww;
end

% Save position of best JJ
[tempf, originalposf] = sort(JJ, 'descend');
BestFrequencyPositions = originalposf(1:50);

% Frequency Feature Extraction
for r = 1:size(BestFrequencyPositions, 2)
    BestFrequencyFeatures(r, :) = FrequencyFeatures(BestFrequencyPositions(r), :);
    BestFrequencyFeatures_Test(r, :) = FrequencyFeatures_Test(BestFrequencyPositions(r), :);
end

% Normalizing Time Features
BestTimeFeatures = mapstd(BestTimeFeatures, 0, 1);
BestTimeFeatures = mapminmax(BestTimeFeatures);

BestTimeFeatures_Test = mapstd(BestTimeFeatures_Test, 0, 1);
BestTimeFeatures_Test = mapminmax(BestTimeFeatures_Test);

% Normalizing Frequency Features
BestFrequencyFeatures = mapstd(BestFrequencyFeatures, 0, 1);
BestFrequencyFeatures = mapminmax(BestFrequencyFeatures);

BestFrequencyFeatures_Test = mapstd(BestFrequencyFeatures_Test, 0, 1);
BestFrequencyFeatures_Test = mapminmax(BestFrequencyFeatures_Test);

% Concatenate Time & Frequency Features
BestFeatures = vertcat(BestTimeFeatures, BestFrequencyFeatures);
BestFeatures_Test = vertcat(BestTimeFeatures_Test, BestFrequencyFeatures_Test);

% Save selected features
save('BestFeatures', 'BestFeatures');
save('BestFeatures_Test', 'BestFeatures_Test');

%% MLP NETWORK_ TWO OUTPUT

% Load selected features and test data
load('BestFeatures');
load('BestFeatures_Test');

% Define the number of neurons for each experiment
number_neuron = [1, 5, 10, 20, 30, 35, 40, 50, 60, 70, 100];

% Initialize accuracy array
accuracy = zeros(1, length(number_neuron));

% Convert class labels to a suitable format for the neural network
Trian_Label = zeros(2, length(TrainLabels));
Trian_Label(1, TrainLabels == 1) = 1;
Trian_Label(2, TrainLabels == -1) = 1;

% Iterate over the number of neurons
for i = 1:length(number_neuron)
    N = number_neuron(i);
    
    % Perform 5-fold cross-validation
    for j = 1:5
        cv = cvpartition(size(BestFeatures, 2), 'HoldOut', 0.2);
        idx = cv.test;

        XTrain = BestFeatures(:, ~idx);
        XValid = BestFeatures(:, idx);

        YTrain = Trian_Label(:, ~idx);
        YValid = Trian_Label(:, idx);

        % Create and train the neural network
        net = patternnet(N);
        net = train(net, XTrain, YTrain);

        % Make predictions on the validation set
        predict_y = net(XValid);

        % Convert predictions to binary format
        for k = 1:length(predict_y)
            if predict_y(1, k) > predict_y(2, k)
                predict_y(1, k) = 1;
                predict_y(2, k) = 0;
            else
                predict_y(1, k) = 0;
                predict_y(2, k) = 1;
            end
        end

        % Calculate accuracy
        accuracy(i) = accuracy(i) + length(find(predict_y(1, :) == YValid(1, :)));
    end

    % Normalize accuracy by the number of experiments and folds
    accuracy(i) = accuracy(i) / (5 * 0.2 * 550);

    % Save the model with the highest accuracy
    if i == 1
        best_neuron_number = N;
        XTest = BestFeatures_Test;

        % Make predictions on the test set
        predict_y = net(XTest);

        % Convert predictions to binary format
        for k = 1:length(predict_y)
            if predict_y(1, k) > predict_y(2, k)
                predict_y(1, k) = 1;
            else
                predict_y(1, k) = 0;
            end
        end

        % Save the predictions
        MLP_TWO_OUTPUT_LABEL = predict_y;
        save('MLP_TWO_OUTPUT_LABEL', 'MLP_TWO_OUTPUT_LABEL');
    elseif accuracy(i) > max(accuracy(1, 1:i-1))
        best_neuron_number = N;
        YTrain = TrainLabels;
        XTest = BestFeatures_Test;

        % Make predictions on the test set
        predict_y = net(XTest);

        % Convert predictions to binary format
        for k = 1:length(predict_y)
            if predict_y(1, k) > predict_y(2, k)
                predict_y(1, k) = 1;
            else
                predict_y(1, k) = 0;
            end
        end

        % Save the predictions
        MLP_TWO_OUTPUT_LABEL = predict_y(1, :);
        save('MLP_TWO_OUTPUT_LABEL', 'MLP_TWO_OUTPUT_LABEL');
    end
end

% Display the best number of neurons and the maximum accuracy
best_neuron_number
max(accuracy)


%%
MLP_Phase1_Labels=(2*MLP_TWO_OUTPUT_LABEL)-1;

%% RBF NETWORK_TWO OUTPUT

% Load selected features and test data
load('BestFeatures');
load('BestFeatures_Test');

% Define parameters for the RBF network
MN = [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 100];
spreed = [1, 5, 10, 15, 20, 25, 30];

% Initialize accuracy array
accuracy = zeros(length(MN), length(spreed));

% Set the goal for the RBF network
goal = 10e-5;
M = 0;

% Convert class labels to a suitable format for the neural network
Trian_Label = zeros(2, length(TrainLabels));
Trian_Label(1, TrainLabels == 1) = 1;
Trian_Label(2, TrainLabels == -1) = 1;

% Iterate over the number of neurons and spread values
for i = 1:length(MN)
    for j = 1:length(spreed)
        % Perform 5-fold cross-validation
        for k = 1:5
            cv = cvpartition(size(BestFeatures, 2), 'HoldOut', 0.2);
            idx = cv.test;

            XTrain = BestFeatures(:, ~idx);
            XValid = BestFeatures(:, idx);

            YTrain = Trian_Label(:, ~idx);
            YValid = Trian_Label(:, idx);

            % Create and train the RBF network
            net = newrb(XTrain, YTrain, goal, spreed(j), MN(i));

            % Make predictions on the validation set
            predict_y = net(XValid);

            % Convert predictions to binary format
            for k = 1:length(predict_y)
                if predict_y(1, k) > predict_y(2, k)
                    predict_y(1, k) = 1;
                    predict_y(2, k) = 0;
                else
                    predict_y(1, k) = 0;
                    predict_y(2, k) = 1;
                end
            end

            % Calculate accuracy
            accuracy(i, j) = accuracy(i, j) + length(find(predict_y(1, :) == YValid(1, :)));
        end

        % Normalize accuracy by the number of experiments and folds
        accuracy(i, j) = accuracy(i, j) / (5 * 0.2 * 550);

        % Save the model with the highest accuracy
        if accuracy(i, j) > M
            best_neuron_number = MN(1, i);
            best_spreed = spreed(1, j);

            XTest = BestFeatures_Test;

            % Make predictions on the test set
            predict_y = net(XTest);

            % Convert predictions to binary format
            for k = 1:length(predict_y)
                if predict_y(1, k) > predict_y(2, k)
                    predict_y(1, k) = 1;
                else
                    predict_y(1, k) = 0;
                end
            end

            % Save the predictions
            RBF_ONE_OUTPUT_LABEL = predict_y;
            save('RBF_TWO_OUTPUT_LABEL', 'RBF_ONE_OUTPUT_LABEL');
        end

        % Update the maximum accuracy
        M = max(accuracy, [], 'all');
    end
end

% Display the best number of neurons, spread, and the maximum accuracy
best_neuron_number
best_spreed
max(accuracy, [], 'all')

%% Evolutionary Algorithm

% Load data
load('TimeFeatures');
load('TimeFeatures_Test');
load('FrequencyFeatures');
load('FrequencyFeatures_Test');

% Choose best features based on J value
Class1 = find(TrainLabels == 1);
Class0 = find(TrainLabels == -1);

for i = 1:size(TimeFeatures)
    u1 = mean(TimeFeatures(i, Class1));
    S1 = (TimeFeatures(i, Class1) - u1) * (TimeFeatures(i, Class1) - u1)';
    u2 = mean(TimeFeatures(i, Class0));
    S2 = (TimeFeatures(i, Class0) - u2) * (TimeFeatures(i, Class0) - u2)';
    Sw = S1 + S2;
    if Sw == 0
        Sw = 0.1;
    end
    u0 = mean(TimeFeatures(i, :));
    Sb = (u1 - u0)^2 + (u2 - u0)^2;

    J(i) = Sb / Sw;
end

[ind, originalpos] = sort(J, 'descend');
n = ind(1:100);
p = originalpos(1:100);
BestTimePositions = p;

% Time Feature Extraction
for t = 1:size(p, 2)
    BestTimeFeatures(t, :) = TimeFeatures(p(t), :);
end

for h = 1:size(BestTimePositions, 2)
    BestTimeFeatures_Test(h, :) = TimeFeatures_Test(BestTimePositions(h), :);
end

for j = 1:size(FrequencyFeatures)
    u11 = mean(FrequencyFeatures(j, Class1));
    S11 = (FrequencyFeatures(j, Class1) - u11) * (TimeFeatures(j, Class1) - u11)';
    u22 = mean(FrequencyFeatures(j, Class0));
    S22 = (FrequencyFeatures(j, Class0) - u22) * (FrequencyFeatures(j, Class0) - u22)';
    Sww = S11 + S22;

    u00 = mean(FrequencyFeatures(j, :));
    Sbb = (u11 - u00)^2 + (u22 - u00)^2;

    JJ(j) = Sbb / Sww;
end

[ind, originalposf] = sort(JJ, 'descend');
nn = ind(1:100);
pp = originalposf(1:100);
BestFrequencyPositions = pp;

for r = 1:size(p, 2)
    BestFrequencyFeatures(r, :) = FrequencyFeatures(pp(r), :);
end

for h = 1:size(BestTimePositions, 2)
    BestFrequencyFeatures_Test(h, :) = FrequencyFeatures_Test(BestFrequencyPositions(h), :);
end

best_train = vertcat(BestTimeFeatures, BestFrequencyFeatures);
best_test = vertcat(BestTimeFeatures_Test, BestFrequencyFeatures_Test);

% Initial Chromosome
Chromosome = zeros(20, length(best_train));

for i = 1:size(Chromosome, 1)
    a = randperm(size(Chromosome, 2));
    Chromosome(i, a(1:100)) = 1;
end

for pop = 1:20
    k = 1;
    for i = 1:200
        if Chromosome(pop, i) == 1
            initial_population(k, :, pop) = best_train(i, :);
            k = k + 1;
        end
    end
end

% Initial Fitness
for b = 1:size(initial_population, 3)
    for c = 1:size(initial_population, 1)
        initial_fitness(b) = Fitness(initial_population(c, :, b), Class1, Class0);
    end
end

% Generation
generation_step = 200;
Generation = zeros(generation_step, size(Chromosome, 1), size(Chromosome, 2));
Generation(1, :, :) = Chromosome(:, :);

cross_number = 1;
mut_number = 1;

fitness = zeros(generation_step, 20);

for n = 1:generation_step
    for g = 1:20
        if fitness(n, g) == max(fitness(n, :))
            max_individual = g;
        end
    end

    % Mutation
    p_rand = rand;
    p_mut = 0.05;
    if p_mut < p_rand
        indx = randperm(20);
        ind_val = Chromosome(indx, :);
    end
    [Generation(mut_number + 1, indx, :)] = mutation(Generation(mut_number, indx, :));

    Generation(mut_number + 1, max_individual, :) = Chromosome(max_individual, :);
    Chromosome(:, :) = Generation(mut_number + 1, :, :);

    for pop = 1:20
        k = 1;
        for i = 1:200
            if Chromosome(pop, i) == 1
                population(k, :, pop) = best_train(i, :);
                k = k + 1;
            end
        end
    end

    for b = 1:size(population, 3)
        for c = 1:size(population, 1)
            fitness(n + 1, b) = Fitness(population(c, :, b), Class1, Class0);
        end
    end

    % Crossover
    p_rand = rand;
    p_crossover = 0.8;

    if p_rand > p_crossover
        inx1 = randperm(20);
        inx2 = randperm(20);
        if inx1 ~= inx2
            ind1 = Chromosome(inx1, :);
            ind2 = Chromosome(inx2, :);
        else
            break
        end

        [Generation(cross_number + 1, inx1, :), Generation(cross_number + 1, inx2, :)] ...
            = crossover(Generation(cross_number, inx1, :), Generation(cross_number, inx2, :));

        % Find best J
        Generation(cross_number + 1, max_individual, :) = Chromosome(max_individual, :);
        Chromosome(:, :) = Generation(cross_number + 1, :, :);

        for pop = 1:20
            k = 1;
            for i = 1:200
                if Chromosome(pop, i) == 1
                    population(k, :, pop) = best_train(i, :);
                    k = k + 1;
                end
            end
        end

        for b = 1:size(initial_population, 3)
            for c = 1:size(initial_population, 1)
                fitness(n + 1, b) = Fitness(population(c, :, b), Class1, Class0);
            end
        end

        num_cross = num_cross + 1;
        num_mut = num_mut + 1;
    end
end

last_chromosome = Chromosome;

for pop = 1:20
    k = 1;
    for i = 1:200
        if last_chromosome(pop, i) == 1
            last_pop_train(k, :, :) = best_train(i, :);
            last_pop_test(k, :, :) = best_test(i, :);
            k = k + 1;
        end
    end
end

last_fitness = Fitness(last_pop_train, Class1, Class0);

for g = 1:20
    if fitness(n, g) == max(fitness(n, :))
        max_individual = g;
    end
end

% Get the best features in the last population
k = 1;
for i = 1:200
    if last_chromosome(max_individual, i) == 1
        last_pop_train(k, :, :) = best_train(i, :);
        last_pop_test(k, :, :) = best_test(i, :);
        k = k + 1;
    end
end
Last_Pop_Train = last_pop_train(:, :, :);
Last_Pop_Test = last_pop_test(:, :, :);

save('Last_Pop_Test', 'Last_Pop_Test');
save('Last_Pop_Train', 'Last_Pop_Train');

%% MLP NETWORK_ TWO OUTPUT

% Load the necessary data
load('Last_Pop_Test');
load('Last_Pop_Train');

% Define a range of neuron numbers for experimentation
number_neuron = [1, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80];

% Initialize an array to store accuracy values for each neuron count
accuracy = zeros(1, length(number_neuron));

% Create binary labels for training data (Two-class problem)
Train_Labels_TwoClass = zeros(2, length(TrainLabels));
Train_Labels_TwoClass(1, TrainLabels == 1) = 1;
Train_Labels_TwoClass(2, TrainLabels == -1) = 1;

% Loop over different neuron counts
for i = 1:length(number_neuron)
    N = number_neuron(i);
    
    % Perform 5-fold cross-validation
    for j = 1:5
        % Split the training data into training and validation sets
        cv = cvpartition(size(Last_Pop_Train, 2), 'HoldOut', 0.2);
        idx = cv.test;
        XTrain = Last_Pop_Train(:, ~idx);
        XValid = Last_Pop_Train(:, idx);
        
        YTrain = Train_Labels_TwoClass(:, ~idx);
        YValid = Train_Labels_TwoClass(:, idx);
        
        % Create and train a pattern recognition neural network
        net = patternnet(N);
        net = train(net, XTrain, YTrain);
        
        % Predict labels for the validation set
        predict_y = net(XValid);
        
        % Convert the network output to binary labels
        for k = 1:length(predict_y)
            if predict_y(1, k) > predict_y(2, k)
                predict_y(1, k) = 1;
                predict_y(2, k) = 0;
            else
                predict_y(1, k) = 0;
                predict_y(2, k) = 1;
            end
        end

        % Calculate accuracy for the current fold
        accuracy(i) = accuracy(i) + length(find(predict_y(1, :) == YValid(1, :))) ;
    end
    
    % Calculate the average accuracy over all folds
    accuracy(i) = accuracy(i) / (5 * 0.2 * 550);
    
    % Save the results based on whether it is the first iteration or if
    % accuracy improves during the evolution
    if i == 1
        best_neuron_number = N;

        XTest = Last_Pop_Test;

        predict_y = net(XTest);
        for k = 1:length(predict_y)
            if predict_y(1, k) > predict_y(2, k)
                predict_y(1, k) = 1;
            else
                predict_y(1, k) = 0;
            end
        end
        
        MLP_TWO_OUTPUT_LABEL = predict_y;
        
        % Save the results for the initial run
        save('MLP_TWO_OUTPUT_LABEL', 'MLP_TWO_OUTPUT_LABEL');
        
    elseif accuracy(i) > max(accuracy(1, 1:i-1))
        best_neuron_number = N;

        XTest = Last_Pop_Test;

        predict_y = net(XTest);
        for k = 1:length(predict_y)
            if predict_y(1, k) > predict_y(2, k)
                predict_y(1, k) = 1;
            else
                predict_y(1, k) = 0;
            end
        end
        
        MLP_TWO_OUTPUT_LABEL = predict_y(1,:);
        
        % Save the results if accuracy improves during the evolution
        save('MLP_TWO_OUTPUT_LABEL_EV', 'MLP_TWO_OUTPUT_LABEL');
    end
    
end

% Display the maximum accuracy achieved and the corresponding neuron count
max(accuracy);
best_neuron_number


%%
MLP_phase2_labels=(2*MLP_TWO_OUTPUT_LABEL)-1;

%% RBF NETWORK_ TWO OUTPUT

% Load the necessary data
load('Last_Pop_Test');
load('Last_Pop_Train');

% Define the number of neurons and spread values for experimentation
MN = [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 100];
spreed = [1, 5, 10, 15, 20, 25, 30];

% Initialize variables
M = 0;
accuracy = zeros(length(MN), length(spreed));
goal = 10e-5;

% Create binary labels for training data (Two-class problem)
Trian_Label = zeros(2, length(TrainLabels));
Trian_Label(1, TrainLabels == 1) = 1;
Trian_Label(2, TrainLabels == -1) = 1;

% Loop over different neuron counts and spread values
for i = 1:length(MN)
    for j = 1:length(spreed)
        for k = 1:5
            % Perform 5-fold cross-validation
            cv = cvpartition(size(Last_Pop_Train, 2), 'HoldOut', 0.2);
            idx = cv.test;
            
            XTrain = Last_Pop_Train(:, ~idx);
            XValid = Last_Pop_Train(:, idx);
            
            YTrain = Trian_Label(:, ~idx);
            YValid = Trian_Label(:, idx);
            
            % Create and train an RBF neural network
            net = newrb(XTrain, YTrain, goal, spreed(j), MN(i));
            
            % Predict labels for the validation set
            predict_y = net(XValid);
            
            % Convert the network output to binary labels
            for k = 1:length(predict_y)
                if predict_y(1, k) > predict_y(2, k)
                    predict_y(1, k) = 1;
                    predict_y(2, k) = 0;
                else
                    predict_y(1, k) = 0;
                    predict_y(2, k) = 1;
                end
            end
            
            % Calculate accuracy for the current fold
            accuracy(i, j) = accuracy(i, j) + length(find(predict_y(1, :) == YValid(1, :))) ;
        end
        
        % Calculate the average accuracy over all folds
        accuracy(i, j) = accuracy(i, j) / (5 * 0.2 * 550);
        
        % Save the results if accuracy improves during the evolution
        if accuracy(i, j) > M
            best_neuron_number = MN(1, i);
            best_spreed = spreed(1, j);
            
            XTest = Last_Pop_Test;
            
            predict_y = net(XTest);
            for k = 1:length(predict_y)
                if predict_y(1, k) > predict_y(2, k)
                    predict_y(1, k) = 1;
                else
                    predict_y(1, k) = 0;
                end
            end
            
            RBF_TWO_OUTPUT_LABEL = predict_y;
            
            % Save the results if accuracy improves during the evolution
            save('RBF_TWO_OUTPUT_LABEL_EV', 'RBF_TWO_OUTPUT_LABEL')
        end
        
        % Update the maximum accuracy value
        M = max(accuracy, [], 'all');
    end
end

% Display the best neuron count, best spread, and maximum accuracy achieved
best_neuron_number
best_spreed
max(accuracy, [], 'all')


% Mutation function for the Evolutionary Algorithm
function child = mutation(parent)
    a = randperm(50);
    b = randperm(50);
    child = parent;
    c = child(1, a);
    child(1, a) = child(1, b);
    child(1, b) = c;
end

% Crossover function for the Evolutionary Algorithm
function [child1, child2] = crossover(parent1, parent2)
    a = randperm(49);
    ex1 = parent1(1, 1:a);
    ex2 = parent2(1, 1:a);
    parent2(1, 1:a) = ex1;
    parent1(1, 1:a) = ex2;
    child1 = parent1;
    child2 = parent2;
end

% Fitness function for the Evolutionary Algorithm
function J = Fitness(population, Class1, Class0)
    u11 = mean(population(Class1)) ;
    S11 = (population(Class1) - u11) * (population(Class1) - u11)' ;
    u22 = mean(population(Class0)) ;
    S22 = (population(Class0) - u22) * (population(Class0) - u22)' ;
    Sww = S11 + S22 ;
    u00 = mean(population(:)) ;
    Sbb = (u11 - u00)^2 + (u22 - u00)^2 ;
    J = Sbb / Sww ;
end



