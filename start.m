%starter code for project 3: feature selection
%pattern recognition, CSE583/EE552
%Weina Ge, Aug 2008
%Christopher Funk, Jan 2017
%Bharadwaj Ravichandran, Jan 2020
%Shimian Zhang, Feb 2021

%Your Details: (The below details should be included in every matlab script
%file that you create)
%{
    Name: Purushartha Singh
    PSU Email ID: pxs288 (984874392)
    Description: (A short description of what this script does).
%}

clear all;
close all;
clc;

%% load the Taiji data 
load('Taiji_data.mat')

frame_num = size(Taiji_data,1);
form_num = size(keyframes,1);
form_list = linspace(1,form_num,form_num);


%% Labelling each frame
M = 60; N = 400;
labels = zeros(frame_num, 1);
for i=1:frame_num
    take_id = sub_info(i,2); 
    frame_idx = Taiji_data(i,end);      % the frame index
    % label the frame to a specific Taiji key form (or "0" indicates the NON KEY FRAME)
    form_idx = 0;
    [val,idx] = min(abs(frame_idx - keyframes(:,take_id)));
    if ((frame_idx - keyframes(idx,take_id) < 0 && val <= M) || frame_idx - keyframes(idx,take_id) >= 0 && val <= N)
        form_idx = form_list(idx);
    end
    labels(i) = form_idx;
end

feature_names = feature_names(:,1:size(feature_names,2)-1);
Taiji_data = Taiji_data(:,1:size(Taiji_data,2)-1);
Dim = size(Taiji_data,2); %dimension of the feature
countfeat(Dim,2) = 0;
% countfeat is a Mx2 matrix that keeps track of how many times a feature has been selected, where M is the dimension of the original feature space.
% The first column of this matrix records how many times a feature has ranked within top 1% during 100 times of feature ranking.
% The second column of this matrix records how many times a feature was selected by forward feature selection during 100 times.

%%%%%%%%%%%%%%%%%%%% test code %%%%%
%comment this out 
% tmp = randperm(Dim);
% topfeatures(:,1) = tmp(1:1000)';
% topfeatures(:,2) = 100*rand(1000,1);
% forwardselected = tmp(1:100)';
%%%%%%%%%%%%%%%%%%%%%%%%************
conf_matrix_accuracies(10) = 0;
conf_matrix_std(10) = 0;
% total_classes = 7;
class_rates(10, 7) = 0;
conf_mat(7,7,10) = 0;
class_mat(7,7,10) = 0;
%% Loop to iterate over the dataset 10 times and producing 10 sets of results
for i=1:10
    % divide into train and test sets with "Leave one subject out"
    [TrainMat, LabelTrain, TestMat, LabelTest]= split(Taiji_data, labels, sub_info,i);

    % start feature ranking
    topfeatures = rankingfeat(TrainMat, LabelTrain); 
    countfeat(topfeatures(:,1),1) =  countfeat(topfeatures(:,1),1) +1;

    % start forward feature selection
    forwardselected = forwardselection(TrainMat, LabelTrain, topfeatures);

    countfeat(forwardselected,2) =  countfeat(forwardselected,2) +1;    

    % start classification
    train = TrainMat(:,forwardselected);
    test = TestMat(:, forwardselected);
%     Mdl = fitcdiscr(train, LabelTrain);
    Mdl = fitcensemble(train, LabelTrain, 'Method', 'bag', 'Prior', 'empirical', 'NumBins',50);
    predict_train = predict(Mdl, train);
    predict_test = predict(Mdl, test);
    
    % Create confusion matrix
    confMat = confusionmat(LabelTest,predict_test);
    
    % Create classification matrix (rows should sum to 1)
    test_ClassMat = confMat./(meshgrid(countcats(categorical(LabelTest)))');
    
    % Add to global arrays to get average stats for the runs
    test_acc = mean(diag(test_ClassMat));
    conf_matrix_accuracies(i) = test_acc;
    
    test_std = std(diag(test_ClassMat));
    conf_matrix_std(i) = test_std;
    
    class_rates(i,:) = diag(test_ClassMat);
    conf_mat(:,:,i) = confMat;
    class_mat(:,:,i) = test_ClassMat;
    
    
end

%% Visualization of top used features
data(:,1)=[1:Dim]';
data(:,2) = countfeat(:,1);
figure(1)
% visualize the features that have ranked within top 20 (or however many you can display) most during 100 times of feature ranking
plotFeat(data,feature_names,20);
title('{\bf Top 50 Feature Ranking Filter}')
export_fig final_fts_filter -png -transparent
% visualize the features that have been selected most during 100 times of forward selection

data(:,2) = countfeat(:,2);
figure(2)
plotFeat(data,feature_names,20);
title('{\bf Top 50 Feature Ranking Wrapper}')
export_fig final_fts_wrapper -png -transparent

%% Results of global averages for the run
avg_accuracy = mean(conf_matrix_accuracies);
avg_std = mean(conf_matrix_std);
avg_conf_mat = mean(conf_mat,3);
avg_class_mat = mean(class_mat,3);
avg_acc_classwise = mean(class_rates, 1);
class_rates(11,:) = avg_acc_classwise;
%% Classification and Confusion Matricies
labels = ["NON KEY FRAME", "Preparation", "Opening", "Part the Wild Horse’s Mane, 1st Left", "Part the Wild Horse’s Mane, Right", "Part the Wild Horse’s Mane, 2nd Left","White Crane Spreads Its Wings"];

figure(1)
heatmap(labels, labels, avg_conf_mat);
title('{\bf Average Confusion Matrix}')
export_fig conf_mat_final -png -transparent

figure(2)
heatmap( avg_class_mat);
title('{\bf Average Classification Matrix}')
export_fig class_mat_final -png -transparent

label = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "Average"];
figure(3)
heatmap(labels, label, class_rates);

title('{\bf Average Classwise Accuracies}')
export_fig cw_acc_final -png -transparent

%% Record of runs with different M and N values
m60n60_acc = 0.7053;
m100n20_acc = 0.6958;
m400n20_acc = 0.7380;
m400n200_acc = 0.7035;
m40n200_acc = 0.7896;
m60n400_acc = 0.7976;
tree = 0.6412;
knn = 0.7068;



