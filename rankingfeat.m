function topfeatures = rankingfeat(TrainMat, LabelTrain)
%% input: TrainMat - a NxM matrix that contains the full list of features
%% of training data. N is the number of training samples and M is the
%% dimension of the feature. So each row of this matrix is the face
%% features of a single person.
%%        LabelTrain - a Nx1 vector of the class labels of training data

%% output: topfeatures - a Kx2 matrix that contains the information of the
%% top 1% features of the highest variance ratio. K is the number of
%% selected feature (K = ceil(M*0.01)). The first column of this matrix is
%% the index of the selected features in the original feature list. So the
%% range of topfeatures(:,1) is between 1 and M. The second column of this
%% matrix is the variance ratio of the selected features.



% Establish global variables for use during loops
[~,m]=size(TrainMat);
class = unique(LabelTrain);
total_classes = length(class);
K = ceil(m*0.05);
feature_avr(m,2) = 0;
class_mean(total_classes) = 0;
class_var(total_classes) = 0;
min_mean_dif(total_classes) = 0;

% Iterate over all m features to find global and class variance
for i = 1:m
    class_mean(:) = 0;
    class_var(:) = 0;
    feature_i = TrainMat(:,i);
    var_f = var(feature_i);
    
    for j = 1:total_classes
        ind = find(LabelTrain == class(j));
        feature_c = feature_i(ind);
        class_var(j) = var(feature_c);
        class_mean(j) = mean(feature_c);
    end
    if(var_f == 0)
        avr = 0;
    else
        % Find the min mean difference for the class
        min_mean_dif(:) = 0;
        for k = 1:total_classes
            tmp = class_mean;
            tmp(k) = [];
            tmp = abs(tmp - class_mean(k));
            min_mean_dif(k) = min(tmp);
        end
        % Find the Augmented Variance Ratio for the feature
        avr_den = sum(class_var./min_mean_dif);
%         avr_den = sum(class_var);
        avr = var_f * total_classes /avr_den;
    end
    if (isnan(avr))
        continue
    end
    feature_avr(i,:) = [i, avr];
end


%sort the AVR features
[~,index] = sort(feature_avr(:,2), 'descend'); 

%sort the whole matrix based on the sorted indices
topfeatures = feature_avr(index, :);  
topfeatures = topfeatures(1:K,:);

    
