function forwardselected = forwardselection2(TrainMat, LabelTrain, topfeatures)
%% input: TrainMat - a NxM matrix that contains the full list of features
%% of training data. N is the number of training samples and M is the
%% dimension of the feature. So each row of this matrix is the face
%% features of a single person.
%%        LabelTrain - a Nx1 vector of the class labels of training data
%%        topfeatures - a Kx2 matrix that contains the information of the
%% top 1% features of the highest variance ratio. K is the number of
%% selected feature (K = ceil(M*0.01)). The first column of this matrix is
%% the index of the selected features in the original feature list. So the
%% range of topfeatures(:,1) is between 1 and M. The second column of this
%% matrix is the variance ratio of the selected features.

%% output: forwardselected - a Px1 vector that contains the index of the 
%% selected features in the original feature list, where P is the number of
%% selected features. The range of forwardselected is between 1 and M. 

index=topfeatures(:,1).'; % get the corresponding indices from AVR ranking
% t(:,index)
num_features=size(index,2);
train_featureVector=TrainMat(:,index);
train_labels=categorical(LabelTrain);

output=[]; % intialize a null vector 
Q=train_featureVector; % full feature vector
row=size(train_featureVector,1); % row
col=size(train_featureVector,2); % col
P=zeros(row,col);
P_hat=zeros(row,col);
acc=zeros(1,num_features);
%begin loop
% stopping codition is that we have 85% accuracy or 10% of the features retained
j=1;
while j<=col*0.1 % if doing for accuracy greater than a percentage run this loop for whole and uncomment the if loop break condition at the end of while loop
    for i=1:size(Q,2)
    
    P_hat(:,j)=Q(:,i);  % take each feature from Q and add it to P_hat
    P_hat=P_hat(:,1:j); % P_hat size shoudl keep growing and not remain stagnant, this ensures it. 
    MdlLinear = fitcdiscr(P_hat,train_labels);
    train_pred = predict(MdlLinear,P_hat);  
    train_ConfMat = confusionmat(train_labels,train_pred);
    train_ClassMat = train_ConfMat./(meshgrid(countcats(train_labels))');
    acc(i) = mean(diag(train_ClassMat));
    %train_std = std(diag(train_ClassMat))
    end
% this loop ensures that the first feature is selected in case there is a
% clash
max_acc_index=find(acc==max(acc));
if size(max_acc_index,2)>1
   max_acc_index=max_acc_index(1);
   Q_new=Q(:,max_acc_index);
   P(:,j)=Q_new(:,1);
   
else  
   P(:,j)=Q(:,max_acc_index);  
end
P_hat=P;
Q(:,max_acc_index)=[];  % remove the current maximum
acc(max_acc_index)=[];  % remove the current maximum
j=j+1;  
output=[output max_acc_index]; % keeps track of feature indices with max accuracy.

% train_featureVector=P(:,1:j-1);
% MdlLinear = fitcdiscr(train_featureVector,train_labels);
% train_pred = predict(MdlLinear,train_featureVector);
% train_ConfMat = confusionmat(train_labels,train_pred);
% train_ClassMat = train_ConfMat./(meshgrid(countcats(train_labels))');
% train_acc = mean(diag(train_ClassMat));
%[TP, FP, TN, FN]=calError(train_labels, train_pred);
%FN_rate=(FN)/(TP+FP+FN+TN);
%if FN_rate<=0.1
% break
% end
end
final_out=index(output);
forwardselected=final_out.';
end