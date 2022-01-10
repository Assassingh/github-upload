function topfeatures = rankingfeat2(TrainMat, LabelTrain)
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
M=size(TrainMat,2);
T=LabelTrain;
l=unique(T,'stable'); % get all the classes avilable 
num_classes=size(l,1);
var_mat=var(TrainMat); % varience of entire feature set
%create cells for each class
var_cell=cell(num_classes,2); %initialization to store variences and means
for i=1:num_classes
    row= find(T == l(i));
    class_i_features=TrainMat(row,:);
    var_cell{i,2}=mean(class_i_features);
    var_cell{i,1}=var(class_i_features);
end
%this for loop does class seperation

% mean diff for n class data set
% i=1;
% y=1;
% g=1;
% q=nchoosek(size(var_cell,1),2);
% z=size(var_cell,1);
% diff_cell=cell(q,1);
% while i<=q    
%          while g<z 
%          diff_cell{i}=abs(var_cell{y,2}-var_cell{z,2});
%          y=y+1;
%          i=i+1;
%          g=g+1;
%          end
%      z=z-1;
%      y=1;
%      g=1;
% end

%since this is only a two class data set we can directly subtract them 

min_mean_diff=abs(var_cell{1,2}-var_cell{2,2});

%the commented part is for EEG data set or any other data set with more
%than 2 classes

% mean_diff=zeros(q,M);
% %collapse the cells into this matrix
% for i=1:q
%     mean_diff(i,:)=diff_cell{i};
% end
% min_mean_diff=min(mean_diff);

% collapse variences into a matrix
class_var=zeros(num_classes,M);

for i=1:num_classes
class_var(i,:)=var_cell{i,1};
end
% now calculate AVR

AVR_denominator=zeros(1,M);
% i goes until 49920
for i=1:M
AVR_denominator(i)=(1/num_classes)*(sum(class_var(:,i))/min_mean_diff(i));
end
AVR=zeros(1,M);
for i=1:M
AVR(i)=var_mat(i)/AVR_denominator(i);
end
% clear NaN values
AVR=double(subs(AVR,nan,0));



% sort AVR
top=ceil(M*0.01);
[out,index]=sort(AVR,'descend');%pick out the respective indices and values
topfeatures=[index(1:top).' out(1:top).']; % give out these indices and values

end
