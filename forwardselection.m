function forwardselected = forwardselection(TrainMat, LabelTrain, topfeatures)
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

    Q = TrainMat(:,topfeatures(:,1));    
    I=topfeatures(:,1);
    
    %Choosing the top features using accuracy
    P=[];
    flag=true;
    acc=0;
    forwardselected=[];
    while flag %    Run the loop till the stopping condition
        q=[];
        fs=0;        
        for k=1:size(Q,2)  
            p=[P Q(:,k)];

            %function to find the accuracy of the linear disciminant 
            Mdl = fitcdiscr(p,LabelTrain);
            predict_train = predict(Mdl, p);
            train_ConfMat = confusionmat(LabelTrain,predict_train);
            train_ClassMat = train_ConfMat./(meshgrid(countcats(categorical(LabelTrain)))');            
            train_acc = mean(diag(train_ClassMat));
            if train_acc > acc 
                acc=train_acc;
                q = Q(:,k);
                fs = k;
            end
        end
        %Stopping condition:When no more feature left to be selected
        if (fs ~= 0 && size(Q,2) ~=0)   
            %Initialize the selected feature matrix including the new selected feature
            P=[P q];  
            %Adding the index of the selected feature to the output matrix
            forwardselected =[forwardselected I(fs)];
            
            %Remove the selected feature
            Q(:,fs) = [];
            I(fs,:) = [];
        else
            flag=false;
        end
    end

     %Return column matrix
     forwardselected=forwardselected';
