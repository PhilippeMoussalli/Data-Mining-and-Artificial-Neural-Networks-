load("breast.mat");


%Initialization of training set 
% xtrain = normalize(trainset)';
% ytrain = labels_train';
% xtest= normalize(testset)';
% labelstest=labels_test';
% labelstest(labelstest==-1)=0;
% 
% net = feedforwardnet(10,'traingd');
% net = train(net,xtrain,ytrain);
% ypred = sim(net,xtest);
% ypred(ypred>mean(ypred))=1;
% ypred(ypred<=mean(ypred))=0;

k =10;

%Kfold cross validation 
indices_K = crossvalind('Kfold',length(trainset),k);


neuronlist = [20];
methods = ["trainbr"];
trainset=trainset(:,[1]);
trainset = trainset';
labels_train = labels_train';


accuracy_mat = zeros(length(methods),length(neuronlist));
sensitiviy_mat = zeros(length(methods),length(neuronlist));
FPR_mat = zeros(length(methods),length(neuronlist));

       b=1;
 for neurons= neuronlist;

         j=1; 
        for method = methods;
            
            validationresult=zeros(length(trainset)/k);
            trainresult=zeros(length(trainset)/k);
            
            
            accuracy= zeros(1,k);
            sensitivity= zeros(1,k);
            FPR= zeros(1,k);
            % K fold cross validation 
            for i = 1:k
                val_idx = (indices_K == i); 
                k_idx = ~val_idx;

                %Convert each K fold for training and cross validation into useful
                %format

                labels_train(labels_train==-1)=0;
                 labels_test(labels_test==-1)=0;
                testset = testset';
                labels_test = labels_test';
                ptr = (trainset(:,k_idx)); 
                ttr = (labels_train(:,k_idx));
                pval = (trainset(:,val_idx )); 
                tval = (labels_train(:,val_idx ));
                
                net1 = feedforwardnet(neurons,method);
      
                net1.trainParam.epochs = 50;
                net1=train(net1,ptr,ttr); 
       
                tt = sim(net1, pval);
                tt (tt >mean(tt))=1;
                tt (tt <=mean(tt))=0;
                
                [acc,sens,spec] = performance(tt,tval);
                fprintf('The acc of method and neurons %d is %f \n',neurons, acc); 
                accuracy(1, i) = acc;
                sensitivity(1, i) = sens;
                FPR(1, i) = 1-spec;
            end

        accuracy_mat(j,b) = sum(accuracy)/k;
        sensitiviy_mat(j,b) = sum(sensitivity)/k;
        FPR_mat(j,b) = sum(FPR)/k;
        j = j + 1;
        
        end
        
      b = b + 1;   
 end



