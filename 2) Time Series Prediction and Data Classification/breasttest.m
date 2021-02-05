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

k = 1;

%Kfold cross validation 
indices_K = crossvalind('Kfold',length(trainset),k);


neuronlist = [20];
methods = ["trainlm"];
trainset = trainset';
labels_train = labels_train';
testset= testset';
labels_test = labels_test';


trainset1=trainset([[20,9,19,12,10,15]],:);  %[8,28,21,23,24,3,7,4,1,14,27,11,26,13]
testset1=testset([[20,9,19,12,10,15]],:); 

    accuracy_mat = zeros(length(methods),length(neuronlist));
    sensitiviy_mat = zeros(length(methods),length(neuronlist));
    FPR_mat = zeros(length(methods),length(neuronlist));

           b=1;
     for neurons= neuronlist;

             j=1; 
            for method = methods;

                validationresult=zeros(length(trainset1)/k);
                trainresult=zeros(length(trainset1)/k);


                accuracy= zeros(1,k);
                sensitivity= zeros(1,k);
                FPR= zeros(1,k);


                labels_train(labels_train==-1)=0;
                labels_test(labels_test==-1)=0;

                 ptr = (trainset1); 
                 ttr = (labels_train);
                pval = (testset1); 
                tval = (labels_test);

                    net1 = feedforwardnet(neurons,method);

                    net1.trainParam.epochs = 50;
                    net1=train(net1,ptr,ttr); 

                    tt = sim(net1, pval);
                    tt (tt >mean(tt))=1;
                    tt (tt <=mean(tt))=0;

                    [acc,sens,spec,TN,FP,FN,TP] = performance(tt,tval);
                    fprintf('The acc of method and neurons %d is %f \n',neurons, acc); 
                    accuracy(1, 1) = acc;
                    sensitivity(1, 1) = sens;
                    FPR(1, 1) = 1-spec;
                    conf_mat = [TN FP;
                                FN TP];
              end

            accuracy_mat(j,b) = sum(accuracy)/k;
            sensitiviy_mat(j,b) = sum(sensitivity)/k;
            FPR_mat(j,b) = sum(FPR)/k;
            j = j + 1;

     end



