%% preprocessing by substituting the missing data with their corresponding interpolation values
load('shanghai2017.mat');

X = shanghai2017;
indices = find(X == -999);

% each missing data is substituted by the order-one interpolation on its
% nearest six observed data.
for i = [1:6]
    copy_indices = indices;
    copy_indices(i) = [];
    all_indices = [1:length(X)];
    all_indices(copy_indices) = [];
    index = find(all_indices == indices(i));
    x = [];
    y = [];
    for j = [1:3,5:7]
        x = [x, all_indices(index+(j-4))];
        y = [y, X(all_indices(index+(j-4)))];
    end
    p = polyfit(x, y, 1);
    f = polyval(p, indices(i));
    X(indices(i)) = f;
end

% the first 700 are training data and the rest 300 are test data
Xtrain = X(1:700);
Xval = X(401:700);
Xpred = X(701:1000);
% plot the preprocessed data set
figure
plot(Xtrain)
hold on
idx = 700:1000;
plot(idx,[Xtrain(end);Xpred],'-')
hold off
xlabel("Data (in Hour)")
ylabel("PM 2.5 Concentration Index")
title("PM 2.5 Value of Shanghai, China in 2017")
legend(["Training data" "Test Data"])

laglist = [250:10:350];
neuronlist = [20 30 50];

Errlist = zeros(length(laglist),length(neuronlist));
sumErr = zeros(length(laglist),length(neuronlist));


iteration = 3;
for it = [1:iteration]
    j=1;
    
    for lag = laglist
        k=1;
        for neurons = neuronlist
            [Xtr,Ytr] = getTimeSeriesTrainData(Xtrain, lag);

            
            % training part and validation part

            
            % convert the data to a useful format
            ptr = con2seq(Xtr);
            ttr = con2seq(Ytr);
            
            %creation of networks
            net1=feedforwardnet(neurons,'trainlm');
                        
            %training and simulation
            net1.trainParam.epochs = 50;
            net1=train(net1,ptr,ttr); 
           
            datapredict = [];
            datapredict(1,:) = Xtrain(end-lag+1:end,:)';
            predictresult = Xtrain(end-lag+1:end,:)';
            
            for i = 1:300,
                datapredict(i,:) = predictresult(i:end);
                ptest = con2seq(datapredict(i,:)');
                tt = sim(net1, ptest);
                predictresult = [predictresult, cell2mat(tt)];
            end
                
            predictpart = predictresult(:,lag+1:end)';
            
%             figure
%             plot(Xpred);
%             hold on;
%             plot(predictpart)
%             legend('prediction','test data');
%             title(['Time series prediction results on test data of lag = ',...
%                num2str(lag), ' and neurons = ', num2str(neurons)]);
            
            err = mse(predictpart,Xpred);
            fprintf('The MSE of lag %d and neurons %d is %f \n', lag, neurons, err); 
            

  
            Errlist(j, k) = err;
            k = k + 1;
        end
        j = j + 1;
    end
    sumErr = sumErr + Errlist;
end

finErr = sumErr/iteration;




figure
for i=1:3
    hold on 
    plot([30:5:120],finerrorshang(:,i))
    xlabel('Lag number')
    ylabel('MSE')
    title('MSE vs lag number on validation set ')
end 

legend('[50 20] Neurons','[50 30]  Neurons','[50 50]  Neurons')